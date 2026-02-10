""" Utility classes and functions related to TQA-PRM (EACL 2026).
Copyright (c) 2026 Robert Bosch GmbH
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
butWITHOUT ANYWARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""


import re
import math
import io
import signal
from copy import *
from utils.util import *
from contextlib import redirect_stdout
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
REPETITION_PENALTY = 1.0
version = 'v1.0'


class timeout:
    """timeout context manager"""

    def __init__(self, seconds=1):
        self.seconds = seconds

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)

    def handle_timeout(self, signum, frame):
        raise TimeoutError("Code execution timed out")

class GenPRM:
    def __init__(self, model_path, tensor_parallel_size=1):
        # Load the model and tokenizer
        timestamped_print(f"Loading model from {model_path}", level="INFO")

        self.model = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                enable_chunked_prefill=True
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        timestamped_print(f"GenPRM loaded successfully", level="INFO")

    def get_reward_score(self, out):
        '''calculate the reward score'''
        generated_text = out.text
        logprobs = out.logprobs
        tokens = out.token_ids
        token_logprobs = logprobs

        # find the position of Yes/No token
        boxed_match = re.search(r'(Yes|No)\}', generated_text, re.IGNORECASE)
        yes_token = self.tokenizer.encode('Yes')[-1]
        no_token = self.tokenizer.encode('No')[-1]

        if boxed_match:
            decision = boxed_match.group(1).capitalize()
            if decision == "Yes":
                yes_index = len(tokens) - 1 - tokens[::-1].index(yes_token)
                yes_logprob = token_logprobs[yes_index][yes_token].logprob
                # convert logprob to probability
                yes_prob = math.exp(yes_logprob)  # e^log(prob) = prob

                # find the position of 'No' token
                try:
                    no_logprob = token_logprobs[yes_index][no_token].logprob
                    no_prob = math.exp(no_logprob)
                except KeyError:
                    # set 'No' probability to the minimum logprob of the remaining 4 logprobs
                    min_logprob = min(v.logprob for k, v in token_logprobs[yes_index].items())
                    no_prob = math.exp(min_logprob)

                # calculate softmax value
                softmax_denominator = yes_prob + no_prob
                if softmax_denominator == 0:
                    softmax_yes = 0.5  # in case of division by zero, assign neutral score
                else:
                    softmax_yes = yes_prob / softmax_denominator

                return softmax_yes

            elif decision == "No":
                no_index = len(tokens) - 1 - tokens[::-1].index(no_token)
                no_logprob = token_logprobs[no_index][no_token].logprob
                # convert logprob to probability
                no_prob = math.exp(no_logprob)  # e^log(prob) = prob

                # find the position of 'Yes' token
                try:
                    yes_logprob = token_logprobs[no_index][yes_token].logprob
                    yes_prob = math.exp(yes_logprob)
                except KeyError:
                    # set 'Yes' probability to the minimum logprob of the remaining 4 logprobs
                    min_logprob = min(v.logprob for k, v in token_logprobs[no_index].items())
                    yes_prob = math.exp(min_logprob)

                # calculate softmax value
                softmax_denominator = yes_prob + no_prob
                if softmax_denominator == 0:
                    softmax_yes = 0.5  # in case of division by zero, assign neutral score
                else:
                    softmax_yes = yes_prob / softmax_denominator

                return softmax_yes
        else:
            # return neutral score if no decision found
            timestamped_print("No boxed{Yes/No} found in the output", level="WARNING")
            return 0.5

    def inference(
        self,
        messages,
        majority_num=1,
        cur_step=1,
        analyze=True,
        time_limit=3,
        max_tokens=2048,
        analyze_template="<analyze>\nLet's analyze the Paragraph {cur_step} step by step: ",
        #analyze_template="<analyze>\nLet' travere the given table iteratively and then analyze the Paragraph {cur_step} step by step across five aspects:\nPart 1: **Restatement**: ",
        output_template="<output>\n**Judgement**: $\\boxed",
        logging=True
    ):
        '''
        messages: the input messages
        majority_num: the number of majority votes
        cur_step: the current step index (start from 1)
        analyze: whether to analyze the input
        verify: whether to verify the input
        execute: whether to execute the code
        time_limit: the time limit for code execution
        max_tokens: the maximum tokens for the output
        analyze_template: the template for analyze start
        verify_template: the template for verify start
        output_template: the template for output start
        logging: whether to log the process
        '''
        output_paths = []
        reward_scores = []
        for i in range(majority_num):
            # perform inference
            output_path, reward_score = self._single_inference(
                messages,
                cur_step=cur_step,
                analyze=analyze,
                time_limit=time_limit,
                max_tokens=max_tokens,
                analyze_template=analyze_template,
                output_template=output_template,
                logging=logging
            )

            output_paths.append(output_path)
            reward_scores.append(reward_score)

        return output_paths, sum(reward_scores) / len(reward_scores)

    def _single_inference(
        self,
        messages,
        cur_step=1,
        analyze=True,
        time_limit=3,
        max_tokens=2048,
        analyze_template="<analyze>\nLet's analyze the Paragraph {cur_step} step by step across five aspects:\nPart 1: **Restatement**: ",
        #analyze_template="<analyze>\nLet's analyze the Paragraph {cur_step} step by step: ",
        output_template="<output>\n**Judgement**: $\\boxed",
        logging=True
    ):
        context = {"cur_step": cur_step}
        analyze_start = analyze_template.format(**context)
        output_start = output_template.format(**context)
        # Prepare the input
        prompt = build_prompt(messages, self.tokenizer)

        # Generate the output
        # Stage 1
        if analyze:
            sampling_params = SamplingParams(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                stop=['</analyze>\n','</analyze>'], 
                include_stop_str_in_output=True,
                max_tokens=max_tokens,
                logprobs=20,  
                repetition_penalty=REPETITION_PENALTY
            )
            if logging:
                cprint(prompt + analyze_start, f'paragraph {cur_step} request 1')
            output1 = self.model.generate(prompt + analyze_start, sampling_params=sampling_params, use_tqdm=False)[0].outputs[0]
            cur_prompt = analyze_start + output1.text + output_start  # directly generate <output> if verify is False
        else:
            cur_prompt = prompt+output_start
        # Stage 2
        cur_time = 0
        sampling_params = SamplingParams(
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        top_k=TOP_K,
                        stop=['</output>\n','</output>'],  # set the stop string
                        include_stop_str_in_output=True,  # include the stop string in the output
                        max_tokens=64,  # Maximum number of tokens to generate
                        logprobs=20,  # Number of log probabilities to return
                        repetition_penalty=REPETITION_PENALTY,
                    )
        if logging:
            cprint(cur_prompt, f'paragraph {cur_step} request {cur_time + 2}')

        output2 = self.model.generate(cur_prompt, sampling_params, use_tqdm=False)[0].outputs[0]

        output2.text = cur_prompt + output2.text

        # extract the Probability of Yes token as the reward score
        reward_score = self.get_reward_score(output2)

        return output2.text, reward_score
