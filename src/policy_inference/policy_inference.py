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


# math_shepherd_ray.py

import argparse
import copy
import json
import os
import sys
from utils.eval import *

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
os.makedirs("bugs", exist_ok=True)

import random
import time
from typing import List, Optional

import numpy as np
import ray
from ray.util import ActorPool
from datetime import datetime
import threading

from inference_utils import (
    load_data, load_json, save_json, is_file_exists, #check_correctness,
    format_query, post_process_act, assign_tasks, check_question_finished, 
    get_current_save_idx, check_process_cnt, check_lock_timeout, create_empty_file
)
from utils.vllm_worker import VLLMRemoteCaller
from utils.parse_utils import parse_ground_truth, extract_answer

os.environ['VLLM_USE_V1'] = '1'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

stop_event = threading.Event()

def heart_beat_worker(chosen_idxs, save_dir, lock_dir):
    while not stop_event.is_set():
        for (i, j) in chosen_idxs:
            if stop_event.is_set():  # Check stop event more frequently
                print('==============heartbeat worker exit==============')
                return
            lock_file_path = os.path.join(save_dir, f"{lock_dir}/question_{i}_{j}.lock")
            try:
                if is_file_exists(lock_file_path):
                    os.utime(lock_file_path)
                    print(f"Update lock file {lock_file_path}.")
                else:
                    create_empty_file(lock_file_path)
                    print(f"Create lock file {lock_file_path}.")
            except Exception as e:
                print(f"Error in heartbeat worker: {str(e)}")
        # Use shorter sleep intervals with stop event checks
        for _ in range(60):  # 60 * 5 = 300 seconds total
            if stop_event.is_set():
                print('==============heartbeat worker exit==============')
                return
            time.sleep(5)
    print('==============heartbeat worker exit==============')


class Generator:
    def __init__(self, args, llm):
        self.args = args
        self.llm = llm

    def generate_step_by_step_paths(self, problem, num_samples_list, temperature_list):
        ########generating step-by-step-paths for each problem
        messages = format_query(self.args.cot_prompt, problem, 'Step 1: ')

        inference_paths = []
        assert len(num_samples_list) == len(temperature_list)
        for n, temperature in zip(num_samples_list, temperature_list):  # Generate inference paths for each sample size and temperature
            temp_paths, finish_reasons = self.llm.generate_fastchat(messages, n=n, temperature=temperature)
            for idx, (path, finish_reason) in enumerate(zip(temp_paths, finish_reasons)):
                if finish_reason != 'stop':
                    step_list = ['<INVALID>']
                else:
                    step_list = post_process_act(path)
                    if len(step_list) == 1:
                        step_list = ['<INVALID>']
                inference_paths.append(step_list)
            print(f"Generated {len(temp_paths)} inference paths for n={n}, temperature={temperature}.")

        return inference_paths


    def inference_problem(self, data):
        # not only monte carlo

        problem = data['problem']
        extracted_groundtruth = data['extracted_groundtruth']
        meta_data = data['meta_data']

        inference_paths = self.generate_step_by_step_paths(problem, self.args.num_samples_list, self.args.temperature_list)
        random.shuffle(inference_paths)
        print(f'Randomly shuffle the inference paths.')



        data['steps'] = inference_paths

        return data


@ray.remote
class RemoteGenerator(Generator):
    def __init__(self, args, llm):
        super().__init__(args, llm)


def process_dataset(args, raw_test_ds, actor_pool, question2id, question_mask):
    results = []

    test_ds, chosen_dict, chosen_idxs, save_dir, lock_dir = assign_tasks(
        raw_test_ds, question_mask, args.num_paths, args.save_dir, args.lock_dir, args.round, args.eager, args.batch_size, args.max_time
    )
    if len(test_ds) == 0:
        sys.exit(0)
    
    print('open thread to update lock file')
    stop_event.clear()
    thread = threading.Thread(target=heart_beat_worker, args=(chosen_idxs, save_dir, lock_dir), daemon=True)
    thread.start()

    res_q = actor_pool.map_unordered(lambda p, x: p.inference_problem.remote(x), test_ds)
    
    start_time = time.time()
    last_time = start_time

    for i, (data) in enumerate(res_q):
        q_idx = question2id[data["problem"]]
        question_path = os.path.join(args.save_dir, f"question_{q_idx}")
        os.makedirs(question_path, exist_ok=True)

        try:
            idx = chosen_dict[q_idx][0]
            if not is_file_exists(os.path.join(question_path, f"record_{idx}.json")):
                chosen_dict[q_idx].pop(0)
            else:
                print(f"Prepare to save but file exists: {os.path.join(question_path, f'record_{idx}.json')}")
                idx = get_current_save_idx()
        except Exception as e:
            print(f"Error: {e}")
            idx = get_current_save_idx()
        save_json(data, os.path.join(question_path, f"record_{idx}.json"))

        temp_time = time.time()
        delta_time = temp_time - last_time
        total_time = temp_time - start_time
        time_str = datetime.now().strftime("%y-%m-%d %H:%M:%S")
        last_time = temp_time

        try:
            success_flag, flag, file_cnt = check_question_finished(question_path, args.num_paths, args.save_dir, args.lock_dir, args.round, detailed=True)
            check_lock_timeout(raw_test_ds, args.save_dir, args.lock_dir, args.round, args.max_time)
            cnt = check_process_cnt(raw_test_ds, args.save_dir)
            total = len(raw_test_ds)
            print(
                f"[{time_str}]   Cnt: {i + 1:>3} / {len(test_ds):>3}  |  Q: {q_idx:>3}  |  Idx: {idx:>3}  |  "
                f"Del T: {delta_time:>6.1f}s  |  Tot T: {total_time:>7.1f}s  |  Avg T: {total_time / (i + 1):>6.1f}s/it  |  "
                f"Pct: {cnt:>5} / {total:>5} = {cnt / total * 100:.2f}%"
            )
        except Exception as e:
            print(
                f"[{time_str}]   Cnt: {i + 1:>3} / {len(test_ds):>3}  |  Q: {q_idx:>3}  |  Idx: {idx:>3}  |  "
                f"Del T: {delta_time:>6.1f}s  |  Tot T: {total_time:>7.1f}s  |  Avg T: {total_time / (i + 1):>6.1f}s/it  |  "
            )
            print(f"Error: {e}")
    
    # Only stop heartbeat after all tasks are complete
    print('==============debug==============')
    stop_event.set()
    thread.join(timeout=10)
    if thread.is_alive():
        print("Warning: Heartbeat thread did not exit cleanly!")
    else:
        print("Heartbeat thread exited successfully")

    return results

def print_args(args: argparse.Namespace, 
              program_name: str = None,
              version: str = None,
              show_version: bool = True) -> None:
    '''
    print the args settings
    '''
    args_dict = {k: v for k, v in vars(args).items() if not k.startswith('_')}
    
    max_len = max(len(str(k)) for k in args_dict.keys())
    sep = '-' * (max_len + 20)
    
    output = []
    if program_name:
        output.append(f"\n\033[1;36m{program_name}\033[0m")
        
    if version and show_version:
        output.append(f"\033[1;34mVersion:\033[0m \033[1;33m{version}\033[0m")
    
    output.append(f"\033[1;35m{sep}\033[0m")
    
    for k, v in sorted(args_dict.items()):
        key = f"\033[1;32m{k:>{max_len}}\033[0m"
        val = f"\033[1;37m{str(v)}\033[0m"
        output.append(f"{key} : {val}")
    
    output.append(f"\033[1;35m{sep}\033[0m\n")
    
    print('\n'.join(output))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_path', '-b', type=str, default=root_dir, help="Path to the JSON file containing problem-answer pairs.")
    parser.add_argument('--data_dir', type=str, default='datasets', help="Path to the JSON file containing problem-answer pairs.")
    # parser.add_argument('--origin_data_dir', type=str, default='data', help="Copy of Path to the JSON file containing problem-answer pairs.")
    parser.add_argument('--origin_model_path', type=str, help="(Only used in montecarlo)The base model's Path")
    parser.add_argument('--special_data_dir', action="store_true")
    parser.add_argument('--data_name', '-d', type=str, default='wtq_train')

    parser.add_argument('--split', type=str, default='train-test', help="Path to the JSON file containing problem-answer pairs.")
    parser.add_argument('--model_path', '-m', type=str, help="Path to the pre-trained model.")
    parser.add_argument('--num_paths', type=int, default=32, help="Number of inference paths to collect per problem.")
    parser.add_argument('--num_samples_list', type=list, default=[32], help="Number of samples to generate per problem.")
    parser.add_argument('--temperature_list', type=list, default=[0.7], help="Temperature for sampling completions.")
    parser.add_argument('--save_dir', type=str, default='_output_scitab_Qwen7b_32paths', help="Path to save the output JSON file.")
    parser.add_argument('--add_sleep', '-s', action='store_true', help="Reverse the order of inference steps.")
  
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--add_step_prompt", action="store_true")
    parser.add_argument("--apply_chat_template", type=bool, default=True)
    parser.add_argument("--serve_type", type=str, default="fastchat", choices=["vllm_model", "vllm_api", "fastchat", "sgl_api"])
    parser.add_argument("--cot_prompt", type=str, default="")
    parser.add_argument("--llm_step_tag", type=str, default="\nStep ")

    # parallel config
    parser.add_argument("--controller_addr", type=str, default="http://localhost:21001")
    parser.add_argument("--worker_addr", type=str, default="http://localhost:21002")
    parser.add_argument("--num_worker", type=int, default=16)
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument("--disable_ray", action="store_true")
    parser.add_argument("--round", '-r', type=int, default=0)
    parser.add_argument("--lock_dir", type=str, default="lock_dir")
    parser.add_argument("--eager", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--max_time", type=int, default=0)
    parser.add_argument("--loop", type=int, default=0)

    args = parser.parse_args()
    # patch
    # args.batch_size = args.batch_size ** 2

    cot_prompt_dict = {
        'llama_official': """Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.""",
        # 'llama': """Please reason step by step, and put your final answer within \\boxed{}.""",
        # 'ministral': """Please reason step by step, and put your final answer within \\boxed{}.""",
        # 'gemma': """Please reason step by step, and put your final answer within \\boxed{}.""",
        # 'qwen': """Please reason step by step, and put your final answer within \\boxed{}.""",
        'skywork-o1': """You are Skywork-o1, a thinking model developed by Skywork AI, specializing in solving complex problems involving mathematics, coding, and logical reasoning through deep thought. When faced with a user's request, you first engage in a lengthy and in-depth thinking process to explore possible solutions to the problem. After completing your thoughts, you then provide a detailed explanation of the solution process in your response.""",
        'qwq': """You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.""",
        'tabfact': """Please inspect the table(s) and then provide a True or False answer to the question.  Please reason step by step. Attention: You MUST put your final answer using the following format: Final answer: \\boxed{True/False}, Attention: You MUST put your final answer using the following format: Final answer: \\boxed{True/False}.""",
        #'default': """Please inspect the table(s) and then provide a True or False answer to the question.  Please reason step by step. Attention: You MUST put your final answer using the following format: Final answer: \\boxed{True/False}, Attention: You MUST put your final answer using the following format: Final answer: \\boxed{True/False}."""
    }
    if 'llama-3' in args.model_path.lower() and 'skywork-o1' not in args.model_path.lower():  # TODO: Check llama
        # args.cot_prompt = cot_prompt_dict['llama_official']
        args.cot_prompt = cot_prompt_dict['default']
    elif 'ministral' in args.model_path.lower():
        args.cot_prompt = cot_prompt_dict['default']
    elif 'gemma' in args.model_path.lower():
        args.cot_prompt = cot_prompt_dict['default']
    elif 'qwen' in args.model_path.lower():
        args.cot_prompt = cot_prompt_dict['default']
    elif 'skywork-o1-open-llama' in args.model_path.lower():
        args.cot_prompt = cot_prompt_dict['skywork-o1']
        args.add_step_prompt = False  # TODO: Check
    elif 'qwq' in args.model_path.lower():
        args.cot_prompt = cot_prompt_dict['qwq']
        args.add_step_prompt = False  # TODO: Check
    elif "tabfact" in args.data_name:
        args.cot_prompt = cot_prompt_dict['tabfact']
    else:
        args.cot_prompt = cot_prompt_dict['default']

    args.origin_save_dir = args.save_dir

    args.save_dir = os.path.join(args.base_path, args.origin_save_dir, f'{args.data_name}_{args.split}_{args.model_path.split("/")[-1].split("--")[-1]}_PRM-Data')
    print("###############################################",args.save_dir,"###############################################")
    print("###############################################",args.base_path,"###############################################")
    print("###############################################",args.origin_save_dir,"###############################################")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, args.lock_dir), exist_ok=True)

    llm = VLLMRemoteCaller(args)
    raw_test_ds = load_data(args)
    question2id = {problem_inst["problem"]: i for i, problem_inst in enumerate(raw_test_ds)}
    question_mask = [True] * len(raw_test_ds)

    actor_pool = ActorPool([RemoteGenerator.remote(args, llm) for _ in range(args.num_worker)])

    while True:
        process_dataset(args, raw_test_ds, actor_pool, question2id, question_mask)


if __name__ == "__main__":
    main()
