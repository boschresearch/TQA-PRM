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


#####################################################           import packeges and args             ########################################################

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
import argparse
import json
import random
import time
import threading
import traceback
from utils.util import *
from copy import *
from datasets import Dataset, load_from_disk
from prm_evaluation.genprm_inference import GenPRM

os.environ['VLLM_USE_V1'] = '0'

version = 'v2.3'

TIME_LIMIT = 300  # set time limit
stop_event = threading.Event()

def heart_beat_worker(file_path):
    start_time = time.time()

    while not stop_event.is_set():
        if os.path.exists(file_path):
            try:
                os.utime(file_path)
                timestamped_print(f"Heartbeat updated: {file_path}")
            except Exception as e:
                timestamped_print(f"Update file time error: {str(e)}", 'ERROR')
        else:
            try:
                with open(file_path, 'w') as f:
                    pass
                timestamped_print(f"Created file while heart beating: {file_path}", 'ERROR')
            except Exception as e:
                timestamped_print(f"Create file error: {str(e)}", 'ERROR')

        for _ in range(6):
            if stop_event.is_set():
                timestamped_print("Heartbeat worker exiting...")
                return
            time.sleep(5)


def parse_args():
    parser = argparse.ArgumentParser(description="Process data with optional generation config.")
    parser.add_argument("--reward_name_or_path", type=str, help="Path to the reward model or data.")
    parser.add_argument("--data_path", type=str, default="/home/tal1rng/GenPRM/src/_output/wtq_eval.jsonl_train_a09a35458c702b33eeacc393d103063234e8bc28_PRM-Data", help="Path to the input data.")
    parser.add_argument("--split_out", type=str, help="Path to the output data.")
    parser.add_argument("--analyze", action='store_true', help='analyze or not')
    parser.add_argument("--analyze_template_E2", type=str, default="<analyze>\nLet' analyze the Paragraph {cur_step} step by step across five aspects:\nPart 1: **Restatement**: ")
    parser.add_argument("--analyze_template", type=str, default="<analyze>\nLet's analyze the Paragraph {cur_step} step by step: ")
    parser.add_argument("--output_template", type=str, default="<output>\n**Judgement**: $\\boxed")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--idd", type=int, default=1)
    parser.add_argument("--experiment", type=str, default="E1", help="Choose the Experiment")
    return parser.parse_args()


args = parse_args()
print_args(
    args,
    program_name="prm_evaluate",
    version=version
)

#####################################################           model load with VLLM             ########################################################

genprm = GenPRM(args.reward_name_or_path, args.tensor_parallel_size)

#####################################################           load splited dataset             ########################################################

random.seed(int(time.time()))


def get_shuffled_folders(directory):
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    random.shuffle(folders)
    return folders


target_list = get_shuffled_folders(args.data_path)

for data_path in target_list:
    folder_name = os.path.basename(data_path)
    save_path = os.path.join(args.split_out, folder_name)

    if args.analyze:
        save_path += '_analyze'

    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
            timestamped_print(f"create folder: {save_path}")
        except Exception as e:
            timestamped_print(f"Error: {e}", 'ERROR')
            continue
    else:
        if not os.listdir(save_path):
            creation_time = os.path.getctime(save_path)
            current_time = time.time()
            if (current_time - creation_time) > TIME_LIMIT:
                os.makedirs(save_path, exist_ok=True)
                timestamped_print(f"create folder again: {save_path}", 'ERROR')
            else:
                timestamped_print(f"skip: {save_path} (not exceed time limit)")
                continue
        else:
            timestamped_print(f"skip: {save_path} (not empty)")
            continue

    stop_event.clear()
    thread = threading.Thread(target=heart_beat_worker, args=(save_path,))
    thread.daemon = True
    thread.start()
    timestamped_print("Heartbeat thread started. Main thread continues...")

    with open(os.path.join(args.data_path, folder_name, 'record_0.json'), 'r') as f:
        data_new = json.load(f)
    sample = deepcopy(data_new)
    all_paths = sample['steps'][:8]
    conversations=[]
    scores=[]
    start = time.perf_counter()
    if args.experiment=="E2":
        args.analyze_template=args.analyze_template_E2
    for candidate_index, reasoning_steps in enumerate(all_paths):
        try:
            reasoning_steps[0] = sample['problem'] + '\n' + reasoning_steps[0]
            if reasoning_steps and reasoning_steps[-1] == '':
                reasoning_steps.pop()
            if args.analyze:
                message = {
                    'conversation': [
                 {'role': 'system', 'content': 'You are a Table Question Answering Expert. Your task is to review and critique the paragraphs in solution step by step.'}
                ]
                }
            else:
                message = {
                    'conversation': [
                {'role': 'system', 'content': 'You are a Table Question Answering Expert. Your task is to review and critique the paragraphs in solution directly. Output your judgement in the format of `\\boxed{Yes}` if the paragraph is correct, or `\\boxed{No}` if the paragraph is incorrect.'}
                ]
                }
            line_length=len(reasoning_steps)
            for j1 in range(len(reasoning_steps)):
                line = {'role': 'user', 'content': reasoning_steps[j1]}
                print("j1 and line is", j1, line)
                message['conversation'].append(line)
                line = {'content': '', 'role': 'assistant'}
                message['conversation'].append(line)
                print("message", message)
            timestamped_print(message)
            try:
                conversation = message['conversation']
                print("conversation", conversation)
                step_scores = []
                cur_step = 0
                
                for step_index, mm in enumerate(conversation):
                    role = mm.get('role', '').lower()
                    if role == 'user' or role == 'system':
                        continue

                    paths = conversation[:step_index]
                    cur_step += 1
                    if cur_step==line_length:
                        outputs, reward = genprm.inference(
                        messages=paths,
                        majority_num=1,
                        cur_step=cur_step,
                        analyze=args.analyze,
                        time_limit=3,
                        max_tokens=2048,
                        analyze_template=args.analyze_template,
                        output_template=args.output_template,
                        logging=True,
                        )
                    else:
                        outputs, reward = genprm.inference(
                        messages=paths,
                        majority_num=1,
                        cur_step=cur_step,
                        analyze=args.analyze,
                        time_limit=3,
                        max_tokens=2048,
                        analyze_template=args.analyze_template,
                        output_template=args.output_template,
                        logging=True,
                        )

                    conversation[step_index] = {
                        'role': 'assistant',
                        'content': outputs[0]
                    }
                    step_scores.append(reward)
            except Exception as e:
                traceback.print_exc()
        except Exception as e:
            traceback.print_exc()
        conversations.append(conversation)
        scores.append(step_scores)
    end = time.perf_counter()
    data_new['time'] = end - start
    data_new['value'] = scores
    data_new['conversations'] = conversations

    timestamped_print(type(data_new))
    with open(os.path.join(save_path, f'result_{folder_name}.json'), 'w') as f:
            json.dump(data_new, f, indent=4)
    timestamped_print(f"dataset has been saved to: {save_path}")


    stop_event.set()
    thread.join(timeout=5)

    if thread.is_alive():
        timestamped_print("Warning: Heartbeat thread did not exit cleanly!", 'ERROR')
    else:
        timestamped_print("Heartbeat thread exited successfully")
