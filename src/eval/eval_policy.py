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


import random
import os
import json
from utils.eval import *
from collections import Counter


def get_json_files(data_dir):
    files = []
    
    base_path = os.path.normpath(data_dir)
    
    for root, _, filenames in os.walk(base_path):
            for filename in filenames:
                if filename.lower().endswith('.json'):
                    abs_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(abs_path, start=base_path)
                    files.append(rel_path)
    
    random.shuffle(files)
    return files

def load_json(data_path):
    with open(data_path,"r") as f:
        data=json.load(f)
    return data

def majority_vote(files,data_dir,args):
    random.seed(32)
    all_cnt=0
    correct_cnt=0
    for filename in files:
        all_cnt+=1
        test_data_pth = os.path.join(data_dir,filename)
        test_data=load_json(test_data_pth)
        step_group=test_data['steps'][:args.size]
        pred_answers=[extract_pred(step[-1]) for step in step_group]
        counter = Counter(pred_answers)
        most_common=counter.most_common(1)[0][0]
        gold_label=test_data['extracted_groundtruth']
        try:
            if check_correctness(most_common,gold_label, args):
                correct_cnt+=1
        except:
            print(filename)
    acc=correct_cnt/all_cnt
    print("Maj@N",all_cnt,correct_cnt,acc)
def pass_1(files,data_dir,args):
    all_cnt=0
    correct_cnt=0
    random.seed(32)

    for filename in files:
        all_cnt+=1
        test_data_pth = os.path.join(data_dir,filename)
        test_data=load_json(test_data_pth)
        random_item = random.choice(test_data['steps'][:args.size])
        #print(random_item)

        gold_label=test_data['extracted_groundtruth']
        try:
            if check_correctness(random_item[-1],gold_label, args):
                correct_cnt+=1
        except:
            print(filename)
    acc=correct_cnt/all_cnt
    print("Pass@1",all_cnt,correct_cnt,acc)
def upperbound(files,data_dir,args):
    all_cnt=0
    correct_cnt=[]
    files=get_json_files(data_dir)
    must_error_lst=[]
    z=[]
    random.seed(2025)
    for filename in files:
        all_cnt+=1
        test_data_pth = os.path.join(data_dir,filename)
        test_data=load_json(test_data_pth)
        gold_label=test_data['extracted_groundtruth']
        for step in test_data['steps'][:args.size]:
            try:
                if check_correctness(step[-1],gold_label, args):
                    correct_cnt.append(filename)
            except:
                print(filename)

    correct_cnt=list(set(correct_cnt))
    must_error_lst=[x for x in files if x not in correct_cnt]
    acc=len(set(correct_cnt))/all_cnt
    print("UpperBound@N",all_cnt,len(correct_cnt),acc)

def eval_policy(data_dir,args):
    files=get_json_files(data_dir)
    majority_vote(files,data_dir,args)
    pass_1(files,data_dir,args)
    upperbound(files,data_dir,args)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Policy Model.")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--size", type=int)
    return parser.parse_args()

def main():
    args = parse_args()
    print(args.data_name)
    print(args.size)
    eval_policy(args.data_path,args)

if __name__=="__main__":
    main()

