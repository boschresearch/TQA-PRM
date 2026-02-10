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


def yes_no_stats(files,data_dir,args):
    all_cnt=0
    yes_cnt=0
    no_cnt=0
    all_score=0
    for filename in files:
        test_data_pth = os.path.join(data_dir,filename)

        test_data=load_json(test_data_pth)
        score_list=test_data["value"][:args.size]
        #steps=test_data['steps'][:args.size]
        #average_score=[sum(i)/len(steps[index]) if len(steps[index]) else 0 for index,i in enumerate(score_list)]
        for score in score_list:
            for s in score:
                all_score+=s
                all_cnt+=1
                if s > 0.5:
                    yes_cnt+=1
                elif s< 0.5:
                    no_cnt+=1

    print("ALL, YES, No Count: ",all_cnt,yes_cnt,no_cnt)
    print("Yes Percentage: ", yes_cnt/all_cnt)
    print("No Percentage: ", no_cnt/all_cnt)
    print("ALL scores: ", all_score)
    print("AVG(scores): ", all_score/all_cnt)
def F1(files,data_dir,args):
    all_cnt=0

    tp,fp,fn,tn=0,0,0,0
    error_lst=[]
    correct_lst=[]
    for filename in files:
        
        test_data_pth = os.path.join(data_dir,filename)

        test_data=load_json(test_data_pth)
        score_list=test_data["value"][:args.size]
        steps=test_data['steps'][:args.size]
        gold_label=test_data['extracted_groundtruth']
        for score,stepset in zip(score_list,steps):
            try:
                all_cnt+=1
                if score[-1]>0.5 and check_correctness(stepset[-1],gold_label, args):
                    tp+=1
                elif score[-1] < 0.5 and check_correctness(stepset[-1],gold_label, args):
                    fn+=1
                elif score[-1]>0.5 and not check_correctness(stepset[-1],gold_label, args):
                    fp+=1
                elif score[-1] < 0.5 and not check_correctness(stepset[-1],gold_label, args):
                    tn+=1
                else:
                    continue
            except:
                continue


    print("TP, TP_rate: ", tp, tp/all_cnt)
    print("TN, TN_rate: ", tn, tn/all_cnt)
    print("FP, FP_rate: ", fp, fp/all_cnt)
    print("FN: FN_rate: ", fn, fn/all_cnt)

def answer_bias(files,data_dir,args): ############Policy Model answer distribution
    all_cnt=0

    true_answer, false_answer = 0, 0
    error_lst=[]
    correct_lst=[]
    for filename in files:
        
        test_data_pth = os.path.join(data_dir,filename)

        test_data=load_json(test_data_pth)
        steps=test_data['steps'][:args.size]
        gold_label=test_data['extracted_groundtruth']
        for stepset in steps:
            try:
                all_cnt+=1
                if check_correctness(stepset[-1],gold_label, args):
                    true_answer +=1
                elif not check_correctness(stepset[-1],gold_label, args):
                    false_answer+=1
                else:
                    continue
            except:
                continue


    print("True_answer, T_rate: ", true_answer, true_answer/all_cnt)
    print("False_answer, F_rate: ", false_answer, false_answer/all_cnt)


def eval(data_dir,args):
    files=get_json_files(data_dir)
    yes_no_stats(files,data_dir,args)
    F1(files,data_dir,args)
    #answer_bias(files,data_dir,args)
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation PRM")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--size",type=int)
    return parser.parse_args()

def main():
    args = parse_args()
    print("Task: ", args.data_name)
    print("Data Size: ", args.size)
    eval(args.data_path,args)

if __name__=="__main__":
    main()

