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
import random
import numpy as np
random.seed(2025)
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


def best_of_N(files,data_dir,args):
    all_cnt=0
    correct_cnt=0
    error_lst=[]
    correct_lst=[]
    for filename in files:
        all_cnt+=1
        test_data_pth = os.path.join(data_dir,filename)

        test_data=load_json(test_data_pth)
        score_list=test_data["value"][:args.size]
        steps=test_data['steps'][:args.size]
        average_score=[sum(i)/len(steps[index]) if len(steps[index]) else 0 for index,i in enumerate(score_list)]
        best_i = average_score.index(max(average_score))
        best_solution=test_data['steps'][best_i]
        gold_label=test_data['extracted_groundtruth']
        try:
            if check_correctness(best_solution[-1],gold_label, args):
                correct_cnt+=1
                correct_lst.append(filename)
            else:
                error_lst.append(filename)
                
        except:
            continue
            
    acc=correct_cnt/all_cnt
    print("best@N",all_cnt,correct_cnt,acc)


def bin_frequencies(scores):
    edges = [0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1.00]
    #edges=[0.975, 0.980, 0.985, 0.990, 0.995, 1.000]
    freq = [0] * (len(edges)-1)

    for x in scores:
        if 0.9 <= x <= 1:
            # Half-open bins: [low, high), last bin closed: [0.975, 1.0]
            if x == 1.0:
                freq[-1] += 1
            else:
                # find the first high edge strictly greater than x
                for i in range(len(edges) - 1):
                    if x < edges[i+1]:
                        freq[i] += 1
                        #print(freq)
                        break
    return freq
def best_of_N_bins(files,data_dir,args,bins):
    #edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    edges = [0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1.00]
    #edges=[0.975, 0.980, 0.985, 0.990, 0.995, 1.000]
    score_stat=[]
    bin_correct_cnt=[]
    for filename in files:
        test_data_pth = os.path.join(data_dir,filename)

        test_data=load_json(test_data_pth)
        score_list=test_data["value"][:args.size]
        steps=test_data['steps'][:args.size]
        #average_score=[sum(i)/len(steps[index]) if len(steps[index]) else 0 for index,i in enumerate(score_list)]
        gold_label=test_data['extracted_groundtruth']
        try:
            for i, step in enumerate(steps):
                average_score= sum(score_list[i])/len(step) if len(step) else 0
                #print(average_score)
                score_stat.append(average_score)
                if check_correctness(step[-1],gold_label,args):
                    bin_correct_cnt.append(average_score)
        except:
            continue
    freq=bin_frequencies(score_stat)   
    correct_freq=bin_frequencies(bin_correct_cnt)
    fre=[x/y if y!=0 else 0 for x,y in zip(correct_freq, freq)]
    print("edge: ", edges)
    print("Freq: ", freq)
    print("Correct_Freq: ", correct_freq)
    print("frequency is: ", fre)
def lscr(files,data_dir,args):
    all_cnt=0
    correct_cnt=0
    error_lst=[]
    correct_lst=[]
    for filename in files:
        all_cnt+=1
        test_data_pth = os.path.join(data_dir,filename)

        test_data=load_json(test_data_pth)
        score_list=test_data["value"][:args.size]
        steps=test_data['steps'][:args.size]
        best_i=random.choice([i for i in range(args.size)])
        try:
            for index,i in enumerate(score_list):
                if i[-1]==1:
                    best_i=index
        except:
            continue
        best_solution=test_data['steps'][best_i]
        gold_label=test_data['extracted_groundtruth']
        try:
            if check_correctness(best_solution[-1],gold_label, args):
                correct_cnt+=1
                correct_lst.append(filename)
            else:
                error_lst.append(filename)
                
        except:
            continue

    acc=correct_cnt/all_cnt


    print("lscr",all_cnt,correct_cnt,acc)

def lscr_best(files,data_dir,args):
    all_cnt=0
    correct_cnt=0
    error_lst=[]
    correct_lst=[]
    for filename in files:
        all_cnt+=1
        test_data_pth = os.path.join(data_dir,filename)

        test_data=load_json(test_data_pth)
        score_list=test_data["value"][:args.size]
        steps=test_data['steps'][:args.size]
        
        candidates=[]
        try:
            for index,i in enumerate(score_list):
                if i[-1]>=0.9:
                    candidates.append(index)
        except:
            continue
        if candidates:
            average_score=[sum(score_list[index])/len(steps[index]) if len(steps[index]) else 0 for _,index in enumerate(candidates)] 
            best_i = candidates[average_score.index(max(average_score))]
        else:
            best_i=random.choice([i for i in range(args.size)])
        best_solution=test_data['steps'][best_i]
        gold_label=test_data['extracted_groundtruth']
        
       
        try:
            if check_correctness(best_solution[-1],gold_label, args):
                correct_cnt+=1
                correct_lst.append(filename)
            else:
                error_lst.append(filename)
                
        except:
            continue

    acc=correct_cnt/all_cnt

    print("lscr_best",all_cnt,correct_cnt,acc)

def lscr_SC(files,data_dir,args):
    all_cnt=0
    correct_cnt=0
    error_lst=[]
    correct_lst=[]
    for filename in files:
        all_cnt+=1
        test_data_pth = os.path.join(data_dir,filename)

        test_data=load_json(test_data_pth)
        score_list=test_data["value"][:args.size]
        steps=test_data['steps'][:args.size]
        candidates=[]
        
        #best_i=random.choice([i for i in range(args.size)])
        try:
            for index,i in enumerate(score_list):
                if i[-1]==1:
                    candidates.append(index)

        except:
            continue
        if candidates:    
            pred_answers=[extract_pred(steps[index][-1]) for index in candidates]
        else:
            pred_answers=[extract_pred(step[-1]) for step in steps]
        #print(pred_answers)
        counter = Counter(pred_answers)
        most_common=counter.most_common(1)[0][0]
        gold_label=test_data['extracted_groundtruth']
        try:
            if check_correctness(most_common,gold_label, args):
                correct_cnt+=1
                correct_lst.append(filename)
            else:
                error_lst.append(filename)
                
        except:
            continue

    acc=correct_cnt/all_cnt


    print("LSCR_SC",all_cnt,correct_cnt,acc)


def select(files,data_dir,args):

    error_lst=[]
    correct_lst=[]
    for filename in files:
        test_data_pth = os.path.join(data_dir,filename)

        test_data=load_json(test_data_pth).copy()
        score_list=test_data["value"][:args.size]
        steps=test_data['steps'][:args.size]
        candidates=[]
        
        #best_i=random.choice([i for i in range(args.size)])
        try:
            for index,i in enumerate(score_list):
                if i[-1]==1:
                    candidates.append(index)

        except:
            continue
        if candidates:    
            test_data['steps']=[test_data['steps'][i] for i in candidates]
            test_data['value']=[test_data['value'][i] for i in candidates]
            test_data['conversations']=[test_data['conversations'][i] for i in candidates]
        
            os.makedirs(os.path.dirname(os.path.join("/home/tal1rng/GenPRM/src/EACL/RQ_LSCR/wtq", filename)), exist_ok=True)
            with open(os.path.join("/home/tal1rng/GenPRM/src/EACL/RQ_LSCR/wtq", filename), "w") as f:
                    json.dump(test_data, f, indent=2)

def all_correct_pth(files,data_dir,args):
    #random.seed(32)
    all_cnt=0
    correct_cnt=0
    for filename in files:

        test_data_pth = os.path.join(data_dir,filename)
        test_data=load_json(test_data_pth)
        score_list=test_data["value"][:args.size]
        steps=test_data['steps'][:args.size]
        average_score=[sum(i)/len(steps[index]) if len(steps[index]) else 0 for index,i in enumerate(score_list)]
        if max(average_score)==1:
            all_cnt+=1
            all_correct_i = average_score.index(max(average_score))

            best_solution=test_data['steps'][all_correct_i]
            gold_label=test_data['extracted_groundtruth']
            try:
                if check_correctness(best_solution[-1],gold_label, args):
                    correct_cnt+=1
            except:
                continue
                #print("Paths maybe problematic:",filename)
        else:
            continue
    acc=correct_cnt/all_cnt
    print(f"total files: {len(files)}","\n","with all correct path:", all_cnt, "\n","Acc:", correct_cnt,acc)
def top_4_SC(files,data_dir,args):
    all_cnt=len(files)
    correct_cnt=0
    for filename in files:

        test_data_pth = os.path.join(data_dir,filename)
        test_data=load_json(test_data_pth)
        score_list=test_data["value"][:args.size]
        gold_label=test_data['extracted_groundtruth']
        steps=test_data['steps'][:args.size]
        average_score=[sum(i)/len(steps[index]) if len(steps[index]) else 0 for index,i in enumerate(score_list)]
        top4_indices=sorted(range(len(average_score)),key=lambda i: average_score[i], reverse=True)[:3]
        step_group=[test_data['steps'][i] for i in top4_indices]
        pred_answers=[extract_pred(step[-1]) for step in step_group]
        counter = Counter(pred_answers)
        most_common=counter.most_common(1)[0][0]
        gold_label=test_data['extracted_groundtruth']
        try:
            if check_correctness(most_common,gold_label, args):
                correct_cnt+=1
        except:
            continue
            #print("Paths maybe problematic:",filename)

    acc=correct_cnt/all_cnt
    print("Top3-SC",all_cnt,correct_cnt,acc)

def eval(data_dir,args):
    files=get_json_files(data_dir)
    best_of_N(files,data_dir,args)
    all_correct_pth(files,data_dir,args)
    top_4_SC(files,data_dir,args)
    lscr(files,data_dir,args)
    lscr_SC(files,data_dir,args)
    lscr_best(files,data_dir,args)
    select(files,data_dir,args)
    #print(correct_lst, error_lst)
    #best_of_N_bins(files,data_dir,args,bins=10)
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

