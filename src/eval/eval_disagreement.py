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


def disagreement(files,data_dir,args):
    total_cnt=0
    total_a_score=0
    total_v_score=0
    all_co_inc=[]
    co_inc=[] #last step, correct-> incorrect
    co_inc_cor=[] #last step, Correct -> incorrect, which are correct revised
    all_inc_co=[]
    inc_co=[] #last step, inc->co
    inc_co_cor=[]#last step, inc->co, which are correctly revised
    for filename in files:
        test_data_pth = os.path.join(data_dir,filename)

        test_data=load_json(test_data_pth)
        steps=test_data['steps']
        values=test_data['prm_score'][:args.size]
        gold_label=test_data['extracted_groundtruth']
        for i, value in enumerate(values): #value is step value set
            for j,step_value in enumerate(value):
                #print(j,len(steps[i]))
                try:
                    score_analyze=step_value[0]
                    score_verify=step_value[1]
                    total_a_score+=score_analyze
                    total_v_score+=score_verify
                    total_cnt+=1
                    if score_analyze < 0.5 and score_verify > 0.5:
                        all_inc_co.append((filename,i+1,j+1))
                        if j==len(steps[i])-1:
                            #print("Yes")
                            inc_co.append((filename, i+1, j+1))
                            final_answer=steps[i][-1]
                            if check_correctness(final_answer, gold_label,args):
                                inc_co_cor.append((filename,i+1,j+1))
                    elif score_analyze > 0.5 and score_verify < 0.5:
                        all_co_inc.append((filename,i+1,j+1))
                        if j==len(steps[i])-1:
                            co_inc.append((filename, i+1, j+1))
                            final_answer=steps[i][-1]
                            if not check_correctness(final_answer,gold_label,args):
                                co_inc_cor.append((filename,i+1,j+1))
                except:
                    print(filename,i,j)
    print("Total cnt: ", total_cnt)
    print("Total(Analyze + Output): ", total_a_score)
    print("AVG(Analyze + Output): ", total_a_score/total_cnt)
    print("Total(Analyze + Verify + Output):", total_v_score)
    print("AVG(Analyze + Verify + Output):", total_v_score/total_cnt)
    print("Total(Inc -> Co): ", len(all_inc_co), "Ratio: ", len(all_inc_co)/total_cnt)
    print("Total(Co -> Inc): ", len(all_co_inc), "Ratio: ", len(all_co_inc)/total_cnt)
    print("Total(Last_Step(Inc -> Co)): ", len(inc_co), "Ratio: ", len(inc_co)/total_cnt)
    print("Correct(Last_Step(Inc -> Co)): ", len(inc_co_cor), "Ratio: ", len(inc_co_cor)/len(inc_co))
    print("Total(Last_Step(Co -> Inc)): ", len(co_inc), "Ratio: ", len(co_inc)/total_cnt)
    print("Correct(Last_Step(Co -> Inc)): ", len(co_inc_cor), "Ratio: ", len(co_inc_cor)/len(co_inc))
    #print(co_inc_cor)
    #print("Inc -> Co: ", inc_co)
    #print("Co -> Inc: ", co_inc)


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
    files=get_json_files(args.data_path)
    disagreement(files,args.data_path,args)

if __name__=="__main__":
    main()

