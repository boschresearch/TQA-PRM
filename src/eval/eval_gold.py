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


def gold_eval(files,data_dir,args):
    

    if args.method!=None:
        
        all_cnt_prm, correct_textual, correct_code=0,0,0
        all_cnt_modify, correct_cnt_modify=0,0
        correct_textual_lst,correct_code_lst,error_textual_lst,error_code_lst=[],[],[],[]

        for filename in files:
            #print(filename)
            test_data_pth = os.path.join(data_dir,filename)
            test_data=load_json(test_data_pth)
            try:
                gold_score_list=test_data["score"]
                prm_score=test_data['textual_disagreement']
                #assert len(gold_score_list)==len(prm_score)
                for i,j in zip(gold_score_list,prm_score):
                    textual_score=j[0]
                    
                    code_score=j[1]
                    #print(textual_score,code_score)
                    #print(textual_score,code_score)
                    all_cnt_prm+=1
                    if i==0 and textual_score<0.5:
                        correct_textual+=1
                    elif i==1 and textual_score>0.5:
                        correct_textual+=1
                    if i==0 and code_score<0.5:
                        correct_code+=1
                    elif i==1 and code_score>0.5:
                        correct_code+=1
                    if textual_score>0.5 and code_score<0.5:
                        all_cnt_modify+=1
                        if i==0:
                            correct_cnt_modify+=1
                        
                    elif textual_score<0.5 and code_score>0.5:
                        all_cnt_modify+=1
                        if i==1:
                            correct_cnt_modify+=1
            except:
                print(filename)
                continue
        print("all_cnt_prm: ", all_cnt_prm)
        print("correct textual: ", correct_textual)
        print("Accuracy: ", correct_textual/all_cnt_prm)
        print("correct code: ", correct_code)
        print("Accuracy: ", correct_code/all_cnt_prm)
        print("All Modification, Correct Modification: ", all_cnt_modify, correct_cnt_modify)
    else:
        annotated_num=0
        all_cnt_prm, correct_prm, all_cnt_llm,correct_llm, all_cnt_e1, correct_e1, all_cnt_prm_32b,correct_prm_32b=0,0,0,0,0,0,0,0
        all_cnt_e2, correct_cnt_e2, all_cnt_e3, correct_cnt_e3=0,0,0,0
        all_cnt_prm_wo_32b, correct_cnt_prm_wo_32b=0,0
        correct_textual_lst,correct_code_lst,error_textual_lst,error_code_lst=[],[],[],[]
        all_step,one_step=0,0
        for filename in files:
            test_data_pth = os.path.join(data_dir,filename)

            test_data=load_json(test_data_pth)

            try:
                gold_score_list=test_data["score"]
                annotated_num+=1
                e1_score = test_data['textual_score_7b']
                prm_score=test_data['prm_score']
                llm_score=test_data['llm_score']
                prm_score_32b=test_data["prm_32b"]
                e2=test_data['e2']
                #e3=test_data['e3']
                prm_withoutcode_32b=test_data['prm_withoutcode_32b']
                #print(gold_score_list)
                #print(llm_score)
                #assert len(gold_score_list)==len(prm_score)
                for i in gold_score_list:
                    all_step+=1
                    if i ==1:
                        one_step+=1
                for i,j in zip(gold_score_list,prm_score):
                    all_cnt_prm+=1
                    if i==0 and j<0.5:
                        correct_prm+=1
                    elif i==1 and j>0.5:
                        correct_prm+=1
                for i,j in zip(gold_score_list,llm_score):    
                    all_cnt_llm+=1
                    if i==0 and j<0.5:
                        correct_llm+=1
                    elif i==1 and j>0.5:
                        correct_llm+=1
                for i,j in zip(gold_score_list,e1_score):    
                    all_cnt_e1+=1
                    if i==0 and j<0.5:
                        correct_e1+=1
                    elif i==1 and j>0.5:
                        correct_e1+=1
                for i,j in zip(gold_score_list,prm_score_32b):    
                    all_cnt_prm_32b+=1
                    if i==0 and j<0.5:
                        correct_prm_32b+=1
                    elif i==1 and j>0.5:
                        correct_prm_32b+=1
                for i,j in zip(gold_score_list,e2):    
                    all_cnt_e2+=1
                    if i==0 and j<0.5:
                        correct_cnt_e2+=1
                    elif i==1 and j>0.5:
                        correct_cnt_e2+=1
                #for i,j in zip(gold_score_list,e3):    
                #    all_cnt_e3+=1
                #    if i==0 and j<0.5:
                #        correct_cnt_e3+=1
                #    elif i==1 and j>0.5:
                #        correct_cnt_e3+=1
                for i,j in zip(gold_score_list,prm_withoutcode_32b):    
                    all_cnt_prm_wo_32b+=1
                    if i==0 and j<0.5:
                        correct_cnt_prm_wo_32b+=1
                    elif i==1 and j>0.5:
                        correct_cnt_prm_wo_32b+=1
            except:
                continue
        print("all_step, one_step, rate: ",all_step, one_step, one_step/all_step)
        print("all_annotated: ", annotated_num)
        print("all_cnt_prm: ", all_cnt_prm)
        print("correct prm: ", correct_prm)
        print("Accuracy: ", correct_prm/all_cnt_prm)
        print("all_cnt_llm: ", all_cnt_llm)
        print("correct llm: ", correct_llm)
        print("Accuracy: ", correct_llm/all_cnt_llm)
        print("correct E1: ", correct_e1)
        print("Accuracy: ", correct_e1/all_cnt_e1)
        print("all_cnt_prm_32b: ", all_cnt_prm_32b)
        print("correct PRM_32b: ", correct_prm_32b)
        print("Accuracy: ", correct_prm_32b/all_cnt_prm_32b)
        print("all_cnt_prm_e2: ", all_cnt_e2)
        print("correct PRM_e2: ", correct_cnt_e2)
        print("Accuracy: ", correct_cnt_e2/all_cnt_e2)
        #print("all_cnt_prm_e3: ", all_cnt_e3)
        #print("correct PRM_e3: ", correct_cnt_e3)
        #print("Accuracy: ", correct_cnt_e3/all_cnt_e3)
        print("all_cnt_prm_wo_32b: ",all_cnt_prm_wo_32b)
        print("correct prm_wo_32b: ", correct_cnt_prm_wo_32b)
        print("Accuracy: ", correct_cnt_prm_wo_32b/all_cnt_prm_wo_32b)
def eval(data_dir,args):
    files=get_json_files(data_dir)
    gold_eval(files,data_dir,args)
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation PRM aganist Gold Labels")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--size",type=int)
    parser.add_argument("--method", type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    print("Task: ", args.data_name)
    print("Data Size: ", args.size)
    eval(args.data_path,args)

if __name__=="__main__":
    main()

