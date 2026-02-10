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

def list_combine(lst1,lst2):
    m=[]
    for h,k in zip(lst1, lst2):
        m.append([h,k])
    return m

def combine(files,data_dir,args):
    if args.modify:
        combine_pth = args.data_path_code
        gold_pth=args.data_path_textual
        idx=0
        for filename in files:
            try:
                combine_data_pth = os.path.join(combine_pth,filename)

                combine_data=load_json(combine_data_pth)

                gold_data_pth= os.path.join(gold_pth,filename)
                gold_data = load_json(gold_data_pth)

                gold_score=gold_data["prm_score"]
                
                random_index=gold_data['random_index']
                combine_score=combine_data["prm_score"][random_index]
                #if idx==0:
                #    print(gold_score)
                #    print(combine_score)
                #    print(filename)
                #    idx+=1
                gold_data['prm_score']=combine_score
                gold_data['prm_conversation']=combine_data['prm_conversation'][random_index]
                with open(gold_data_pth, "w", encoding="utf-8") as f:
                    json.dump(gold_data, f, indent=2)
            except:
                continue
    else:
        code_pth=args.data_path_code
        for filename in files:
            try:
                textual_data_pth = os.path.join(data_dir,filename)

                textual_data=load_json(textual_data_pth)

                code_data_pth= os.path.join(code_pth,filename)
                code_data = load_json(code_data_pth)

                problem=code_data['problem']
                extracted_groundtruth=code_data["extracted_groundtruth"]
                code_score=code_data["value"][:args.size]
                steps=code_data['steps'][:args.size]
                prm_conversation=code_data['conversations'][:args.size]
                textual_score=textual_data["value"][:args.size]

                score_combined_lst=[]
                for i,j in zip(textual_score,code_score):
                    score_combined_lst.append(list_combine(i,j))


                save_dict = {
                        "filename": filename,
                        "problem": problem,
                        "extracted_groundtruth": extracted_groundtruth,
                        "steps": steps,
                        "prm_score": score_combined_lst,
                        "prm_conversation": prm_conversation,

                    }

                os.makedirs(os.path.dirname(os.path.join(args.save_path, filename)), exist_ok=True)
                with open(os.path.join(args.save_path, filename), "w") as f:
                    json.dump(save_dict, f, indent=2)
            except:
                continue

def combine2(files,data_dir,args):
    random_pth=args.data_path_textual
    e1_pth = args.data_path_code
    
    
    for filename in files:
        #print(filename)
        try:
            e1_data_pth = os.path.join(e1_pth,filename)
            e1_data=load_json(e1_data_pth)
            random_data_pth= os.path.join(random_pth,filename)
            random_data = load_json(random_data_pth)
            
            random_index=random_data['random_index']
            e1_score=e1_data["value"][random_index]
            random_data['prm_withoutcode_32b']=e1_score
            with open(random_data_pth, "w", encoding="utf-8") as f:
                json.dump(random_data, f, indent=2)
        except:
            continue
def eval(data_dir,args):
    files=get_json_files(data_dir)
    combine2(files,data_dir,args)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation PRM")
    parser.add_argument("--data_path_textual", type=str)
    parser.add_argument("--data_path_code", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--size",type=int,default=8)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--modify", type=bool, default=False)
    return parser.parse_args()

def main():
    args = parse_args()
    eval(args.data_path_textual,args)

if __name__=="__main__":
    main()

