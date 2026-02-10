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

def is_valid(scores):
    return all(not (scores[i-1] == 0 and scores[i] == 1) for i in range(1, len(scores)))

def valid(files,data_dir,args):
    valid_list=0
    all_list=0
    if args.prm:
        for filename in files:
            try:
                test_data_pth = os.path.join(data_dir,filename)

                test_data=load_json(test_data_pth)
                values=test_data['value']
                
                for value in values:
                    all_list+=1
                    if is_valid(value):
                        valid_list+=1
            except:
                continue
    else:    
        for filename in files:
            try:
                test_data_pth = os.path.join(data_dir,filename)

                test_data=load_json(test_data_pth)
                values=test_data['score']
                all_list+=1
                if is_valid(values):
                    valid_list+=1
            except:
                continue
    print("# Valid_list is: ", valid_list)
    print("# All list is: ", all_list)
    print("Rate is: ", valid_list/all_list)
            
                    



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation PRM")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--size",type=int)
    parser.add_argument("--prm", type=bool, default=False)
    return parser.parse_args()

def main():
    args = parse_args()
    print("Task: ", args.data_name)
    print("Data Size: ", args.size)
    files=get_json_files(args.data_path)
    valid(files,args.data_path,args)

if __name__=="__main__":
    main()

