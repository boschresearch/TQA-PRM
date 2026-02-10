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


import re
import os
import json
from copy import deepcopy
import random
import argparse

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



def parse_args():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge")
    parser.add_argument("--data_path", type=str, required=True, help="Dataset Dir which needs to be preprocessed")
    return parser.parse_args()

def main():
    args = parse_args()
    print(args)
    files=get_json_files(args.data_path)
    print(f"{len(files)} Files Loaded Successfully:")
    count_all=0
    all_conversations=[]
    time = 0
    for file in files:
        
        rel_data=load_json(os.path.join(args.data_path,file))
        data=deepcopy(rel_data)
        if data:
            try:
                time += data['time']
                count_all+=1
            except:
                continue
    average_time=time/count_all   
    print("###Files: ", {count_all})
    print("###Time: ", {time})
    print("##Average Time ", {average_time})
          
if __name__=="__main__":
    main()