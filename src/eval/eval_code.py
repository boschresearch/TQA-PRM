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

import re

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
def extract_assistant(rationales,args):
    assistant=[]
    for i in range(len(rationales)):
        for j in range(len(rationales[i]['conversations'])):
            if rationales[i]['conversations'][j]['role']=="assistant":
                assistant.append(rationales[i]['conversations'][j])
    
    return assistant

def extract_code(rationales,args):
    assistants=extract_assistant(rationales,args)
    print("# Assistant Steps: ", len(assistants))
    code_execute=[]
    code_execute_output=[]
    _pattern=r'```python\s*(.*?)\s*```'
    _pattern_output=r'```python\s*.*?\s*```.*\[Code Output\]\s*.*?\s*```(.*?)\s*```'
    for i in range(len(assistants)):
        content=assistants[i]['content']
        if re.search(_pattern,content,re.DOTALL):
            code_execute.append(re.search(_pattern, content, re.DOTALL).group(1))
            
        if re.search(_pattern_output, content, re.DOTALL):
            code_execute_output.append(re.search(_pattern_output, content, re.DOTALL).group(1))
    return code_execute, code_execute_output

def extract_conversations(data_dir,args):
    files=get_json_files(data_dir)
    conversations=[]
    for filename in files:
        eval_data_pth = os.path.join(data_dir,filename)
        eval_data=load_json(eval_data_pth)
        conversation_group=eval_data['conversations'][:args.size]
        conversations+=[{"conversations": conversation} for conversation in conversation_group]
    
    return conversations


def check_code(code_execute,code_execute_output):
    blank_output_lst=[]
    error_output_lst=[]
    no_code_lst=[]
    for i in range(len(code_execute_output)):
        if code_execute_output[i]=="":

            blank_output_lst.append(i)
        if "Code execute Error" in code_execute_output[i]:

            error_output_lst.append(i)
        if "Code execute time out" in code_execute_output[i]:

            error_output_lst.append(i)
    for i in range(len(code_execute)):
        if len(code_execute[i].split("\n"))<=2:
                no_code_lst.append(i)
        if "# no need for code" in code_execute[i].lower():
            no_code_lst.append(i)
        if "no code" in code_execute[i].lower():
            no_code_lst.append(i)

    return blank_output_lst,error_output_lst,no_code_lst

def eval(data_dir,args):
    conversations=extract_conversations(data_dir,args)
    #print(conversations[0])
    print(len(conversations))
    eval_code, eval_code_output=extract_code(conversations,args)
    blank_output_lst,error_output_lst,no_code_lst=check_code(eval_code,eval_code_output)
    all_error_code=sorted(list(set(blank_output_lst+error_output_lst+no_code_lst)))
    print("# Containing Code: ", len(eval_code))
    print("# Containing [Code] Output: ", len(eval_code_output))
    print("# Blank output: ", len(blank_output_lst))
    print("# Error output: ", len(error_output_lst))
    print("# No Code Needed: ", len(no_code_lst))
    print("# All Problematic Code: ",len(all_error_code))
def parse_args():
    parser = argparse.ArgumentParser(description="Code Evaluation PRM")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--size", type=int, default=8)
    return parser.parse_args()

def main():
    args = parse_args()
    print("Task: ", args.data_name)
    print("Data Size: ", args.size)
    eval(args.data_path,args)

if __name__=="__main__":
    main()

