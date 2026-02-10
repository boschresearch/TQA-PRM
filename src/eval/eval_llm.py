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

def clean_cot(content):
    if re.search("<think>", content):
        match=re.search("<think>",content)
        content=content[match.end():]
    if re.search("### Final Conclusion", content):
        match2=re.search("### Final Conclusion", content)
        content= content[:match2.start()]
    return content

def separate_cot(cot_content):
    cot_list=re.split("\n### Paragraph \d+\n", cot_content)[1:]
    return cot_list

def cot_indices_check(cot_indice): #################check if cot indices consistent [-1,-1,-1,-1]
    
    return all(x == cot_indice[0] for x in cot_indice) if len(cot_indice)>0 else False
        
def boxed_check(cot):
    BOXED_PATTERN1 = re.compile(
    r'(\\\[\s*)?\\boxed\s*{\s*(\\text\s*{\s*)?([yY][eE][sS]|[nN][oO])\s*(}\s*)?}(\s*\\\])?',
    flags=re.IGNORECASE)
    predicted_pattern=re.compile(r'(correct|incorrect|accurate|inaccurate|true|right)',re.IGNORECASE)
    match1 = re.search(BOXED_PATTERN1, cot)
    if match1:
        return cot, match1.group(3)
    else:
        try:
            predicted=re.findall(predicted_pattern,cot)[-1]
            if predicted.lower() in ["correct", "accurate"]:
                cot+= "\n<output>\n**Judgement**: $\\boxed{Yes}$ \n</output>\n\n"
                result="Yes"
            else:
                cot+= "\n<output>\n**Judgement**: $\\boxed{No}$ \n</output>\n\n"
                result="No"
        except IndexError:
            cot = None
            result=None
            
    return cot,result

def cot_label_check(cot_list): ##############check cot rationale label
    cot_label=[]
    for cot in cot_list:
        cot, result = boxed_check(cot)
        if result=="Yes":
            cot_label.append(1)
        
        elif result=="No":
            cot_label.append(0)
        
        else:
            cot_label.append(0.5)
    
    return cot_label


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
    count_filtered=0
    all_conversations=[]
    for file in files:
        
        rel_data=load_json(os.path.join(args.data_path,file))
        data=deepcopy(rel_data)
        if data:
            data['value']=[]
            #print(file, base_score)
            cot_rationales=data['cot_rationale']
            step_group=data['steps']
            for i in range(8): ####the num of paths you have
                count_all+=1

                single_rationale=separate_cot(clean_cot(cot_rationales[i][0][2]['content']
                                        if cot_rationales[i][0] else ""))
                cot_label=cot_label_check(single_rationale)

                data['value'].append(cot_label)
        with open(os.path.join(args.data_path, file), 'w') as f:
            json.dump(data, f, indent=2)
          
if __name__=="__main__":
    main()