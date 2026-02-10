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
            cot_label.append([])
    
    return cot_label

def mc_hard(mc_score_list,args):
    mc_hard_labels=[1 if i>=args.epsilon else 0 for i in mc_score_list]
    return mc_hard_labels

def mc_rpe(mc_score_list,args,base_score):
    assert len(mc_score_list) >=1, "mc_score_list can't be empty"
    mc_rpe_labels=[]
    
    for i in range(0,len(mc_score_list)):
        if i==0:
            if base_score!=0:
                score = mc_score_list[i]/base_score
            else:
                score = mc_score_list[i]
        elif mc_score_list[i-1]==0:
            score=0
        
        else:
            score=mc_score_list[i]/mc_score_list[i-1]    
        mc_rpe_labels.append(1 if score>=args.epsilon else 0)
    
    return mc_rpe_labels


def consistent_check(cot_label, mc_label):
    return cot_label == mc_label

def parse_args():
    parser = argparse.ArgumentParser(description="Data Preparation")
    parser.add_argument("--data_path", type=str, required=True, help="Dataset Dir which needs to be preprocessed")
    parser.add_argument("--save_path",type=str, required=True, help="Save Path")
    parser.add_argument("--epsilon",type=int, default=0.8, help="epsilon")
    parser.add_argument("--RPE", type=bool, default=True, help="Use RPE to generate labels")
    return parser.parse_args()

def main():
    args = parse_args()
    print(args)
    os.makedirs(args.save_path, exist_ok=True)
    files=get_json_files(args.data_path)
    print(f"{len(files)} Files Loaded Successfully:")
    count_all=0
    count_filtered=0
    all_conversations=[]
    for file in files:
        
        rel_data=load_json(os.path.join(args.data_path,file))
        data=deepcopy(rel_data)
        if data:
            
            base_score=data['metrics']['base_score']
            #print(file, base_score)
            cot_rationales=data['cot_rationale']
            step_group=data['steps']
            for i in range(4): ####the num of paths you have
                count_all+=1

                single_rationale=separate_cot(clean_cot(cot_rationales[i][0][2]['content']
                                        if cot_rationales[i][0] else ""))
                cot_label=cot_label_check(single_rationale)
                if args.RPE:
                    mc_label=mc_rpe(data['monte_carlo_scores'][i],args,base_score)
                else:
                    mc_label=mc_hard(data['monte_carlo_scores'][i],args) 

                if consistent_check(cot_label, mc_label):
                    conversations=[]
                    count_filtered+=1
                    conversations.append({
                        'role': 'system',
                        'content': 'You are a table question answering expert. Your task is to review and critique the paragraphs in solution step by step with python code.'
                    })

                   # conversations.append({
                    #    'role': 'user',
                     #   'content': data['problem']
                    #})
                    for j in range(len(cot_label)):
                        if j==0:
                            conversations.append({
                        'role': 'user',
                        'content': data['problem']+"\n\n"+re.sub(r'^Step\s*\d+:\s*', '', step_group[i][j], flags=re.IGNORECASE)
                        })
                        else:
                            conversations.append({
                            'role': 'user',
                            'content': re.sub(r'^Step\s*\d+:\s*', '', step_group[i][j], flags=re.IGNORECASE)
                        })
                        conversations.append({
                            'role': 'assistant',
                            'content': single_rationale[j]
                        })
                    all_conversations.append({'conversations': conversations})
    rate=count_filtered/count_all
    filename = f'TABFACT_{args.data_path.strip("/").split("/")[-1]}.json'
    with open(os.path.join(args.save_path, filename), 'w') as f:
        json.dump(all_conversations, f, indent=2)

    print("################# Dataset Processing Done ##################\n")
    print(f"################ Before Filtering: {count_all} ################\n")
    print(f"################ After Filering: {count_filtered} ################\n")
    print(f"################ Filtering Rate: {rate:.2f} ################\n")            
if __name__=="__main__":
    main()