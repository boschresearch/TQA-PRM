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
import matplotlib.pyplot as plt
import numpy as np 
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


def score_stats(files,data_dir,args):
    s_lst=[]
    for filename in files:
        test_data_pth = os.path.join(data_dir,filename)

        test_data=load_json(test_data_pth)
        score_list=test_data["value"][:args.size]
        for score in score_list:
            for s in score:
                s_lst.append(s)
    return s_lst

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation PRM")
    parser.add_argument("--data_path1", type=str)
    parser.add_argument("--data_path2", type=str)
    parser.add_argument("--data_path3", type=str)
    parser.add_argument("--data_path4", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--size",type=int)
    parser.add_argument("--save_path", type=str)

    return parser.parse_args()

def main():
    args = parse_args()
    print("Task: ", args.data_name)
    print("Data Size: ", args.size)

    
    files_7b=get_json_files(args.data_path1)
    files_32b=get_json_files(args.data_path2)
    files_prm_7b=get_json_files(args.data_path3)
    files_prm_32b=get_json_files(args.data_path4)
    score_lst_7b = score_stats(files_7b,args.data_path1,args)
    score_lst_32b = score_stats(files_32b,args.data_path2,args)
    score_prm_7b = score_stats(files_prm_7b,args.data_path3,args)
    score_prm_32b = score_stats(files_prm_32b,args.data_path4,args)
    fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True, figsize=(10, 8))
    axs[0,0].hist(score_lst_7b, bins=5)
    axs[0,0].set_title("7B LLM-as-a-Judge")
    axs[0,0].set_xlabel("Score")
    axs[0,0].set_ylabel("Count")

    axs[0,1].hist(score_lst_32b, bins=5)
    axs[0,1].set_title("32B LLM-as-a-Judge")
    axs[0,1].set_xlabel("Score")

    axs[1,0].hist(score_prm_7b, bins=5)
    axs[1,0].set_title("7B PRM")
    axs[1,0].set_xlabel("Score")


    axs[1,1].hist(score_prm_32b, bins=5)
    axs[1,1].set_title("32B PRM")
    axs[1,1].set_xlabel("Score")
    out_path = args.save_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {out_path}")


if __name__=="__main__":
    main()

