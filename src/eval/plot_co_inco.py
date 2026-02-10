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
    s_correct_lst = []
    s_wrong_lst = []
    for filename in files:
        test_data_pth = os.path.join(data_dir,filename)

        test_data=load_json(test_data_pth)
        score_list=test_data["value"][:args.size]
        gold_label=test_data['extracted_groundtruth']
        for i in range(len(score_list)):
            try:
                if check_correctness(test_data['steps'][i][-1], gold_label,args):
                    for s in score_list[i]:
                        s_correct_lst.append(s)
                else:
                    for s in score_list[i]:
                        s_wrong_lst.append(s)
            except:
                continue

    return s_correct_lst, s_wrong_lst

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation PRM")
    parser.add_argument("--data_path1", type=str, help="7bLLM-as-a-Judge")
    parser.add_argument("--data_path2", type=str, help="32bLLM-as-a-Judge")
    parser.add_argument("--data_path3", type=str, help="7bPRM")
    parser.add_argument("--data_path4", type=str, help="32bPRM")
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--size",type=int)
    parser.add_argument("--save_path", type=str)

    return parser.parse_args()

def main():
    args = parse_args()
    print("Task: ", args.data_name)
    print("Data Size: ", args.size)

    files_7b      = get_json_files(args.data_path1)
    files_32b     = get_json_files(args.data_path2)
    files_prm_7b  = get_json_files(args.data_path3)
    files_prm_32b = get_json_files(args.data_path4)

    # gather correct & incorrect scores
    score_llm_cor_7b,  score_llm_inc_7b  = score_stats(files_7b,      args.data_path1, args)
    score_llm_cor_32b, score_llm_inc_32b = score_stats(files_32b,     args.data_path2, args)
    score_prm_cor_7b,  score_prm_inc_7b  = score_stats(files_prm_7b,  args.data_path3, args)
    score_prm_cor_32b, score_prm_inc_32b = score_stats(files_prm_32b, args.data_path4, args)

    out_path = args.save_path

    # datasets for plotting
    datasets_cor = [
        (np.array(score_llm_cor_7b),  "7B LLM-as-a-Judge",  "tab:blue"),
        (np.array(score_llm_cor_32b), "32B LLM-as-a-Judge", "tab:orange"),
        (np.array(score_prm_cor_7b),  "7B PRM",             "tab:green"),
        (np.array(score_prm_cor_32b), "32B PRM",            "tab:red"),
    ]
    datasets_inc = [
        (np.array(score_llm_inc_7b),  "7B LLM-as-a-Judge",  "tab:blue"),
        (np.array(score_llm_inc_32b), "32B LLM-as-a-Judge", "tab:orange"),
        (np.array(score_prm_inc_7b),  "7B PRM",             "tab:green"),
        (np.array(score_prm_inc_32b), "32B PRM",            "tab:red"),
    ]


    def _safe_array(x):
        # ensure float array and drop NaNs
        a = np.asarray(x, dtype=float)
        return a[np.isfinite(a)]
    
    def plot_counts_threshold(score_arrays, *, ge=True, title, outfile):
        """
        One bar per model.
        ge=True  -> count scores >= 0.5
        ge=False -> count scores < 0.5
        """
        fig, ax = plt.subplots(figsize=(8, 6))
    
        xs, heights, labels, colors = [], [], [], []
        for i, (scores, label, color) in enumerate(score_arrays):
            s = _safe_array(scores)
            count = np.count_nonzero(s >= 0.5) if ge else np.count_nonzero(s < 0.5)
            xs.append(i)
            heights.append(int(count))
            labels.append(label)
            colors.append(color)
    
        ax.bar(xs, heights, tick_label=labels, color=colors, alpha=0.9)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    
        for x, h in zip(xs, heights):
            ax.text(x, h, str(h), ha="center", va="bottom", fontsize=9)
    
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close(fig)
    
    # CORRECT figure — counts of scores >= 0.5 (from your *_cor_* arrays)
    plot_counts_threshold(
        datasets_cor,
        ge=True,
        title="Correct Prediction Scores (≥ 0.5) — one bar per model",
        outfile=out_path + "_cor_counts.png",
    )
    
    # INCORRECT figure — counts of scores < 0.5 (from your *_inc_* arrays)
    plot_counts_threshold(
        datasets_inc,
        ge=False,
        title="Incorrect Prediction Scores (< 0.5) — one bar per model",
        outfile=out_path + "_inc_counts.png",
    )
    
if __name__=="__main__":
    main()

