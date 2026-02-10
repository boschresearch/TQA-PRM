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


import sys
import numpy as np
import torch
import argparse
import json
import os
import re
import math
import gc
import ray
import random
import time
import threading
import traceback
import psutil
from typing import List, Tuple
from utils.util import *
from copy import *
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from datasets import Dataset
from accelerate import Accelerator
from vllm import LLM, SamplingParams
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
from datetime import datetime
from colorama import Fore, init

# initialize colorama log
init()


def cprint(s, start):
    if not isinstance(s, str):
        s = str(s)

    print(f"{'*' * 40}")
    print(f"Start: {start}")
    print(f"{'-' * 40}")

    print(s.replace('\n', '\\n'))

    print(f"{'-' * 40}")
    print(f"End: {start}")
    print(f"{'*' * 40}\n")


def timestamped_print(message, level="INFO"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    color = {
        "ERROR": Fore.RED,
        "WARNING": Fore.YELLOW,
        "INFO": Fore.GREEN
    }.get(level, Fore.WHITE)
    print(f"{Fore.CYAN}[{now}]{Fore.RESET} {color}[{level}]{Fore.RESET} {message}")


def build_prompt(messages, tokenizer):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if prompt.endswith(f"{tokenizer.eos_token}\n"):
        prompt = prompt[:-len(f"{tokenizer.eos_token}\n")]
    elif prompt.endswith(tokenizer.eos_token):
        prompt = prompt[:-len(tokenizer.eos_token)]
    return prompt


def print_args(
    args: argparse.Namespace,
    program_name: str = None,
    version: str = None,
    show_version: bool = True
) -> None:
    '''
    print the args settings
    '''
    args_dict = {k: v for k, v in vars(args).items() if not k.startswith('_')}

    max_len = max(len(str(k)) for k in args_dict.keys())
    sep = '-' * (max_len + 20)

    output = []
    if program_name:
        output.append(f"\n\033[1;36m{program_name}\033[0m")

    if version and show_version:
        output.append(f"\033[1;34mVersion:\033[0m \033[1;33m{version}\033[0m")

    output.append(f"\033[1;35m{sep}\033[0m")

    for k, v in sorted(args_dict.items()):
        key = f"\033[1;32m{k:>{max_len}}\033[0m"
        val = f"\033[1;37m{str(v)}\033[0m"
        output.append(f"{key} : {val}")

    output.append(f"\033[1;35m{sep}\033[0m\n")

    print('\n'.join(output))
