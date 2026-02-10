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


import json
import os
import ast
with open("GenPRM/src/eval/wtq_error_llm_lst.txt", "r") as f:
    data_llm = ast.literal_eval(f.read())
with open("GenPRM/src/eval/wtq_error_lst.txt", "r") as f:
    data_prm=ast.literal_eval(f.read())


data_llm = [x.split("/")[0] for x in data_llm]
data_prm = [x.split("/")[0].split("_analyze")[0] for x in data_prm]


prm_correct_llm_wrong=[z for z in data_llm if z not in data_prm]
llm_correct_prm_wrong=[m for m in data_prm if m not in data_llm]
print("prm_correct_llm_wrong: ", prm_correct_llm_wrong)
print("llm_correct_prm_wrong: ", llm_correct_prm_wrong)