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


import argparse
import json
import os
from datasets import load_dataset, Dataset


def export_all_splits(dataset_name, root_output_dir):
    """Process all available splits in the dataset"""
    # Load full dataset dictionary
    dataset_dict = load_dataset(dataset_name)

    # Create root output directory
    os.makedirs(root_output_dir, exist_ok=True)

    # Process each split
    for split_name in dataset_dict.keys():
        process_split(
            dataset=dataset_dict[split_name],
            split_name=split_name,
            root_output_dir=root_output_dir
        )


def process_split(dataset, split_name, root_output_dir):
    """Handle individual split processing"""
    # Create split-specific parent directory
    split_dir = root_output_dir
    os.makedirs(split_dir, exist_ok=True)

    # Save split-level dataset info
    dataset.info.write_to_directory(split_dir)

    # Process individual examples
    for idx, example in enumerate(dataset):
        example_dir = os.path.join(
            split_dir,
            f"{split_name}_{idx:03d}"
        )
        os.makedirs(example_dir, exist_ok=True)
        with open(os.path.join(example_dir, "sample.json"), "w") as f:
            json.dump(example, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset name")
    parser.add_argument("--split_dir", required=True, help="output directory")
    args = parser.parse_args()

    export_all_splits(
        dataset_name=args.dataset,
        root_output_dir=args.split_dir
    )
