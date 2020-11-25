from typing import List, Dict, Tuple
import sys
import os
import json
import copy
import argparse
import random

random.seed(42)

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
    ),
)

from scripts.nlvr_v2.data.nlvr_instance import NlvrInstance, read_nlvr_data, write_nlvr_data, print_dataset_stats


BOXES = ["box", "boxes", "tower", "towers", "gray square", "gray squares"]
BLOCKS = ["item", "items", "block", "blocks", "object", "objects"]
COLORS = ["yellow", "black", "blue"]
SHAPES = ["circle", "triangle", "square"]
SIZES = ["small", "medium", "large"]

number_strings = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def get_structure(question: str):
    structure: str = copy.deepcopy(question)
    structure = structure.lower()

    # replacing boxes first, since "gray square" => BOX, otherwise, square would convert to a SHAPE
    for box in BOXES:
        structure = structure.replace(box, "BOX")
    for block in BLOCKS:
        structure = structure.replace(block, "BLOCK")
    for color in COLORS:
        structure = structure.replace(color, "COLOR")
    for shape in SHAPES:
        structure = structure.replace(shape, "SHAPE")
    for number in number_strings.values():
        structure = structure.replace(number, "NUMBER")
    for number in number_strings.keys():
        structure = structure.replace(number, "NUMBER")
    for size in SIZES:
        structure = structure.replace(size, "SIZE")

    structure = structure.replace("BOXs", "BOX")
    structure = structure.replace("BOXes", "BOX")
    structure = structure.replace("BLOCKs", "BLOCK")
    structure = structure.replace("COLORs", "COLOR")
    structure = structure.replace("SHAPEs", "SHAPE")
    structure = structure.replace("NUMBERs", "NUMBER")

    structure = structure.replace(" are ", " IS ")
    structure = structure.replace(" is ", " IS ")


    print(question)
    print(structure)
    print()

    return structure


def get_compgen_split(train_jsonl):
    train_instances: List[NlvrInstance] = read_nlvr_data(train_jsonl)
    print_dataset_stats(train_instances)

    structure2identifiers = {}
    structure2count = {}

    for instance in train_instances:
        structure = get_structure(instance.sentence)
        if structure not in structure2identifiers:
            structure2identifiers[structure] = []
            structure2count[structure] = 0
        structure2identifiers[structure].append(instance.identifier)
        structure2count[structure] += 1

    num_structures = len(structure2identifiers)
    print("Number of abstract structures : {}".format(num_structures))

    sorted_struc2count = sorted(structure2count.items(), key=lambda x: x[1], reverse=True)
    truncated_sorted_struc2count = [(x,y) for (x,y) in sorted_struc2count if y > 2]
    print(truncated_sorted_struc2count)
    print(len(truncated_sorted_struc2count))
    print()

    mono_sorted_struc2count = [(x, y) for (x, y) in sorted_struc2count if y == 1]
    print(mono_sorted_struc2count)
    print(len(mono_sorted_struc2count))

    multstructures = [structure for structure, count in structure2count.items() if count > 2]
    # print(multstructures)
    print(len(multstructures))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_jsonl", type=str, help="Input data file")
    # parser.add_argument(
    #     "paired_phrases_json",
    #     type=str,
    #     help="Input file containing paired phrases",
    # )
    # parser.add_argument(
    #     "output_jsonl",
    #     type=str,
    #     help="Path to archived model.tar.gz to use for decoding",
    # )
    # parser.add_argument(
    #     '--max_samples_per_phrase',
    #     type=int,
    #     default=1,
    #     help="For each grounded phrase, sample these many paired instances per instance"
    # )
    # parser.add_argument(
    #     '--max_samples_per_instance',
    #     type=int,
    #     default=1,
    #     help="Max number of paired instance per example"
    # )

    args = parser.parse_args()
    get_compgen_split(train_jsonl=args.train_jsonl)