#! /usr/bin/env python
from typing import List, Dict
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

from scripts.nlvr_v2.data.nlvr_instance import NlvrInstance, read_nlvr_data

COLORS = ["yellow", "black", "blue"]
SHAPES = ["circle", "triangle", "square"]
NUMBERS = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]

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


class PairedPhrase:
    def __init__(
        self, abstract_phrases: List[str], grounded_phrases: List[str], sentences: List[str] = []
    ):
        self.abstract_phrases: List[str] = abstract_phrases
        self.grounded_phrases: List[str] = grounded_phrases
        self.sentences: List[str] = sentences

    def __str__(self):
        output = "\n*********\n"
        output += "\n".join(self.abstract_phrases)
        output += "\n-----------\n"
        indices = list(range(len(self.sentences)))
        random.shuffle(indices)
        indices = indices[0:10]
        sents = [self.sentences[i] for i in indices]
        output += "\n".join(sents)
        output += "\n*********\n"
        return output


def convert_abstract_phrase_to_grounded(abstract_phrases: List[str]) -> List[List[str]]:
    """Convert a set of abstract phrases into multiple grounded sets each set containing a unique combination of
    abstract token to grounded token mapping.
    For example, input: ["COLOR1 at the base", "COLOR1 as the base"] would be converted to multiple sets, each one
    containing one value for the COLOR1 variable

    We currently limit to two colors, two shapes and one number in the phrase.
    """

    # This contains different equivalent (partially) grounded phrases.
    grounded_phrases_sets: List[List[str]] = [abstract_phrases]

    abstractions = ["COLOR1", "COLOR2", "SHAPE1", "SHAPE2", "NUMBER1"]
    for abstract_token in abstractions:
        # Go through each possible abstract token in order and expand `grounded_phrases` by considering all possible
        # groundings of this abstract token
        new_grounded_phrases_sets = []
        for equivalent_set in grounded_phrases_sets:
            # Mutate this set into multiple equivalent sets by replacing the abstract token with all its groundings.
            # For example, if this set is
            # ["yellow SHAPE1 at the base"], it should lead to three new sets ["yellow square at the base"],
            # ["yellow triangle at the base"], and ["yellow circle at the base"].
            if not all([abstract_token in x for x in equivalent_set]):
                # If phrases in this set does not contain the abstract token, mutations cannot be made
                # add this equivalent set to the final sets as it is
                new_grounded_phrases_sets.append(equivalent_set)
                continue
            if "COLOR" in abstract_token:
                options = COLORS
            elif "SHAPE" in abstract_token:
                options = SHAPES
            elif "NUMBER" in abstract_token:
                options = NUMBERS
            else:
                raise NotImplementedError
            # This equivalent set would be mutated into as many new sets as grounding options
            new_equivalent_sets = []
            for grounding_token in options:
                # Each phrase in the current equivalent set will be grounded with this token and added to the new set
                new_equivalent_set = []
                for phrase in equivalent_set:
                    new_phrase = phrase.replace(abstract_token, grounding_token)
                    new_equivalent_set.append(new_phrase)
                if grounding_token in number_strings:
                    alternate_token = number_strings[grounding_token]
                    for phrase in equivalent_set:
                        new_phrase = phrase.replace(abstract_token, alternate_token)
                        new_equivalent_set.append(new_phrase)
                if new_equivalent_set:
                    new_equivalent_sets.append(new_equivalent_set)
            # Add these new sets to the new collection
            new_grounded_phrases_sets.extend(new_equivalent_sets)
        grounded_phrases_sets = copy.deepcopy(new_grounded_phrases_sets)

    return grounded_phrases_sets


def make_data(
    data_jsonl: str,
    paired_phrases_json: str,
    output_jsonl: str,
) -> None:
    print("\nReading NLVR data ... ")
    nlvr_instances: List[NlvrInstance] = read_nlvr_data(data_jsonl)
    print("NLVR instances read: {}\n".format(len(nlvr_instances)))

    print("Reading paired data ... ")
    # Each inner list is a set of equivalent abstract phrases; e.g. ["COLOR1 as the base", "the base is COLOR1"]
    with open(paired_phrases_json) as f:
        paired_phrases_list: List[List[str]] = json.load(f)

    num_abstract_phrases, num_grounded_phrases = 0, 0
    paired_phrases: List[PairedPhrase] = []
    # Each abstract set is converted to multiple grounded sets by considering all possible values of the abstract tokens
    for equivalent_abstract_set in paired_phrases_list:
        # One set of equivalent abstract phrases is converted to many equivalent sets after grounding
        equivalent_grounded_sets: List[List[str]] = convert_abstract_phrase_to_grounded(
            equivalent_abstract_set
        )
        for grounded_set in equivalent_grounded_sets:
            # Instances that contain any of the equivalent phrases are paired
            paired_nlvr_sentences: List[str] = []
            for instance in nlvr_instances:
                if any([x in instance.sentence for x in grounded_set]):
                    paired_nlvr_sentences.append(instance.sentence)
            if not paired_nlvr_sentences:
                continue
            # Keep this paired phrase only if there are instances containing it
            paired_phrase = PairedPhrase(
                abstract_phrases=equivalent_abstract_set,
                grounded_phrases=grounded_set,
                sentences=paired_nlvr_sentences,
            )
            paired_phrases.append(paired_phrase)
            num_abstract_phrases += 1
            num_grounded_phrases += len(grounded_set)

    avg_grounded_phrases_per_set = float(num_grounded_phrases) / num_abstract_phrases
    print(
        "Paired phrases generated. Num of equivalent grounded phrases: {}  "
        "Avg num of grounded phrases per set:{}".format(
            num_abstract_phrases, avg_grounded_phrases_per_set
        )
    )

    for paired_phrase in paired_phrases:
        print(paired_phrase)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_jsonl", type=str, help="Input data file")
    parser.add_argument(
        "paired_phrases_json",
        type=str,
        help="Input file containing paired phrases",
    )
    parser.add_argument(
        "output_jsonl",
        type=str,
        help="Path to archived model.tar.gz to use for decoding",
    )
    args = parser.parse_args()
    make_data(args.data_jsonl, args.paired_phrases_json, args.output_jsonl)
