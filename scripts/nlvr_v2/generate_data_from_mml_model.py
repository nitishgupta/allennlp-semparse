#! /usr/bin/env python


import sys
import os
import json
import argparse

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
)

from allennlp.models.archival import load_archive
from allennlp_semparse.dataset_readers import NlvrV2DatasetReader
from allennlp_semparse.models import NlvrMMLSemanticParser
from allennlp_semparse.domain_languages import NlvrLanguageFuncComposition
from allennlp_semparse.domain_languages.nlvr_language_v2 import Box
from allennlp_semparse.common import ParsingError, ExecutionError


def make_data(
    input_file: str,
    output_file: str,
    archived_model_file: str,
    max_num_decoded_sequences: int,
    cuda_device: int,
) -> None:
    reader = NlvrV2DatasetReader(output_agendas=False)
    model = load_archive(archived_model_file, cuda_device=cuda_device).model
    if not isinstance(model, NlvrMMLSemanticParser):
        model_type = type(model)
        raise RuntimeError(
            f"Expected an archived NlvrMMLSemanticParser, but found {model_type} instead"
        )
    # Tweaking the decoder trainer to coerce the it to generate a k-best list. Setting k to 100
    # here, so that we can filter out the inconsistent ones later.
    # model._decoder_beam_search == allennlp_semparse.state_machines.beam_search.BeamSearch
    model._decoder_beam_search._beam_size = 100
    model._decoder_beam_search._per_node_beam_size = 100
    model.training = False
    num_outputs = 0
    num_sentences = 0
    num_correct, num_correct_after_pruning = 0, 0
    with open(output_file, "w") as outfile:
        for line in open(input_file):
            num_sentences += 1
            input_data = json.loads(line)
            sentence = input_data["sentence"]
            structured_representations = input_data["worlds"]

            labels = input_data["labels"]
            instance = reader.text_to_instance(sentence, structured_representations)
            outputs = model.forward_on_instance(instance)
            action_strings = outputs["best_action_strings"]
            logical_forms = outputs["logical_form"]
            correct_sequences = []
            # Checking for consistency
            # worlds = [NlvrLanguageFuncComposition(structure) for structure in structured_representations]
            worlds = []
            for structured_representation in structured_representations:
                boxes = {
                    Box(object_list, box_id)
                    for box_id, object_list in enumerate(structured_representation)
                }
                worlds.append(NlvrLanguageFuncComposition(boxes))

            for sequence, logical_form in zip(action_strings, logical_forms):
                try:
                    denotations = [world.execute(logical_form) for world in worlds]
                except (ParsingError, ExecutionError, TypeError):
                    print("Error executing: {}\n".format(logical_form))
                    denotations = [False for world in worlds]
                denotations_are_correct = [
                    label.lower() == str(denotation).lower()
                    for label, denotation in zip(labels, denotations)
                ]
                if all(denotations_are_correct):
                    correct_sequences.append(sequence)
            num_correct += len(correct_sequences)
            correct_sequences = correct_sequences[:max_num_decoded_sequences]
            num_correct_after_pruning += len(correct_sequences)
            if not correct_sequences:
                continue
            output_data = {
                "id": input_data["identifier"] if "identifier" in input_data else input_data["id"],
                "sentence": sentence,
                "correct_sequences": correct_sequences,
                "worlds": structured_representations,
                "labels": input_data["labels"],
            }
            json.dump(output_data, outfile)
            outfile.write("\n")
            num_outputs += 1
        outfile.close()
    print(f"{num_outputs} out of {num_sentences} sentences have outputs.")
    avg_correct = float(num_correct) / num_outputs
    avg_correct_after_pruning = float(num_correct_after_pruning) / num_outputs
    print("Num candidates per example: {}".format(avg_correct))
    print("Num candidates per example after pruning: {}".format(avg_correct_after_pruning))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input data file")
    parser.add_argument("output", type=str, help="Output data file")
    parser.add_argument(
        "archived_model", type=str, help="Path to archived model.tar.gz to use for decoding"
    )
    parser.add_argument(
        "--max-num-sequences",
        type=int,
        dest="max_num_sequences",
        help="Maximum number of sequences per instance to output",
        default=20,
    )
    parser.add_argument("--cuda-device", dest="cuda_device", default=0)
    args = parser.parse_args()
    make_data(
        args.input, args.output, args.archived_model, args.max_num_sequences, args.cuda_device
    )
