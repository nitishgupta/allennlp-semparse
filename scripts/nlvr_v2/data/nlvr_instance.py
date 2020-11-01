from typing import List, Dict, Tuple, Union
import json

from allennlp_semparse.domain_languages.nlvr_language_v2 import NlvrLanguageFuncComposition, Box


class NlvrInstance:
    def __init__(self, instance_dict: Dict):
        self.identifier: str = instance_dict["identifier"]
        self.sentence: str = instance_dict["sentence"]

        if "worlds" in instance_dict:
            # This means that we are reading grouped nlvr data. There will be multiple
            # worlds and corresponding labels per sentence.
            labels = instance_dict["labels"]
            structured_representations = instance_dict["worlds"]
        else:
            # We will make lists of labels and structured representations, each with just
            # one element for consistency.
            labels = [instance_dict["label"]]
            structured_representations = [instance_dict["structured_rep"]]

        self.labels = labels
        self.structured_representations: List[Dict] = structured_representations
        self.worlds: List[NlvrLanguageFuncComposition] = None

        if "correct_sequences" in instance_dict:
            self.correct_candidate_sequences = instance_dict["correct_sequences"]

    def convert_structured_to_worlds(self):
        self.worlds = []
        for structured_representation in self.structured_representations:
            boxes = {
                Box(object_list, box_id)
                for box_id, object_list in enumerate(structured_representation)
            }
            self.worlds.append(NlvrLanguageFuncComposition(boxes))


def read_nlvr_data(input_jsonl: str) -> List[NlvrInstance]:
    instances: List[NlvrInstance] = []
    with open(input_jsonl) as data_file:
        for line in data_file:
            line = line.strip("\n")
            if not line:
                continue
            data = json.loads(line)
            instance: NlvrInstance = NlvrInstance(data)
            instances.append(instance)
    return instances
