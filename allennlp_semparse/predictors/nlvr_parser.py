import json

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register("nlvr-parser")
class NlvrParserPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        if "worlds" in json_dict:
            # This is grouped data
            worlds = json_dict["worlds"]
            if isinstance(worlds, str):
                worlds = json.loads(worlds)
        else:
            structured_rep = json_dict["structured_rep"]
            if isinstance(structured_rep, str):
                structured_rep = json.loads(structured_rep)
            worlds = [structured_rep]
        identifier = json_dict["identifier"] if "identifier" in json_dict else None
        instance = self._dataset_reader.text_to_instance(
            sentence=sentence,  # type: ignore
            structured_representations=worlds,
            identifier=identifier,
        )
        return instance

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        if "identifier" in outputs:
            # Returning CSV lines for official evaluation
            identifier = outputs["identifier"]
            # Denotation, for each program, for each world --- List[List[str]]
            # Note: if no programs were decoded for this example, this list would be empty
            denotations = outputs["denotations"]
            if denotations:
                denotation = denotations[0][0]
            else:
                denotation = "NULL"
            return f"{identifier},{denotation}\n"
        else:
            return json.dumps(outputs) + "\n"


@Predictor.register("nlvr-parser-visualize")
class NlvrParserPredictor(NlvrParserPredictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        if "worlds" in json_dict:
            # This is grouped data
            worlds = json_dict["worlds"]
            labels = json_dict["labels"]
            if isinstance(worlds, str):
                worlds = json.loads(worlds)
        else:
            structured_rep = json_dict["structured_rep"]
            labels = [json_dict["label"]]
            if isinstance(structured_rep, str):
                structured_rep = json.loads(structured_rep)
            worlds = [structured_rep]
        identifier = json_dict["identifier"] if "identifier" in json_dict else None
        instance = self._dataset_reader.text_to_instance(
            sentence=sentence,  # type: ignore
            structured_representations=worlds,
            identifier=identifier,
            labels=labels,
        )
        return instance


    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        identifier = outputs.get("identifier", "N/A")
        sentence = outputs["sentence"]
        best_action_strings = outputs["best_action_strings"]
        logical_form = outputs["logical_form"]
        denotations = outputs["denotations"]
        sequence_is_correct = outputs.get("sequence_is_correct", None)

        consistent = None
        if sequence_is_correct is not None:
            consistent = all(sequence_is_correct)


        output_dict = {
            "identifier": identifier,
            "sentence": sentence,
            "best_action_strings": best_action_strings,
            "best_logical_forms": logical_form,
            "denotations": denotations,
            "sequence_is_correct": sequence_is_correct,
            "consistent": consistent,
        }

        output_str = json.dumps(output_dict, indent=2) + "\n"

        return output_str



