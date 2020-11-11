# Paired Training - NLVR

# Candidate programs via exhaustive search
Run `get_nlvr_logical_forms.py` to search for candidate programs. 
With function composition and currying langauge with one action removed, 
we can search for programs with length = 11. Takes about 8 hours.

```
time python scripts/nlvr_v2/get_nlvr_logical_forms.py \
    resources/data/nlvr/processed/train_grouped.json \
    resources/data/nlvr/processed/agenda_v6_ML11/train_grouped.json \
    --write-action-sequences \
    --max-path-length 11
```
Coverage: `54.6%`

# MML parser
Train a parser with MML loss `training_config/nlvr_mml_parser.jsonnet`.
Requires a set of program candidates for an utterance. 
For example, using exhaustive search as above.

Run `bash scripts/train/nlvr_mml_train.sh` with appropriate data paths.

# Candidate programs from trained MML parser
Use a pre-trained MML parser to generate longer consistent candidates for instances.
The inference uses the same number of max-decoding-steps as used by the MML parser
(e.g. 14 in the example below)

```
time scripts/nlvr_v2/generate_data_from_mml_model.py \
    resources/data/nlvr/processed/train_grouped.json \
    resources/data/nlvr/processed/agenda_v6_ML11/train_mml_cands.json \
    ./resources/checkpoints/mml_parser/nlvr/agenda_v6_ML11/MDS_14/S_42/model.tar.gz \
    --cuda-device -1
```






 
