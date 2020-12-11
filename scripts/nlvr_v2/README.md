# Paired Training - NLVR

# Candidate programs via exhaustive search
Run `get_nlvr_logical_forms.py` to search for candidate programs. 
With function composition and currying langauge with one action removed, 
we can search for programs with length = 11. Takes about 8 hours.

```
time python scripts/nlvr_v2/get_nlvr_logical_forms.py \
    resources/data/nlvr/processed/train_grouped.json \
    resources/data/nlvr/processed/agendav6_SORT_ML11/train_grouped.json \
    --write-action-sequences \
    --max-path-length 11
```
Coverage: `54.6%`

# MML parser
Train a parser with MML loss `training_config/nlvr_mml_parser.jsonnet`.
Requires a set of program candidates for an utterance. 
For example, using exhaustive search as above.

Run `bash scripts/train/nlvr_mml_train.sh` with appropriate data paths.

Note:
1. The reader `nlvr_v2_mml` can read data.json which contains examples without candidate programs 
(no `correct_sequences` field). It disregards such examples.

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
Now `resources/data/nlvr/processed/agenda_v6_ML11/train_mml_cands.json` can be used with a MML parser as is.


# Paired examples data
We manually identify certain NL phrases that should be paired in `scripts/nlvr_v2/data/paired_phrases_v1.json`.

Using this, we can automatically find paired examples in the training data using,
```
python scripts/nlvr_v2/paired_supervision/generate_paired_data.py \
    resources/data/nlvr/processed/train_grouped.json \
    scripts/nlvr_v2/data/paired_phrases_v1.json \
    resources/data/nlvr/processed/paired_data/train_v1.json
```
All examples from the input are written; instances for which a paired example is found,
an additional field `paired_example` is added containing info about the paired instance.

The output can be used with a ERM parser that does not require candidate programs  

Note: 
1. Currently, we only limit to a single paired example per instance.
2. Candidate-programs --- We do write the candidate programs (`correct_sequences` field) 
from the input to the output as well, in the anticipation that we could use some paired loss with MML parser.
 
## Iterative Parser

**Basic-erm** training:
1. Uses `training_config/nlvr_paired_parser.jsonnet`, which uses the `nlvr_v2_paired` reader 
2. `train_json` should contain paired_examples
```
python scripts/train/iterative_train.py \
    --train_search_json ./resources/data/nlvr/processed/agendav6_SORT_ML11/train_grouped.json \
    --train_json ./resources/data/nlvr/processed/train_grouped.json \
    --erm_model paired \
    --ckpt_root ./resources/checkpoints/nlvr/pairedv1_SORT_01 \
    --seed 1337
```

**Paired-erm** training:
1. Uses `training_config/nlvr_paired_parser.jsonnet`, which uses the `nlvr_v2_paired` reader
2. `train_json` should be vanilla
```
python scripts/train/iterative_train.py \
    --train_search_json ./resources/data/nlvr/processed/agendav6_SORT_ML11/train_grouped.json \
    --train_json  ./resources/data/nlvr/processed/paired_data/train_v1.json \
    --erm_model paired \
    --ckpt_root ./resources/checkpoints/nlvr/pairedv1_SORT_01 \
    --seed 1337
```

**Coverage-erm** training:
1. Uses `training_config/nlvr_coverage_parser.jsonnet`, which uses the `nlvr_v2` reader and a different model 
which uses the agenda-coverage transition function from Dasigi et al.
2. `train_json` should be vanilla
3. *TODO: paired version for this is not yet implemented*
```
python scripts/train/iterative_train.py \
    --train_search_json ./resources/data/nlvr/processed/agendav6_SORT_ML11/train_grouped.json \
    --train_json ./resources/data/nlvr/processed/train_grouped.json \
    --erm_model coverage \
    --ckpt_root ./resources/checkpoints/nlvr/coverage_SORT \
    --seed 1337
```

## Compositional generalization

###Abstract structures v1
**Splitting data** 
Construct abstract structures by masking colors, numbers, shapes, etc.
Re-split all of NLVR data by sampling few structures for test, 
and splitting the rest of the instances into train/dev.   

The following command creates `train.json`, `dev.json` and `test.json` in `absstr_v1` 
```
python scripts/nlvr_v2/comp_gen/structural_compgen.py \
    ./resources/data/nlvr/processed/all_data.json \
    ./resources/data/nlvr/comp_gen/absstr_v1
```

**Search candidates**
For the training instances, create `train_search_NEWLANG.json` that contains 
program candidates from exhaustive-search for MML.

The following command uses 
`agendav6_SORT_ML11_SD/all_data.json` (which contains program candidates for all of the NLVR data)
to write `train_search_NEWLANG.json` for instances in `absstr_v1/train.json`
```
python scripts/nlvr_v2/comp_gen/search_cands.py \
    ./resources/data/nlvr/comp_gen/absstr_v1/train.json \
    ./resources/data/nlvr/processed/agendav6_SORT_ML11_SD/all_data.json \
    train_search_NEWLANG.json
```
**Paired examples**
Same as before, generate data with paired examples using `generate_paired_data.py`
```
python scripts/nlvr_v2/paired_supervision/generate_paired_data.py \
    ./resources/data/nlvr/comp_gen/absstr_v1/train.json \
    scripts/nlvr_v2/data/paired_phrases_v1.json \
    ./resources/data/nlvr/comp_gen/absstr_v1/train_paired_v1_P1M1.json \
    --max_samples_per_phrase 1 \
    --max_samples_per_instance 1
```

### Attribute pair v1
**Splitting data** 
Given abstract paired phrases, sample a attribute value combination and put all 
examples with those phrases in test.
```
python scripts/nlvr_v2/comp_gen/attrpair_compgen.py \
    ./resources/data/nlvr/processed/all_data.json \
    ./scripts/nlvr_v2/data/paired_phrases_v13.json \
    ./resources/data/nlvr/comp_gen/attrpair
```

**Search candidates**
Get search-candidates for train examples in this split
```
python scripts/nlvr_v2/comp_gen/search_cands.py \
    ./resources/data/nlvr/comp_gen/attrpair/train.json \
    ./resources/data/nlvr/processed/agenda_PL_ML11/all_data.json \
    train_search.json
```

**Paired examples**
Generate paired examples; example below creates a version where instances are 
also paired with examples such that they only contain non-terminal matching.
```
python ./scripts/nlvr_v2/paired_supervision/generate_paired_data.py \
    ./resources/data/nlvr/comp_gen/attrpair/train.json \
    ./scripts/nlvr_v2/data/paired_phrases_v13.json \
    ./resources/data/nlvr/comp_gen/attrpair/train_v13_P1M1NT1.json \
    --add-nt-matches
```


**Iterative parser training**
Same as before, run `iterative_train.py` with appropriate input arguments:
```
python scripts/train/iterative_train.py \
    --train_search_json ./resources/data/nlvr/comp_gen/absstr_v1/train_search_NEWLANG.json \
    --dev_json ./resources/data/nlvr/comp_gen/absstr_v1/dev.json \
    --erm_model paired \
    --train_json ./resources/data/nlvr/comp_gen/absstr_v1/train_paired_v1_P1M1.json \
    --ckpt_root ./resources/checkpoints/nlvr/comp-gen/absstr_v1/pairederm_NEWLANG_T07_P1M1 \
    --seed 21
```








 
 
