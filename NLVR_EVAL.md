# NLVR EVAL 


## Conda environment
```
conda create -n nlvr_eval python=3.6
conda activate nlvr_eval
pip install -r requirements.txt 
```

## Code clone
```
git clone https://github.com/nitishgupta/allennlp-semparse.git
cd allennlp_semparse
git pull origin nlvr-eval
git checkout nlvr-eval
```

## Untar models
```
# This should create 
# files: modelA_best.tar.gz / modelB_best.tar.gz
# dirs: all_modelA and all_modelB
cd resources/checkpoints
tar -xvzf models.tar.gz
cd ../..
```

## Copy Test data
Copy hidden test data into `resources/data/`

## Best model eval
Run eval for modelA and modelB. 
Each command should create a corresponding `model{A,B}-test-p-metrics.txt` in the `resources/checkpoints` directory.

Will need to change `test-p` to `test-h` (or something) for the hidden test set.
```
# Model A
bash ./scripts/nlvr_eval/run_best_eval.sh \
  ./resources/data/test-p.json \
  ./resources/checkpoints/modelA_best.tar.gz \
  ./resources/checkpoints/modelA-test-p.csv \
  ./resources/checkpoints/modelA-test-p-metrics.txt

# Model B
bash ./scripts/nlvr_eval/run_best_eval.sh \
  ./resources/data/test-p.json \
  ./resources/checkpoints/modelB_best.tar.gz \
  ./resources/checkpoints/modelB-test-p.csv \
  ./resources/checkpoints/modelB-test-p-metrics.txt
```


## All models eval
Will need to change `test-p` to `test-h` (or something) for the hidden test set.
```
# Model A
bash ./scripts/nlvr_eval/run_all_eval.sh \
  ./resources/data/test-p.json \
  ./resources/checkpoints/all_modelA \
  test-p \
  test-p-metrics

# Model B
bash ./scripts/nlvr_eval/run_all_eval.sh \
  ./resources/data/test-p.json \
  ./resources/checkpoints/all_modelB \
  test-p \
  test-p-metrics
```