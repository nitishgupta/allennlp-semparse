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
git pull origin nlvr-v2
git checkout nlvr-v2
```

## Resources

```
mkdir -p resources/data
mkdir -p resources/checkpoints

cp ${TEST-P-JSON} ./resources/data
cp ${TEST-H-JSON} ./resources/data

# Put the models.tar.gz in ./resources/checkpoints an untar
# This should create 
# files: modelA_best.tar.gz / modelB_best.tar.gz
# dirs: all_modelA and all_modelB 
tar -xvzf models.tar.gz
```

## Best models
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


## Run all models
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





