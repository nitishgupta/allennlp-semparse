#!/usr/bin/env

export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export CUDA=-1

TRAIN_SEARCH_JSON=./resources/data/nlvr/processed/agenda_PL_ML11/train_grouped.json

TRAIN_ERM_JSON=./resources/data/nlvr/processed/paired_data/train_v13_P1M1.json

DEV_JSON=./resources/data/nlvr/processed/dev_grouped.json

CKPT_ROOT=/tmp/v13P1M1

for SEED in 21 42 1337 15 17
do
  python scripts/train/iterative_train.py \
    --train_search_json ${TRAIN_SEARCH_JSON} \
    --dev_json ${DEV_JSON} \
    --train_json ${TRAIN_ERM_JSON} \
    --ckpt_root ${CKPT_ROOT} \
    --seed ${SEED} &
done


