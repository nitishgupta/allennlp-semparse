#!/usr/bin/env

export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OPENMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

export CUDA=-1

INCLUDE_PACKAGE=allennlp_semparse
CONFIGFILE=training_config/nlvr_mml_parser.jsonnet

# DATA PATH
export DATADIR=agenda_v6_ML11
export TRAIN_DATA=./resources/data/nlvr/processed/ERM_DATAGEN/paired_iter1.json
# ${DATADIR}/train_mml_cands_L18.json
# train_grouped.json
# train_mml_cands.json
export DEV_DATA=./resources/data/nlvr/processed/dev_grouped.json

# HYPER-PARAMETERS
export MDS=18
export EPOCHS=1
export SEED=42


# SERIALIZATION PATH
CHECKPOINT_ROOT=./resources/checkpoints
MODEL_DIR=mml_parser/nlvr/${DATADIR}
PARAMETERS=MDS_${MDS}/S_${SEED}
SERIALIZATION_DIR=${CHECKPOINT_ROOT}/${MODEL_DIR}/${PARAMETERS}-MMLCands

SERIALIZATION_DIR=${CHECKPOINT_ROOT}/test

allennlp train --include-package allennlp_semparse  -s ${SERIALIZATION_DIR} ${CONFIGFILE}
