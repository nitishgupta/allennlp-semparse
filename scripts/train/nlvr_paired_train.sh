#!/usr/bin/env

export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OPENMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

export CUDA=-1

INCLUDE_PACKAGE=allennlp_semparse
CONFIGFILE=training_config/nlvr_paired_parser.jsonnet

# DATA PATH
export DATADIR=paired_data
# export TRAIN_DATA=./resources/data/nlvr/processed/${DATADIR}/train_v1.json
export TRAIN_DATA=./resources/data/nlvr/processed/train_grouped.json
# train_mml_cands.json
# train_grouped.json
export DEV_DATA=./resources/data/nlvr/processed/dev_grouped.json

# HYPER-PARAMETERS
export MDS=14

export EPOCHS=50
export SEED=1337

export MML_MODEL_TAR=./resources/checkpoints/mml_parser/nlvr/agenda_v6_ML11/MDS_18/S_42/model.tar.gz

# SERIALIZATION PATH
CHECKPOINT_ROOT=./resources/checkpoints
MODEL_DIR=nlvr_paired_parser/${DATADIR}
PARAMETERS=MDS_${MDS}/S_${SEED}
SERIALIZATION_DIR=${CHECKPOINT_ROOT}/${MODEL_DIR}/${PARAMETERS}-ActionShare


SERIALIZATION_DIR=./resources/checkpoints/test; rm -rf ./resources/checkpoints/test

## Confirming the checkpoint-path and making sure it does not exist
#echo "SERIALIZATION_DIR: ${SERIALIZATION_DIR}"
#read -p "Continue (Y/N) " continue
#if ! ( [ "${continue}" = "y" ] || [ "${continue}" = "Y" ] ); then exit 1; else echo "Continuing ... "; fi
#
## Simple logic to make sure existing serialization dir is safely deleted
#if [ -d "${SERIALIZATION_DIR}" ]; then
#  echo "SERIALIZATION_DIR EXISTS: ${SERIALIZATION_DIR}"
#  read -p "Delete (Y/N) " delete
#
#  if [ "${delete}" = "y" ] || [ "${delete}" = "Y" ]; then
#    echo "Deleting ${SERIALIZATION_DIR}"
#    rm -r ${SERIALIZATION_DIR}
#  else
#    echo "Not deleting ${SERIALIZATION_DIR}"
#    echo "Cannot continue with non-empty serialization dir. Exiting"
#    exit 1
#  fi
#fi

allennlp train --include-package allennlp_semparse  -s ${SERIALIZATION_DIR} ${CONFIGFILE}
