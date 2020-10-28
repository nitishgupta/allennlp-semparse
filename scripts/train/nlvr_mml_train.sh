#!/usr/bin/env

export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OPENMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

export CUDA=0

INCLUDE_PACKAGE=allennlp_semparse
CONFIGFILE=training_config/nlvr_mml_parser.jsonnet

# DATA PATH
export DATADIR=agenda_v2SC_partial_False
export TRAIN_DATA=../../nfs2_nitishg/data/nlvr/processed/${DATADIR}/train_grouped.json
# train_grouped.json
export DEV_DATA=../../nfs2_nitishg/data/nlvr/processed/dev_grouped.json

# HYPER-PARAMETERS
export MDS=18

export SEED=42

# SERIALIZATION PATH
CHECKPOINT_ROOT=../../nfs2_nitishg/checkpoints
MODEL_DIR=mml_parser/nlvr/${DATADIR}
PARAMETERS=MDS_${MDS}/S_${SEED}
SERIALIZATION_DIR=${CHECKPOINT_ROOT}/${MODEL_DIR}/${PARAMETERS}

# SERIALIZATION_DIR=${CHECKPOINT_ROOT}/test

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
