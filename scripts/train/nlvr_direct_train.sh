#!/usr/bin/env

export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OPENMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

export CUDA=0

INCLUDE_PACKAGE=allennlp_semparse
CONFIGFILE=training_config/nlvr_direct_parser.jsonnet

# DATA PATH
export TRAIN_DATA=../../nfs2_nitishg/data/nlvr/processed/train_grouped_lfs.json  # _wagenda.json
export DEV_DATA=../../nfs2_nitishg/data/nlvr/processed/dev_grouped.json

# HYPER-PARAMETERS
AGENDA=false

# SERIALIZATION PATH
CHECKPOINT_ROOT=../../nfs2_nitishg/checkpoints
MODEL_DIR=mml_parser/nlvr
PARAMETERS=agenda_${AGENDA}
SERIALIZATION_DIR=${CHECKPOINT_ROOT}/${MODEL_DIR}/${PARAMETERS}_MDS-12

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
