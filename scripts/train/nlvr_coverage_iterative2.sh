#!/usr/bin/env

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OPENMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

INCLUDE_PACKAGE=allennlp_semparse
export CUDA=-1
mkdir -p ./resources/data/nlvr/processed/ERM_DATAGEN

export DEV_DATA=./resources/data/nlvr/processed/dev_grouped.json
export SEED=1337
export EPOCHS=50

###   ITERATION 0 -- MML parser   ###
ITER=0
CONFIGFILE=training_config/nlvr_mml_parser.jsonnet
# DATA PATH
export DATADIR=agenda_v6_ML11
export TRAIN_DATA=./resources/data/nlvr/processed/${DATADIR}/train_grouped.json
# HYPER-PARAMETERS
export MDS=12
# SERIALIZATION PATH
MML_0_SERDIR=./resources/checkpoints/mml_parser/nlvr/agenda_v6_ML11/S_${SEED}-MDS_${MDS}-Iter${ITER}
allennlp train --include-package allennlp_semparse  -s ${MML_0_SERDIR} ${CONFIGFILE}
######

###   ITERATION 1 -- ERM parser   ###
ITER=1
CONFIGFILE=training_config/nlvr_erm_parser.jsonnet
# DATA PATH
export TRAIN_DATA=./resources/data/nlvr/processed/train_grouped.json
# HYPER-PARAMETERS
export MML_MODEL_TAR=${MML_0_SERDIR}/model.tar.gz
export MDS=14
# SERIALIZATION PATH
ERM_1_SERDIR=./resources/checkpoints/erm_parser/nlvr/S_${SEED}-MDS_${MDS}-Iter${ITER}
allennlp train --include-package allennlp_semparse  -s ${ERM_1_SERDIR} ${CONFIGFILE}
######

### GENERATE DATA from ERM
INPUT_FILE=./resources/data/nlvr/processed/train_grouped.json
OUTPUT_FILE=./resources/data/nlvr/processed/ERM_DATAGEN/train_ermS${SEED}_${ITER}.json.json
MODEL_TAR=${ERM_1_SERDIR}/model.tar.gz
time python scripts/nlvr_v2/generate_data_from_erm_model.py ${INPUT_FILE} ${OUTPUT_FILE} ${MODEL_TAR}
######

###   ITERATION 1 -- MML parser   ###
CONFIGFILE=training_config/nlvr_mml_parser.jsonnet
export TRAIN_DATA=./resources/data/nlvr/processed/ERM_DATAGEN/train_ermS${SEED}_${ITER}.json.json
# HYPER-PARAMETERS
export MDS=14
# SERIALIZATION PATH
MML_1_SERDIR=./resources/checkpoints/mml_parser/nlvr/agenda_v6_ML11/S_${SEED}-MDS_${MDS}-Iter${ITER}
allennlp train --include-package allennlp_semparse  -s ${MML_1_SERDIR} ${CONFIGFILE}

##############################
###   ITERATION 2 -- ERM parser   ###
ITER=2
CONFIGFILE=training_config/nlvr_erm_parser.jsonnet
# DATA PATH
export TRAIN_DATA=./resources/data/nlvr/processed/train_grouped.json
# HYPER-PARAMETERS
export MML_MODEL_TAR=${MML_1_SERDIR}/model.tar.gz
export MDS=16
# SERIALIZATION PATH
ERM_2_SERDIR=./resources/checkpoints/erm_parser/nlvr/S_${SEED}-MDS_${MDS}-Iter${ITER}
allennlp train --include-package allennlp_semparse  -s ${ERM_2_SERDIR} ${CONFIGFILE}
######

### GENERATE DATA from ERM
INPUT_FILE=./resources/data/nlvr/processed/train_grouped.json
OUTPUT_FILE=./resources/data/nlvr/processed/ERM_DATAGEN/train_ermS${SEED}_${ITER}.json.json
MODEL_TAR=${ERM_2_SERDIR}/model.tar.gz
time python scripts/nlvr_v2/generate_data_from_erm_model.py ${INPUT_FILE} ${OUTPUT_FILE} ${MODEL_TAR}
######

###   ITERATION 2 -- MML parser   ###
CONFIGFILE=training_config/nlvr_mml_parser.jsonnet
export TRAIN_DATA=./resources/data/nlvr/processed/ERM_DATAGEN/train_ermS${SEED}_${ITER}.json.json
# HYPER-PARAMETERS
export MDS=16
# SERIALIZATION PATH
MML_2_SERDIR=./resources/checkpoints/mml_parser/nlvr/agenda_v6_ML11/S_${SEED}-MDS_${MDS}-Iter${ITER}
allennlp train --include-package allennlp_semparse  -s ${MML_2_SERDIR} ${CONFIGFILE}

##############################
###   ITERATION 3 -- ERM parser   ###
ITER=3
CONFIGFILE=training_config/nlvr_erm_parser.jsonnet
# DATA PATH
export TRAIN_DATA=./resources/data/nlvr/processed/train_grouped.json
# HYPER-PARAMETERS
export MML_MODEL_TAR=${MML_2_SERDIR}/model.tar.gz
export MDS=18
# SERIALIZATION PATH
ERM_3_SERDIR=./resources/checkpoints/erm_parser/nlvr/S_${SEED}-MDS_${MDS}-Iter${ITER}
allennlp train --include-package allennlp_semparse  -s ${ERM_3_SERDIR} ${CONFIGFILE}
######

### GENERATE DATA from ERM
INPUT_FILE=./resources/data/nlvr/processed/train_grouped.json
OUTPUT_FILE=./resources/data/nlvr/processed/ERM_DATAGEN/train_ermS${SEED}_${ITER}.json.json
MODEL_TAR=${ERM_3_SERDIR}/model.tar.gz
time python scripts/nlvr_v2/generate_data_from_erm_model.py ${INPUT_FILE} ${OUTPUT_FILE} ${MODEL_TAR}
######

###   ITERATION 3 -- MML parser   ###
CONFIGFILE=training_config/nlvr_mml_parser.jsonnet
export TRAIN_DATA=./resources/data/nlvr/processed/ERM_DATAGEN/train_ermS${SEED}_${ITER}.json.json
# HYPER-PARAMETERS
export MDS=18
# SERIALIZATION PATH
MML_3_SERDIR=./resources/checkpoints/mml_parser/nlvr/agenda_v6_ML11/S_${SEED}-MDS_${MDS}-Iter${ITER}
allennlp train --include-package allennlp_semparse  -s ${MML_3_SERDIR} ${CONFIGFILE}

##############################
###   ITERATION 4 -- ERM parser   ###
ITER=4
CONFIGFILE=training_config/nlvr_erm_parser.jsonnet
# DATA PATH
export TRAIN_DATA=./resources/data/nlvr/processed/train_grouped.json
# HYPER-PARAMETERS
export MML_MODEL_TAR=${MML_3_SERDIR}/model.tar.gz
export MDS=20
# SERIALIZATION PATH
ERM_4_SERDIR=./resources/checkpoints/erm_parser/nlvr/S_${SEED}-MDS_${MDS}-Iter${ITER}
allennlp train --include-package allennlp_semparse  -s ${ERM_4_SERDIR} ${CONFIGFILE}
######

### GENERATE DATA from ERM
INPUT_FILE=./resources/data/nlvr/processed/train_grouped.json
OUTPUT_FILE=./resources/data/nlvr/processed/ERM_DATAGEN/train_ermS${SEED}_${ITER}.json.json
MODEL_TAR=${ERM_4_SERDIR}/model.tar.gz
time python scripts/nlvr_v2/generate_data_from_erm_model.py ${INPUT_FILE} ${OUTPUT_FILE} ${MODEL_TAR}
######

###   ITERATION 3 -- MML parser   ###
CONFIGFILE=training_config/nlvr_mml_parser.jsonnet
export TRAIN_DATA=./resources/data/nlvr/processed/ERM_DATAGEN/train_ermS${SEED}_${ITER}.json.json
# HYPER-PARAMETERS
export MDS=20
# SERIALIZATION PATH
MML_4_SERDIR=./resources/checkpoints/mml_parser/nlvr/agenda_v6_ML11/S_${SEED}-MDS_${MDS}-Iter${ITER}
allennlp train --include-package allennlp_semparse  -s ${MML_4_SERDIR} ${CONFIGFILE}