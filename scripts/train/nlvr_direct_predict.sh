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
export DEV_DATA=../../nfs2_nitishg/data/nlvr/processed/dev_grouped.json

# MODEL PATH
SERIALIZATION_DIR=../../nfs2_nitishg/checkpoints/mml_parser/nlvr/agenda_true_MDS-14
MODEL_TAR_GZ=${SERIALIZATION_DIR}/model.tar.gz

mkdir ${SERIALIZATION_DIR}/predictions
OUTPUT_PREDICTION_PATH=${SERIALIZATION_DIR}/predictions/dev-visualize.jsonl
OUTPUT_METRICS_PATH=${SERIALIZATION_DIR}/predictions/dev-metrics.json

PREDICTOR=nlvr-parser-visualize

allennlp predict --output-file ${OUTPUT_PREDICTION_PATH} \
                 --batch-size 4 --silent \
                 --cuda-device ${CUDA} \
                 --predictor ${PREDICTOR} \
                 --include-package allennlp_semparse \
                 --overrides "{"model": {"max_decoding_steps": 14}}" \
                 ${MODEL_TAR_GZ} ${DEV_DATA}


allennlp evaluate --output-file ${OUTPUT_METRICS_PATH} \
                  --cuda-device ${CUDA} \
                  --include-package allennlp_semparse \
                  --overrides "{"model": {"max_decoding_steps": 14}}" \
                  ${MODEL_TAR_GZ} ${DEV_DATA}


echo -e "Metrics written to: ${OUTPUT_METRICS_PATH}"
echo -e "Predictions written to: ${OUTPUT_PREDICTION_PATH}"