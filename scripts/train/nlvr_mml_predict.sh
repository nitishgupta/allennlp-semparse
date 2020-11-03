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
export DEV_DATA=./resources/data/nlvr/processed/dev_grouped.json

# MODEL PATH
SERIALIZATION_DIR=./resources/checkpoints/mml_parser/nlvr/agenda_v6_ML11/MDS_18/S_42-MML
MODEL_TAR_GZ=${SERIALIZATION_DIR}/model.tar.gz

mkdir ${SERIALIZATION_DIR}/predictions
OUTPUT_VISUALIZE_PATH=${SERIALIZATION_DIR}/predictions/dev-visualize.jsonl
OUTPUT_PREDICTION_PATH=${SERIALIZATION_DIR}/predictions/dev-predictions.jsonl
OUTPUT_METRICS_PATH=${SERIALIZATION_DIR}/predictions/dev-metrics.json

VISUALIZE_PREDICTOR=nlvr-parser-visualize
PREDICTION_PREDICTOR=nlvr-parser-predictions


allennlp predict --output-file ${OUTPUT_VISUALIZE_PATH} \
                 --batch-size 4 --silent \
                 --cuda-device ${CUDA} \
                 --predictor ${VISUALIZE_PREDICTOR} \
                 --include-package allennlp_semparse \
                 ${MODEL_TAR_GZ} ${DEV_DATA}


allennlp predict --output-file ${OUTPUT_PREDICTION_PATH} \
                 --batch-size 4 --silent \
                 --cuda-device ${CUDA} \
                 --predictor ${PREDICTION_PREDICTOR} \
                 --include-package allennlp_semparse \
                 ${MODEL_TAR_GZ} ${DEV_DATA}


allennlp evaluate --output-file ${OUTPUT_METRICS_PATH} \
                  --cuda-device ${CUDA} \
                  --include-package allennlp_semparse \
                  ${MODEL_TAR_GZ} ${DEV_DATA}

# --overrides "{"model": {"max_decoding_steps": 14}}" \

echo -e "Metrics written to: ${OUTPUT_METRICS_PATH}"
echo -e "Predictions written to: ${OUTPUT_PREDICTION_PATH}"
echo -e "Visualizations written to: ${OUTPUT_VISUALIZE_PATH}"