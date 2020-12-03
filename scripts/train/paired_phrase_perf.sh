#!/usr/bin/env

export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SPLIT=dev

BASIC_ERM=./resources/checkpoints/nlvr/basicerm_SORT_SD_01/SEED_21/ERM/Iter5_MDS22/predictions/${SPLIT}-predictions.jsonl

PAIRED_V1=./resources/checkpoints/nlvr/pairedv1_SORT_SD_T07_P1M2/SEED_42/MML/Iter5_MDS22/predictions/${SPLIT}-predictions.jsonl

PAIRED_V2=./resources/checkpoints/nlvr/pairedv2_NEWLANG_T07_P1M1/SEED_42/ERM/Iter5_MDS22/predictions/${SPLIT}-predictions.jsonl
PAIRED_V2RE=./resources/checkpoints/nlvr/pairedv2_NEWLANG_T07_P1M2-RE/SEED_42/ERM/Iter5_MDS22/predictions/${SPLIT}-predictions.jsonl

PAIRED_V3=./resources/checkpoints/nlvr/pairedv3_NEWLANG_T07_P1M2/SEED_42/ERM/Iter5_MDS22/predictions/${SPLIT}-predictions.jsonl

COMPGEN_BASIC=./resources/checkpoints/nlvr/comp-gen/absstr_v1/basicerm_NEWLANG/SEED_1337/ERM/Iter5_MDS22/predictions/${SPLIT}-predictions.jsonl

COMPGEN_PAIRED_V1=./resources/checkpoints/nlvr/comp-gen/absstr_v1/pairederm_NEWLANG_T07_P1M2/SEED_21/ERM/Iter5_MDS22/predictions/${SPLIT}-predictions.jsonl

COMPGEN_PAIRED_V2=./resources/checkpoints/nlvr/comp-gen/absstr_v1/pairedv2_NEWLANG_T07_P1M1/SEED_21/ERM/Iter5_MDS22/predictions/${SPLIT}-predictions.jsonl

COMPGEN_PAIRED_V3=./resources/checkpoints/nlvr/comp-gen/absstr_v1/pairedv3_NEWLANG_T07_P1M1/SEED_1337/ERM/Iter5_MDS22/predictions/${SPLIT}-predictions.jsonl


VERSION=v2
PHRASES=./scripts/nlvr_v2/data/paired_phrases_${VERSION}.json


python scripts/nlvr_v2/analysis/phrase_based_performance.py ${PHRASES} ${BASIC_ERM} ${PAIRED_V2RE}

