#!/usr/bin/env bash
PROJ_ROOT=/PANDAS
export PYTHONPATH=${PROJ_ROOT}
cd ${PROJ_ROOT}

SAVE_PATH=/pandas_experiments
BASE_CKPT=/pandas_experiments/voc_base_phase/model_recent.pth
DATA_PATH=/VOC2012/
SPLIT_PATH=/PANDAS/voc_splits


# run PANDAS with three k-means seeds and various numbers of novel clusters
for SEED in 42 43 44;
do
for NUM_CLUSTERS in 10 20 50 100 250 500 1000;
do
    EXPT_NAME=voc_discovery_10_10_clusters_${NUM_CLUSTERS}_pandas_seed_${SEED}
    LOG_FILE=${SAVE_PATH}/logs/${EXPT_NAME}.log
    OUTPUT_DIR=${SAVE_PATH}/${EXPT_NAME}
    python -u ${PROJ_ROOT}/ncd_experiment.py \
            --dataset voc \
            --data_path ${DATA_PATH} \
            --split_path ${SPLIT_PATH} \
            --base_num_classes 11 \
            --num_classes 21 \
            --num_clusters ${NUM_CLUSTERS} \
            --base_detection_ckpt ${BASE_CKPT} \
            --kmeans_n_init 10 \
            --kmeans_max_iter 1000 \
            --kmeans_seed ${SEED} \
            --prototype_init pandas \
            --proba_norm l1 \
            --similarity_metric invert_square \
            --background_classifier softmax \
            --output_dir ${OUTPUT_DIR} >${LOG_FILE}
done
done
