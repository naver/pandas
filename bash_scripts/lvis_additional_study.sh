#!/usr/bin/env bash
PROJ_ROOT=/PANDAS
export PYTHONPATH=${PROJ_ROOT}
cd ${PROJ_ROOT}

SAVE_PATH=/pandas_experiments
BASE_CKPT=/pandas_experiments/coco_half_base_phase/model_recent.pth
DATA_PATH=/Microsoft-COCO-2017
COCO_HALF_TRAIN_JSON=/Microsoft-COCO-2017/annotations/coco_half_train_lvis_ann.json
COCO_SECOND_HALF_TRAIN_JSON=/Microsoft-COCO-2017/annotations/coco_second_half.json
LVIS_TRAIN_JSON=/LVISv1/lvis_v1_train.json
LVIS_VAL_JSON=/LVISv1/lvis_v1_val.json

BATCH_SIZE=16
KMEANS_INIT=10
KMEANS_ITERS=1000

# PANDAS No Background (load clusters from main experiment)
for NUM_CLUSTERS in 5000;
do
EXPT_NAME=lvis_discovery_clusters_${NUM_CLUSTERS}_pandas_no_BG
LOG_FILE=${SAVE_PATH}/logs/${EXPT_NAME}.log
OUTPUT_DIR=${SAVE_PATH}/${EXPT_NAME}
python -u ${PROJ_ROOT}/ncd_experiment.py \
        --dataset lvis \
        --ncd_ckpt ${SAVE_PATH}/lvis_discovery_clusters_${NUM_CLUSTERS}_pandas/ncd_model.pth \
        --data_path ${DATA_PATH} \
        --coco_half_train_json ${COCO_HALF_TRAIN_JSON} \
        --coco_second_half_train_json ${COCO_SECOND_HALF_TRAIN_JSON} \
        --lvis_train_json ${LVIS_TRAIN_JSON} \
        --lvis_val_json ${LVIS_VAL_JSON} \
        --base_num_classes 91 \
        --num_classes 1204 \
        --num_clusters ${NUM_CLUSTERS} \
        --base_detection_ckpt ${BASE_CKPT} \
        --dets_per_image 300 \
        --score_thresh 0.0 \
        --batch_size ${BATCH_SIZE} \
        --kmeans_n_init ${KMEANS_INIT} \
        --kmeans_max_iter ${KMEANS_ITERS} \
        --prototype_init pandas \
        --proba_norm l1 \
        --similarity_metric invert_square \
        --background_classifier none \
        --output_dir ${OUTPUT_DIR} >${LOG_FILE}
done
