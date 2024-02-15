#!/usr/bin/env bash
PROJ_ROOT=/PANDAS
export PYTHONPATH=${PROJ_ROOT}
cd ${PROJ_ROOT}

SAVE_PATH=/pandas_experiments
SSL_CKPT=/moco_v2_800ep_pretrain.pth.tar
IMAGES_PATH=/Microsoft-COCO-2017
COCO_HALF_ANNOTATION_ROOT=/Microsoft-COCO-2017/annotations/

LR=0.02
EXPT_NAME=coco_half_base_phase
LOG_FILE=${SAVE_PATH}/logs/${EXPT_NAME}.log
OUTPUT_DIR=${SAVE_PATH}/${EXPT_NAME}
python -u train_base_network.py \
        --backbone_ckpt ${SSL_CKPT} \
        --data_path ${IMAGES_PATH} \
        --coco_anno_root ${COCO_HALF_ANNOTATION_ROOT} \
        --dataset coco_half \
        --num_classes 91 \
        --batch-size 16 \
        --epochs 26 \
        --lr ${LR} \
        --freeze_bn 0 \
        --output_dir ${OUTPUT_DIR} >${LOG_FILE}
        