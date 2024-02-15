#!/usr/bin/env bash
PROJ_ROOT=/PANDAS
export PYTHONPATH=${PROJ_ROOT}
cd ${PROJ_ROOT}

SAVE_PATH=/pandas_experiments
BASE_CKPT=/pandas_experiments/voc_base_phase/model_recent.pth
DATA_PATH=/VOC2012/
SPLIT_PATH=/PANDAS/voc_splits


# GT Prototypes
EXPT_NAME=voc_discovery_10_10_gt_prototypes
LOG_FILE=${SAVE_PATH}/logs/${EXPT_NAME}.log
OUTPUT_DIR=${SAVE_PATH}/${EXPT_NAME}
python -u ${PROJ_ROOT}/ncd_experiment.py \
        --dataset voc \
        --data_path ${DATA_PATH} \
        --split_path ${SPLIT_PATH} \
        --base_num_classes 11 \
        --num_classes 21 \
        --base_detection_ckpt ${BASE_CKPT} \
        --prototype_init gt_prototypes \
        --output_dir ${OUTPUT_DIR} >${LOG_FILE}


# All Clusters
for SEED in 42 43 44;
do
for NUM_CLUSTERS in 20 30 60 110 260 510 1010;
do
    EXPT_NAME=voc_discovery_10_10_clusters_${NUM_CLUSTERS}_all_clusters_seed_${SEED}
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
            --prototype_init cluster_all \
            --output_dir ${OUTPUT_DIR} >${LOG_FILE}
done
done


# Sim={Invert,Cosine,DotProd}, Proba Norm={L1,Softmax}, BG Classifier
NUM_CLUSTERS=250
for SIM in invert cosine dot_prod;
do
for PROBA_NORM in l1 softmax;
do
for SEED in 42 43 44;
do
    EXPT_NAME=voc_discovery_10_10_clusters_${NUM_CLUSTERS}_pandas_${SIM}_${PROBA_NORM}_BG_seed_${SEED}_additional_study
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
            --proba_norm ${PROBA_NORM} \
            --similarity_metric ${SIM} \
            --background_classifier softmax \
            --output_dir ${OUTPUT_DIR} >${LOG_FILE}
done
done
done

# Sim={InvertSquare}, Proba Norm={Softmax}, BG Classifier
NUM_CLUSTERS=250
for SIM in invert_square;
do
for PROBA_NORM in softmax;
do
for SEED in 42 43 44;
do
    EXPT_NAME=voc_discovery_10_10_clusters_${NUM_CLUSTERS}_pandas_${SIM}_${PROBA_NORM}_BG_seed_${SEED}_additional_study
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
            --proba_norm ${PROBA_NORM} \
            --similarity_metric ${SIM} \
            --background_classifier softmax \
            --output_dir ${OUTPUT_DIR} >${LOG_FILE}
done
done
done


# Sim={InvertSquare}, Proba Norm={Softmax}, No BG Classifier
NUM_CLUSTERS=250
for SIM in invert_square;
do
for PROBA_NORM in l1;
do
for SEED in 42 43 44;
do
    EXPT_NAME=voc_discovery_10_10_clusters_${NUM_CLUSTERS}_pandas_${SIM}_${PROBA_NORM}_No_BG_seed_${SEED}_additional_study
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
            --proba_norm ${PROBA_NORM} \
            --similarity_metric ${SIM} \
            --background_classifier none \
            --output_dir ${OUTPUT_DIR} >${LOG_FILE}
done
done
done
