#!/bin/bash
# Directories
# export DEEPCLEAN_CONTAINER_ROOT=~/images/deepclean
# export DATA_DIR=~/deepclean/data
# export RESULTS_DIR=~/deepclean/results
# GPU Settings
export CUDA_VISIBLE_DEVICES=0
export GPU_INDEX=0
# export CUDA_VISIBLE_DEVICES=1
# export GPU_INDEX=1
export CUDA_LAUNCH_BLOCKING=1
# Run Training Task
export DEEPCLEAN_IFO=H1
poetry run law run deepclean.tasks.Train \
    --dev \
    --image train.sif \
    --gpus $GPU_INDEX \
    --data-fname $DATA_DIR/deepclean-1378402219-3072.hdf5 \
    --train-config ${HOME}/deepcleanv2/projects/train/config_dcprod.yaml \
    --output-dir ${RESULTS_DIR}/O4-CDC_120Hz_offline
