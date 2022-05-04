#!/bin/bash

EXP_ID=B00084
BASE_DIR=/sailhome/demiguo/demiguo/research/Wav2Lip
LOG_DIR=${BASE_DIR}/logs/${EXP_ID}

TRAIN_DATA=${BASE_DIR}/data/p_words/p_train
TEST_DATA=${BASE_DIR}/data/p_words/p_test

GPU=0
BS=32
LR=0.00100
LR_P=10
N_LAYER=1
EPOCHS=100
HIDDEN_DIM=1024
WD=0.0000000


cd ${BASE_DIR}
mkdir -p ${LOG_DIR}
CUDA_VISIBLE_DEVICES=${GPU} python A2LMapper/autoreg/main.py --train_path ${TRAIN_DATA} --test_path ${TEST_DATA} --output ${LOG_DIR} \
    --batch_size ${BS} --lr ${LR} --lr_patience ${LR_P} --wd ${WD} \
    --seq_len 1 --model mlp --optim adam \
    --nlayer ${N_LAYER} --hidden_dim ${HIDDEN_DIM} --epochs ${EPOCHS} 1> ${LOG_DIR}/log.out 2>${LOG_DIR}/log.err

export PATH=/nlp/scr/demiguo/miniconda3/bin/:$PATH
eval "$(conda shell.bash hook)"
conda activate e4e_env
cd ../talking-head-stylegan
CUDA_VISIBLE_DEVICES=${GPU} python get_video_from_latents.py --input ${LOG_DIR}/inference
CUDA_VISIBLE_DEVICES=${GPU} python get_video_from_latents.py --input ${LOG_DIR}/tf_inference
conda deactivate 
