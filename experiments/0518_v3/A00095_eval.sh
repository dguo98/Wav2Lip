#!/bin/bash

EXP_ID=A00095
BASE_DIR=/sailhome/demiguo/demiguo/research/Wav2Lip
LOG_DIR=${BASE_DIR}/logs/${EXP_ID}

TRAIN_DATA=/u/nlp/data/timit_videos_v2/videos
TEST_DATA=/u/nlp/data/timit_videos_v2/videos/test/s3

GPU=0
BS=2
WARMUP=4000
EPOCHS=20
H=2
D_MODEL=512
D_FF=512
DP=0.3
N=5
SEQ_LEN=1
USE_POSE=0
IMG_TYPE=gt
IMG_LOSS=0.000
IMG_SIZE=256
MODE=default
CKPT=none
IMG_MOUTH=1
LMK_LOSS=0.000
PERC_LOSS=0.000

cd ${BASE_DIR}
mkdir -p ${LOG_DIR}
CUDA_VISIBLE_DEVICES=${GPU} python A2LMapper/autoreg2/evaluate.py --train_path ${TRAIN_DATA} --test_path ${TEST_DATA} --output ${LOG_DIR} \
    --batch_size ${BS} --warmup ${WARMUP} --seq_len ${SEQ_LEN} --dropout ${DP} \
    --N ${N} --d_ff ${D_FF} --d_model ${D_MODEL} --h ${H} --use_pose ${USE_POSE} \
    --image_type ${IMG_TYPE} --img_loss ${IMG_LOSS} --image_size ${IMG_SIZE} --mode ${MODE} \
    --lmk_loss ${LMK_LOSS} --perceptual_loss ${PERC_LOSS} \
    --viz_every 2000 --save_every 4 \
    --image_mouth ${IMG_MOUTH} --mouth_box 175 80 215 176 \
    --model transformer --optim noam --epochs ${EPOCHS} #1> ${LOG_DIR}/eval_log.out 2>${LOG_DIR}/eval_log.err

