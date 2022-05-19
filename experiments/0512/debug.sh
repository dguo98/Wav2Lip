#!/bin/bash

EXP_ID=A00072_debug
BASE_DIR=/sailhome/demiguo/demiguo/research/Wav2Lip

LOG_DIR=/jagupard29/scr0/demiguo/${EXP_ID}
#LOG_DIR=${BASE_DIR}/logs/${EXP_ID}

TRAIN_DATA=${BASE_DIR}/data/timit/videos
TEST_DATA=${BASE_DIR}/data/timit/videos/test/s3

GPU=0
BS=4
WARMUP=4000
EPOCHS=3
H=2
D_MODEL=512
D_FF=512
DP=0.3
N=5
SEQ_LEN=1
USE_POSE=0
IMG_TYPE=gt
IMG_LOSS=0.800
IMG_SIZE=256
MODE=slow
CKPT=logs/A00026/best_val_checkpoint.pt
IMG_MOUTH=1

cd ${BASE_DIR}
mkdir -p ${LOG_DIR}
CUDA_VISIBLE_DEVICES=${GPU} python A2LMapper/autoreg2/main.py --train_path ${TRAIN_DATA} --test_path ${TEST_DATA} --output ${LOG_DIR} \
    --batch_size ${BS} --warmup ${WARMUP} --seq_len ${SEQ_LEN} --dropout ${DP} \
    --N ${N} --d_ff ${D_FF} --d_model ${D_MODEL} --h ${H} --use_pose ${USE_POSE} \
    --image_type ${IMG_TYPE} --img_loss ${IMG_LOSS} --image_size ${IMG_SIZE} --mode ${MODE} \
    --load_ckpt ${CKPT} --viz_every 1000 \
    --image_mouth ${IMG_MOUTH} --mouth_box 175 80 215 176 \
    --model transformer --optim noam --epochs ${EPOCHS} #1> ${LOG_DIR}/log.out 2>${LOG_DIR}/log.err

cd ../talking-head-stylegan

CUDA_VISIBLE_DEVICES=${GPU} python get_video_from_latents.py --input ${LOG_DIR}/inference
rm ${LOG_DIR}/inference/*.pt
rm ${LOG_DIR}/inference/*.jpg

#CUDA_VISIBLE_DEVICES=${GPU} python get_video_from_latents.py --input ${LOG_DIR}/tf_inference
#rm ${LOG_DIR}/tf_inference/*.pt
#rm ${LOG_DIR}/tf_inference/*.jpg

#conda deactivate 
