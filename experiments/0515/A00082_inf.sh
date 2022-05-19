#!/bin/bash

EXP_ID=A00082
BASE_DIR=/sailhome/demiguo/demiguo/research/Wav2Lip
LOG_DIR=${BASE_DIR}/logs/${EXP_ID}

TRAIN_DATA=${BASE_DIR}/data/timit/videos
TEST_DATA=${BASE_DIR}/data/timit/videos/train/s0

GPU=0
BS=2
WARMUP=4000
EPOCHS=1
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
MODE=slow2
CKPT=logs/A00026/best_val_checkpoint.pt
IMG_MOUTH=1
LMK_LOSS=0.500

cd ${BASE_DIR}
mkdir -p ${LOG_DIR}
CUDA_VISIBLE_DEVICES=${GPU} python A2LMapper/autoreg2/inference.py --train_path ${TRAIN_DATA} --test_path ${TEST_DATA} --output ${LOG_DIR} \
    --batch_size ${BS} --warmup ${WARMUP} --seq_len ${SEQ_LEN} --dropout ${DP} \
    --N ${N} --d_ff ${D_FF} --d_model ${D_MODEL} --h ${H} --use_pose ${USE_POSE} \
    --image_type ${IMG_TYPE} --img_loss ${IMG_LOSS} --image_size ${IMG_SIZE} --mode ${MODE} \
    --lmk_loss ${LMK_LOSS} \
    --load_ckpt ${CKPT} --viz_every 1000 \
    --image_mouth ${IMG_MOUTH} --mouth_box 175 80 215 176 \
    --model transformer --optim noam --epochs ${EPOCHS} #1> ${LOG_DIR}/log.out 2>${LOG_DIR}/log.err

cd ../talking-head-stylegan

CUDA_VISIBLE_DEVICES=${GPU} python get_video_from_latents.py --input ${LOG_DIR}/inference
rm ${LOG_DIR}/inference/*.pt
rm ${LOG_DIR}/inference/*.jpg

#cp -r ${LOG_DIR} ${CLOG_DIR}

#CUDA_VISIBLE_DEVICES=${GPU} python get_video_from_latents.py --input ${LOG_DIR}/tf_inference
#rm ${LOG_DIR}/tf_inference/*.pt
#rm ${LOG_DIR}/tf_inference/*.jpg

#conda deactivate 
