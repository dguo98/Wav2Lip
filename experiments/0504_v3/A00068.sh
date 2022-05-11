#!/bin/bash

EXP_ID=A00068
BASE_DIR=/sailhome/demiguo/demiguo/research/Wav2Lip
LOG_DIR=${BASE_DIR}/logs/${EXP_ID}

TRAIN_DATA=${BASE_DIR}/data/timit/videos
TEST_DATA=${BASE_DIR}/data/timit/videos/test/s3

GPU=0
BS=32
WARMUP=2000
EPOCHS=20
H=2
D_MODEL=512
D_FF=512
DP=0.3
N=5
SEQ_LEN=1
USE_POSE=0
LATENT=stylespace
C_W=0.20000
C_MASK=${BASE_DIR}/logs/stylespace_local/mouth_mask.npy


cd ${BASE_DIR}
mkdir -p ${LOG_DIR}
CUDA_VISIBLE_DEVICES=${GPU} python A2LMapper/autoreg2/main.py --train_path ${TRAIN_DATA} --test_path ${TEST_DATA} --output ${LOG_DIR} \
    --batch_size ${BS} --warmup ${WARMUP} --seq_len ${SEQ_LEN} --dropout ${DP} \
    --N ${N} --d_ff ${D_FF} --d_model ${D_MODEL} --h ${H} --use_pose ${USE_POSE} --latent_type ${LATENT} \
    --channel_weight ${C_W} --channel_mask ${C_MASK} \
    --model transformer --optim noam --epochs ${EPOCHS} 1> ${LOG_DIR}/log.out 2>${LOG_DIR}/log.err

#export PATH=/nlp/scr/demiguo/miniconda3/bin/:$PATH
#eval "$(conda shell.bash hook)"

CUDA_VISIBLE_DEVICES=${GPU} python A2LMapper/autoreg2/stylespace/convert_s2img.py --input ${LOG_DIR}/inference --fps 25
rm ${LOG_DIR}/inference/*.npy
rm ${LOG_DIR}/inference/*.jpg

CUDA_VISIBLE_DEVICES=${GPU} python A2LMapper/autoreg2/stylespace/convert_s2img.py --input ${LOG_DIR}/tf_inference --fps 25
rm ${LOG_DIR}/tf_inference/*.npy
rm ${LOG_DIR}/tf_inference/*.jpg

#conda deactivate 
