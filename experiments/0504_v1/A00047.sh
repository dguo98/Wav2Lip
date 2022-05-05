#!/bin/bash

EXP_ID=A00047
BASE_DIR=/sailhome/demiguo/demiguo/research/Wav2Lip
LOG_DIR=${BASE_DIR}/logs/${EXP_ID}

TRAIN_DATA=${BASE_DIR}/data/timit/videos
TEST_DATA=${BASE_DIR}/data/timit/videos/test/s3

GPU=0
BS=32
WARMUP=4000
EPOCHS=20
H=2
D_MODEL=512
D_FF=512
DP=0.3
N=5
SEQ_LEN=1
USE_POSE=1


cd ${BASE_DIR}
mkdir -p ${LOG_DIR}
CUDA_VISIBLE_DEVICES=${GPU} python A2LMapper/autoreg2/main.py --train_path ${TRAIN_DATA} --test_path ${TEST_DATA} --output ${LOG_DIR} --use_pose ${USE_POSE} \
    --batch_size ${BS} --warmup ${WARMUP} --seq_len ${SEQ_LEN} --dropout ${DP} \
    --N ${N} --d_ff ${D_FF} --d_model ${D_MODEL} --h ${H} --pca tmp/timit_n128/pca.pkl --neutral_path data/timit/videos/neutral.npy \
    --model transformer --optim noam --epochs ${EPOCHS} 1> ${LOG_DIR}/log.out 2>${LOG_DIR}/log.err

export PATH=/nlp/scr/demiguo/miniconda3/bin/:$PATH
eval "$(conda shell.bash hook)"
conda activate e4e_env
cd ../talking-head-stylegan

CUDA_VISIBLE_DEVICES=${GPU} python get_video_from_latents.py --input ${LOG_DIR}/inference
rm ${LOG_DIR}/inference/*.pt
rm ${LOG_DIR}/inference/*.jpg

CUDA_VISIBLE_DEVICES=${GPU} python get_video_from_latents.py --input ${LOG_DIR}/tf_inference
rm ${LOG_DIR}/tf_inference/*.pt
rm ${LOG_DIR}/tf_inference/*.jpg

conda deactivate 
