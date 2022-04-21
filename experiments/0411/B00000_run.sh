EXP_ID=B00000
SCRIPT_DIR=experiments/0411/${EXP_ID}
LOG_DIR=logs/${EXP_ID}
mkdir -p ${LOG_DIR}


python wav2lip_train.py --data_root ./data/lrs2_preprocessed/ --checkpoint_dir ./logs/${EXP_ID}/checkpoints --syncnet_checkpoint_path ./checkpoints/lipsync_expert.pth 1>${LOG_DIR}/log.out 2>${LOG_DIR}/log.err

