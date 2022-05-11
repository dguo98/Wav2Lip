#!/bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --job-name=demiguo-job-1430681
#SBATCH --mem=16G
#SBATCH --open-mode=append
#SBATCH --output=/nlp/scr/demiguo/wav2lip/logs/debug/preprocess.log
#SBATCH --partition=jag-hi
#SBATCH --time=14-0

cd /sailhome/demiguo/demiguo/research/Wav2Lip

srun 'python preprocess.py --data_root /sailhome/demiguo/demiguo-scr/wav2lip/data/mvlrs_v1/main --preprocessed_root /sailhome/demiguo/demiguo-scr/wav2lip/data/lrs2_preprocessed/ --ngpu 1 --batch_size 256'

