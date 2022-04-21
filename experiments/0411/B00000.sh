#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=demiguo-job-wav2lip
#SBATCH --mem=40G
#SBATCH --open-mode=append
#SBATCH --output=sbatch.log
#SBATCH --partition=jag-hi
#SBATCH --time=14-0

srun bash ./experiments/0411/B00000_run.sh


