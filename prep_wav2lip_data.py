""" Prep Wav2Lip Data for MLP/Linear Mapping Experiment From Speech To SytleGAN Latent Vector """

import sys
import os
import subprocess
import cv2
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm

import audio
from hparams import hparams as hp
import face_detection

 

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input Directory: *.mp4")
    parser.add_argument("--output", type=str, help="Output Director, one folder per mp4")
    args = parser.parse_args()
    
    vfiles = glob(f"{args.input}/*.mp4")
    assert len(vfiles) > 0, f"no video files in args.input: {args.input}"
    
    # preprocess
    command = f"python preprocess_v2.py --data_root {args.input} --preprocessed_root {args.output}"
    print("Data Preprocesing")
    os.system(command)
    print("Finish Data Preprocessing")
    
    """
    command = f"python wav2lip_extract.py --data_root {args.output} --checkpoint_path checkpoints/wav2lip.pth"
    print("Extract encodings from wav2lip")
    os.system(command)
    print("Finish extracting encodings from wav2lip")
    """
    
    
