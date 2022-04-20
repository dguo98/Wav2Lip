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
import torch
from hparams import hparams as hp
import face_detection

 

    
if __name__ == "__main__":
    # NB(demi): need to make sure audio sample rate is 16k, and video sample rate is 29.97
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input Directory: *.mp4")
    args = parser.parse_args()
    
    """
    # preprocess
    command = f"python preprocess_v2.py --data_root {args.input} --preprocessed_root {os.path.dirname(args.input)}"
    print("Data Preprocesing")
    print(command)
    os.system(command)
    print("Finish Data Preprocessing")
    
    # audio preprocess
    command = f"ffmpeg -i {args.input}/audio-raw.wav -ar 16000 {args.input}/audio.wav"
    print(command)
    os.system(command)
            
    command = f"python wav2lip_extract.py --data_root {args.input} --checkpoint_path checkpoints/wav2lip.pth"
    print("Extract encodings from wav2lip")
    os.system(command)
    print("Finish extracting encodings from wav2lip")
    """ 
       
    """
    # NB(demi): need to copy neutral latent, or generate latent
    print("in talking head directory")
    cmd = f"cd ../talking-head-stylegan;conda activate e4e_env;python scripts/inference.py --images_dir ../Wav2Lip/{args.input}/frames --save_dir ../Wav2Lip/{args.input}/frame_latents e4e_ffhq_encode.pt --align;conda deactivate"
    print(cmd)
    #os.system(cmd)
    print("assume frame 000000 is neutral")
    print(f"{args.input}/frame_latents/latents_frame_000000.pt {args.input}/frame_latents/neutral.pt")
    #os.system(f"{args.input}/frame_latents/frame_latents_000000.pt {args.input}/frame_latents/neutral.pt")
    
    """
    # save in combined npy
    audio_files = sorted(glob(f"{args.input}/wav2lip_latents/wav2lip_audio_[0123456789]*.pt"))
    audio_vecs = []
    for f in tqdm(audio_files, desc="load audio"):
        audio_vec = torch.load(f)
        audio_vecs.append(audio_vec.detach().cpu().numpy().reshape(-1))
    audio_vecs = np.stack(audio_vecs, axis=0)
    np.save(f"{args.input}/wav2lip.npy", audio_vecs)

    latent_files = sorted(glob(f"{args.input}/frame_latents/latents_frame*.npy"))
    latent_vecs = []
    for f in tqdm(latent_files, desc="load latents"):
        latent_vec = np.load(f)
        latent_vecs.append(latent_vec)
    latent_vecs = np.stack(latent_vecs, axis=0)
    print("latent vecs shape=", latent_vecs.shape)
    np.save(f"{args.input}/frame.npy", latent_vecs)
