""" Prep Wav2Lip Data for MLP/Linear Mapping Experiment From Speech To SytleGAN Latent Vector """

import sys
import os
import shutil
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
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    # NB(demi): need to make sure audio sample rate is 16k, and video sample rate is 29.97
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input Directory: *.mp4")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ngpu", type=int, default=1)
    args = parser.parse_args()
    
    # input is the containing folder of different mp4s files
    folder = args.input
    mp4s = sorted(glob(f"{args.input}/*.mp4"))
    for t, mp4 in enumerate(mp4s):
        if t % args.ngpu != args.gpu:
            continue
        args.input = mp4.replace(".mp4", "")  # hack
        print("input is:", args.input)

        # preprocessing
        print("data processing")
        command = f"python preprocess_v2.py --data_root {args.input} --preprocessed_root {os.path.dirname(args.input)}"
        print(command)
        if not os.path.exists(f"{args.input}/wav2lip_faces") and not os.path.exists(f"{args.input}/wav2lip.npy"):
            os.system(command)
        print("finish data preprocessing")
    
        # audio preprocess
        command = f"ffmpeg -y -i {args.input}/audio-raw.wav -ar 16000 {args.input}/audio.wav"
        print(command)
        if not os.path.exists(f"{args.input}/audio.wav"):
            os.system(command)
                
        command = f"python wav2lip_extract.py --data_root {args.input} --checkpoint_path checkpoints/wav2lip.pth"
        print("Extract encodings from wav2lip")
        if not os.path.exists(f"{args.input}/wav2lip_latents") and not os.path.exists(f"{args.input}/wav2lip.npy"):
            os.system(command)
        print("Finish extracting encodings from wav2lip")
            
        """
        print("extract latents")
        cmd = f"cd ../talking-head-stylegan;python scripts/inference.py --images_dir ../Wav2Lip/{args.input}/frames --save_dir ../Wav2Lip/{args.input}/frame_latents e4e_ffhq_encode.pt --align"
        print(cmd)
        if not os.path.exists(f"{args.input}/frame_latents") and not os.path.exists(f"{args.input}/frame.npy"):
            os.system(cmd)
        print("finish extracting latents")
        """
          
        # save in combined npy
        if not os.path.exists(f"{args.input}/wav2lip.npy"):
            audio_files = sorted(glob(f"{args.input}/wav2lip_latents/wav2lip_audio_[0123456789]*.pt"))
            audio_vecs = []
            for f in tqdm(audio_files, desc="load audio"):
                audio_vec = torch.load(f)
                audio_vecs.append(audio_vec.detach().cpu().numpy().reshape(-1))
            audio_vecs = np.stack(audio_vecs, axis=0)
            np.save(f"{args.input}/wav2lip.npy", audio_vecs)
        if os.path.exists(f"{args.input}/wav2lip.npy"):
            if os.path.exists(f"{args.input}/wav2lip_latents"):
                shutil.rmtree(f"{args.input}/wav2lip_latents")
            if os.path.exists(f"{args.input}/wav2lip_faces"):
                shutil.rmtree(f"{args.input}/wav2lip_faces")

        """ 
        if not os.path.exists(f"{args.input}/frame.npy"):
            latent_files = sorted(glob(f"{args.input}/frame_latents/latents_frame*.npy"))
            latent_vecs = []
            for f in tqdm(latent_files, desc="load latents"):
                latent_vec = np.load(f)
                latent_vecs.append(latent_vec)
            latent_vecs = np.stack(latent_vecs, axis=0)
            print("latent vecs shape=", latent_vecs.shape)
            np.save(f"{args.input}/frame.npy", latent_vecs)
        """
           
        """
        # assume frame 000000 is neutral
        if (t==0):
            os.system(f"cp {args.input}/frames/frame_000000.jpg {folder}/neutral.jpg")
            os.system(f"cp {args.input}/frame_latents/latents_frame_000000.npy {folder}/neutral.npy")
        """
    
        if os.path.exists(f"{args.input}/frame.npy") and os.path.exists(f"{args.input}/frame_latent"):
            shutil.rmtree(f"{args.input}/frame_latents")
        
