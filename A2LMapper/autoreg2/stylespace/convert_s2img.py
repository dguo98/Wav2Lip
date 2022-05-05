import os
import sys
import numpy as np
import argparse
import cv2
import torch
import dlib
from glob import glob
from tqdm import tqdm
from PIL import Image

sys.path.append(".")
from manipulate import Manipulator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="../Wav2Lip/output/debug/inference")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--bsz", type=int, default=64)
    args = parser.parse_args()
    
    dataset_name = "ffhq"
    M=Manipulator(dataset_name=dataset_name)
    np.set_printoptions(suppress=True)

    # convert latents to images
    s_shapes = [512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,256,256,256,128,128,128,64,64,64,32,32]
    s_latents = np.load(f"{args.input}/predict_stylespace.npy")

    # HACK(demi)
    s_latents = s_latents[:300]

    n = len(s_latents)
    for i in range(0, n, args.bsz):
        j = min(n, i+args.bsz)
        
        # break down into layers, formatting
        tmp_s = []
        cur = 0
        for shape in s_shapes:
            tmp_s.append(s_latents[i:j, cur:cur+shape])
            cur = cur+shape
        assert cur == s_latents.shape[1] and cur == 9088
        
        # gen image
        M.dlatents = tmp_s
        M.img_index=0
        M.num_images=j-i
        M.alpha=[0]
        M.step=1
        lindex,bname=0,0
     
        M.manipulate_layers=[lindex]
        codes,out=M.EditOneC(bname)  # NB(demi): we set alpha=0, so no actual edit (?), just a hack
        
        assert len(out) == j-i
        for t in range(len(out)):
            img = out[t, 0]
            img = Image.fromarray(img)
            img.save(f"{args.input}/predict_{i+t:06d}.jpg")


    # generate video
    command = f"ffmpeg -y -r {args.fps} -i {args.input}/predict_%06d.jpg {args.input}/predict.mp4"
    os.system(command)

    command = f"ffmpeg -y -i {args.input}/predict.mp4 -i {args.input}/audio.wav -map 0 -map 1:a -c:v copy -shortest {args.input}/predict_with_audio.mp4"
    os.system(command)
