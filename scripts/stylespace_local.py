import numpy as np
import os
import sys
from tqdm import tqdm
from glob import glob
import torch
from PIL import Image
from IPython import embed
import pickle

def convert_pair2ind(pr):
    l,c=pr
    s_shapes = [512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,256,256,256,128,128,128,64,64,64,32,32]
    
    s_sums = np.cumsum(np.array(s_shapes))

    index = 0
    if l > 0:
        index += s_sums[l-1]
    return index +c


if __name__ == "__main__":
    limit = 500
    root_output_dir = f"logs/stylespace_local"
    os.makedirs(root_output_dir, exist_ok=True)

    mouth_pairs = [(12, 253), (11, 447), (14, 110), (14, 85), (14, 12), (12, 410), (12, 381), (12, 262), (12, 239), (12, 186), (11, 174), (15, 45), (11, 86), (11, 6), (8, 17), (6, 501), (6, 491), (6, 378), (6, 202), (6, 113), (14, 490), (6, 21), (15, 251), (15, 235), (17, 241), (18, 35), (17, 247), (17, 112), (9, 321), (9, 91), (9, 117), (9, 232), (9, 294), (15, 121), (9, 351), (9, 77), (11, 73), (11, 116), (17, 186), (11, 204), (11, 279), (9, 452), (9, 48), (11, 314), (9, 26), (8, 456), (8, 389), (8, 191), (8, 122), (8, 118), (8, 85), (18, 0), (18, 52), (18, 57), (6, 259), (6, 214), (20, 25), (20, 73), (11, 313), (11, 409), (11, 374), (15, 249), (15, 102), (15, 75), (15, 68), (15, 178), (15, 37), (15, 228), (14, 339), (14, 286), (14, 263), (14, 230), (14, 223), (14, 222), (14, 213), (14, 107), (14, 66), (15, 104), (12, 482), (17, 29), (17, 37), (17, 66), (20, 103), (17, 126), (12, 183), (12, 177), (12, 80), (12, 59), (12, 56), (12, 5), (11, 481), (17, 165), (23, 45)]

    mouth_channels = [convert_pair2ind(p) for p in mouth_pairs]
    print("mouth channels=", str(mouth_channels))
    
    mouth_mask = np.zeros(9088, dtype=np.float32)
    mouth_mask[mouth_channels] = 1
    print("sum mouth mask=",np.sum(mouth_mask))
    np.save(f"{root_output_dir}/mouth_mask.npy", mouth_mask)

    #sys.exit(0)
    
    data_path = f"data/timit/videos/test/s3"
    ss_vecs = np.load(f"{data_path}/frame_stylespace.npy")
    
    default_vec = ss_vecs[0:1]
    final_ss_vecs = np.repeat(default_vec, len(ss_vecs),axis=0)
    final_ss_vecs[:, mouth_channels] = ss_vecs[:, mouth_channels]
    mse_loss = np.mean(np.sum((final_ss_vecs-ss_vecs)**2,axis=1),axis=0)
    print("mse_loss=", mse_loss)
    
    tmp_vecs = np.load(f"logs/A00052/inference/predict_stylespace.npy")
    mse_loss = np.mean(np.sum((tmp_vecs-ss_vecs)**2,axis=1),axis=0)
    print("A00052 mse_loss=", mse_loss)

    # HACK(demi)
    sys.exit(0)

    # all
    output_dir = f"{root_output_dir}/all"
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/predict_stylespace.npy", ss_vecs[:limit])
    os.system(f"cp {data_path}/audio.wav {output_dir}/")
    cmd = f"python A2LMapper/autoreg2/stylespace/convert_s2img.py --input {output_dir} --fps 25"
    os.system(cmd)

    # mouth 93 
    output_dir = f"{root_output_dir}/mouth93"
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/predict_stylespace.npy", final_ss_vecs[:limit])
    os.system(f"cp {data_path}/audio.wav {output_dir}/")
    cmd = f"python A2LMapper/autoreg2/stylespace/convert_s2img.py --input {output_dir} --fps 25"
    os.system(cmd)
    
    # compare
    cmd = f"ffmpeg -i {root_output_dir}/all/predict_with_audio.mp4 -i {root_output_dir}/mouth93/predict_with_audio.mp4 -filter_complex hstack=2 {root_output_dir}/compare.mp4"
    os.system(cmd)
