import numpy as np
import os
import sys
import torch
from tqdm import tqdm
from glob import glob
import torch
from manipulate import Manipulator
from PIL import Image
from IPython import embed

if __name__ == "__main__":
    dataset_name = "ffhq"
    M=Manipulator(dataset_name=dataset_name)
    np.set_printoptions(suppress=True)

    base_dir = "/nlp/u/demiguo/research/Wav2Lip/data/timit/videos"
    folders = [f"{base_dir}/train", f"{base_dir}/test", f"{base_dir}/val"]
    batch_size = 32

    for folder in folders:
        subfolders = sorted(glob(f"{folder}/*"))
        for subfolder in subfolders:
            if not os.path.isdir(subfolder):
                continue

            w_plus = np.load(f"{subfolder}/frame.npy").reshape(-1, 18, 512)

            # batched
            s = [] 
            for i in tqdm(range(0, len(w_plus), batch_size), desc=f"converting subfolder {os.path.basename(subfolder)}"):
                j = min(i+batch_size, len(w_plus))
                
                tmp_w_plus = w_plus[i:j]
                tmp_s = M.W2S(tmp_w_plus)
                tmp_s = np.concatenate(tmp_s, axis=1)
                assert tmp_s.shape == (j-i, 9088)
                s.append(tmp_s)

            s = np.concatenate(s, axis=0)
            assert s.shape == (len(w_plus), 9088)
            np.save(f"{subfolder}/frame_stylespace.npy", s)

