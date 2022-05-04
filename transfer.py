import os
import torch
import sys
import numpy as np
import pickle
from tqdm import tqdm
from glob import glob

if __name__ == "__main__":
    src_neutral = np.load("data/timit/videos/neutral.npy")
    src_latents = np.load("data/timit/videos/test/s3/frame.npy")
    limit = 500
    
    for tgt_id in ["p2", "p4"]:
        # transfer gt
        output_dir = f"data/ron/processed/{tgt_id}"
        os.makedirs(output_dir, exist_ok=True)
        os.system(f"cp data/timit/videos/test/s3/audio.wav {output_dir}/")
        tgt_neutral = np.load(f"data/ron/processed/latents_{tgt_id}.npy")

        src_neutral = src_neutral.reshape(-1, 512*18)
        src_latents = src_latents.reshape(-1, 512*18)[:limit]
        tgt_neutral = tgt_neutral.reshape(-1, 512*18)

        tgt_latents = src_latents - src_neutral + tgt_neutral
        
        for i, latent in tqdm(enumerate(tgt_latents), desc=f"process {tgt_id}"):
            torch.save(torch.tensor(latent), f"{output_dir}/predict_{i:06d}.pt")

        # transfer A00026 prediction
        output_dir = f"data/ron/processed/A00026_{tgt_id}"
        os.makedirs(output_dir, exist_ok=True)
        os.system(f"cp data/timit/videos/test/s3/audio.wav {output_dir}/")
            
        tgt_paths = sorted(glob(f"logs/A00026/inference/predict*.pt"))
        
        for i in range(min(limit, len(tgt_paths))):
            tgt_path = tgt_paths[i]
            src_latent = torch.load(tgt_path).detach().cpu().numpy().reshape(1, 512*18) 
            tgt_latent = src_latent - src_neutral + tgt_neutral
            torch.save(torch.tensor(tgt_latent), f"{output_dir}/predict_{i:06d}.pt")
        
    


