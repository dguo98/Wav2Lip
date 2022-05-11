import numpy as np
import os
import sys
from tqdm import tqdm
from glob import glob
import torch
#from A2LMapper.autoreg2.stylespace.manipulate import Manipulator
from PIL import Image
from IPython import embed
import pickle

def convert_w2s(M, w_plus):
    s = [] 
    for i in tqdm(range(0, len(w_plus), batch_size), desc=f"converting from w+ to s"):
        j = min(i+batch_size, len(w_plus))
        
        tmp_w_plus = w_plus[i:j]
        tmp_s = M.W2S(tmp_w_plus)
        tmp_s = np.concatenate(tmp_s, axis=1)
        assert tmp_s.shape == (j-i, 9088)
        s.append(tmp_s)

    s = np.concatenate(s, axis=0)
    assert s.shape == (len(w_plus), 9088)

if __name__ == "__main__":
    limit = 500
    output_dir = f"logs/stylespace_attribute"
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(f"{output_dir}/cached.pkl"):
        with open(f"{output_dir}/cached.pkl", "rb") as f:
            res = pickle.load(f)
        population_ss_vecs = res["population_ss_vecs"]
        sort_channels = res["sort_channels"]
        e_theta = res["e_theta"].reshape(-1)
        root_path = "data/timit/videos/train"
    else:
        # first aggregate A00026 results
        files = sorted(glob(f"logs/A00026/inference/predict_*.pt"))
        wplus_vecs = []
        for f in files:
            vec = torch.load(f)
            wplus_vecs.append(vec.detach().cpu().numpy().reshape(-1))
        wplus_vecs = np.stack(wplus_vecs, axis=0)
        np.save(f"{output_dir}/frame.npy", wplus_vecs)
        os.system(f"python A2LMapper/autoreg2/stylespace/convert_w2s.py {output_dir}")
        attribute_ss_vecs = np.load(f"{output_dir}/frame_stylespace.npy")
        
        """
        # convert to stylespace vecs
        dataset_name = "ffhq"
        M=Manipulator(dataset_name=dataset_name)
        np.set_printoptions(suppress=True)
        
        # talking stylespace vectors
        attribute_ss_vecs = convert_w2s(wplus_vecs).reshape(-1, 9088)
        """

        # get population stylespace vectors
        root_path = "data/timit/videos/train"
        subfolders = glob(f"{root_path}/*")
        population_ss_vecs = []
        for i, subfolder in enumerate(subfolders):
            assert os.path.isdir(subfolder)
            vec = np.load(f"{subfolder}/frame_stylespace.npy")
            population_ss_vecs.append(vec)
            
            # sanity check, due to assumption made later
            if i == 0:
                assert "s0" in subfolder and len(vec) >= limit
        population_ss_vecs = np.concatenate(population_ss_vecs, axis=0).reshape(-1, 9088)
        
        # get population stats
        p_mean = np.mean(population_ss_vecs, axis=0).reshape(1,9088)
        p_std = np.std(population_ss_vecs, axis=0).reshape(1,9088)

        # normalized attribute vectors
        e_delta = (attribute_ss_vecs - p_mean) / p_std
        assert e_delta.shape == attribute_ss_vecs.shape
        e_mean = np.mean(e_delta, axis=0).reshape(1,9088)
        e_std = np.std(e_delta, axis=0).reshape(1,9088)

        e_theta = np.abs(e_mean) / e_std  # high theta_u implies that channel u is likely only relevant to talking expression
        
        sort_channels = np.argsort(-e_theta.reshape(-1))  # large, first
        print("e_theta top 10:", e_theta.reshape(-1)[sort_channels[:10]])
        
        res = {"e_theta": e_theta, "sort_channels": sort_channels, "population_ss_vecs": population_ss_vecs[:limit]}
        with open(f"{output_dir}/cached.pkl", "wb") as f:
            pickle.dump(res, f)

    # visualize results 
    ss_vecs = population_ss_vecs[:limit].reshape(-1, 9088)
   
    sort_channels = np.flip(sort_channels) 
    print("sort_channels values top10:", e_theta[sort_channels[:10]])

    for topk in []:
        topk_dir = f"{output_dir}/invtop{topk}"        
        os.makedirs(topk_dir, exist_ok=True)
        os.system(f"cp {root_path}/s0/audio.wav {topk_dir}/")
        default_vec = ss_vecs[0:1]
        final_ss_vecs = np.repeat(default_vec, len(ss_vecs), axis=0)
        final_ss_vecs[:, sort_channels[:topk]] = ss_vecs[:, sort_channels[:topk]]
         
        np.save(f"{topk_dir}/predict_stylespace.npy", final_ss_vecs)

        cmd = f"python A2LMapper/autoreg2/stylespace/convert_s2img.py --input {topk_dir} --fps 25"
        print(cmd)
        os.system(cmd)
        os.system(f"rm {topk_dir}/*.jpg")

    # (1000,3000)
