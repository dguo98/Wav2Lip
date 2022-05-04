import torch
from tqdm import tqdm
import os, sys
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from IPython import embed
from glob import glob
from scipy.linalg import orth

n=10
data_dir ="data/timit/videos"
prefix="timithack"

"""
frame_paths = glob(f"{data_dir}/train/**/frame.npy")
frames = []
for frame_path in frame_paths:
    frames.append(np.load(frame_path).reshape(-1, 512*18))
frames = np.concatenate(frames, axis=0)
"""
frame_paths = sorted(glob(f"logs/A00026/inference/predict*.pt"))
frames=[]
for frame_path in frame_paths:
    frames.append(torch.load(frame_path).detach().cpu().numpy().reshape(-1, 512*18))
frames = np.concatenate(frames, axis=0)  # [data, dim]
neutral = np.load(f"{data_dir}/neutral.npy").reshape(1, -1)
frames = frames - neutral

basis = orth(frames.T)
assert basis.shape[0] == 512*18
basis = basis.T
print("rank K=", basis.shape[0])

# get test
vecs = np.load(f"{data_dir}/test/s3/frame.npy").reshape(-1, 512*18) - neutral
inv_vecs=(vecs@basis.T)@basis

loss=np.sum((vecs-inv_vecs)**2)/len(vecs)
print("loss=",loss)

# hack debug(demi)
inv_vecs=frames

# basis [K, dim]
# vecs [N, dim]

# new_vecs[i, j] = \sum_z dot(vecs[i], basis[z]) x basis[z,j]
# dots[i, j] = \sum z vecs[i][z] * basis[j][z]
# dots[i,j]=vecs*basis.T
#new_vecs=(vecs@basis.T)@basis


# now create videos 
os.makedirs(f"./tmp/{prefix}_pca_inverse_n{n}", exist_ok=True)
os.system(f"cp {data_dir}/test/s3/audio.wav ./tmp/{prefix}_pca_inverse_n{n}/")
limit = 500
vecs = vecs[:limit]
#new_vecs = new_vecs[:limit]
inv_vecs = inv_vecs[:limit]

for i,vec in enumerate(inv_vecs):
    torch.save(torch.tensor(vec+neutral), f"./tmp/{prefix}_pca_inverse_n{n}/{i:06d}.pt")
 
#new_neutral = pca.transform(neutral.reshape((1,-1))*0)
"""
# analyze each dimension    

for i in tqdm(range(n), "varying dims"):  # fix {i}-th dim
    ith_new_vecs = np.repeat(new_neutral, len(new_vecs), axis=0)
    assert ith_new_vecs.shape == new_vecs.shape
    ith_new_vecs[:, i] = new_vecs[:, i]
    inv_vecs = pca.inverse_transform(ith_new_vecs)
    os.makedirs(f"./tmp/timit_pca_inverse_n{n}/{i}th", exist_ok=True)
    os.system(f"cp {data_dir}/test/s3/audio.wav ./tmp/timit_pca_inverse_n{n}/{i}th")
    for j,vec in enumerate(inv_vecs):
        torch.save(torch.tensor(vec+neutral), f"./tmp/timit_pca_inverse_n{n}/{i}th/{j:06d}.pt")

"""
# pose only
"""
name = "pose"
ith_new_vecs = np.repeat(new_neutral, len(new_vecs), axis=0)
assert ith_new_vecs.shape == new_vecs.shape
ith_new_vecs[:, 0] = new_vecs[:, 0]
ith_new_vecs[:, 3] = new_vecs[:, 3]
inv_vecs = pca.inverse_transform(ith_new_vecs)
os.makedirs(f"./tmp/p_test_pca_inverse_n{n}/{name}", exist_ok=True)
os.system(f"cp {data_dir}/audio.wav ./tmp/p_test_pca_inverse_n{n}/{name}")
for j,vec in enumerate(inv_vecs):
    torch.save(torch.tensor(vec+neutral), f"./tmp/p_test_pca_inverse_n{n}/{name}/{j:06d}.pt")

name = "talking"
ith_new_vecs = np.repeat(new_neutral, len(new_vecs), axis=0)
assert ith_new_vecs.shape == new_vecs.shape
for i in [2, 3,4,5,7,8,9]:
    ith_new_vecs[:, i] = new_vecs[:, i]
inv_vecs = pca.inverse_transform(ith_new_vecs)
os.makedirs(f"./tmp/{prefix}_pca_inverse_n{n}/{name}", exist_ok=True)
os.system(f"cp {data_dir}/test/s3/audio.wav ./tmp/{prefix}_pca_inverse_n{n}/{name}")
for j,vec in enumerate(inv_vecs):
    torch.save(torch.tensor(vec+neutral), f"./tmp/{prefix}_pca_inverse_n{n}/{name}/{j:06d}.pt")
"""
