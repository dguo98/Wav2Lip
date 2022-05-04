import torch
from tqdm import tqdm
import os, sys
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from IPython import embed
from glob import glob

n=10
data_dir ="data/timit/videos"

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
frames = np.concatenate(frames, axis=0)
"""

neutral = np.load(f"{data_dir}/neutral.npy").reshape(1, -1)
y = frames
vecs = y - neutral.reshape((1, -1))

# run PCA
prefix="timit"
output_dir = f"tmp/{prefix}_n{n}"
os.makedirs(output_dir, exist_ok=True)
if os.path.exists(f"{output_dir}/pca.pkl"):
    with open(f"{output_dir}/pca.pkl", "rb") as f:
        pca = pickle.load(f)
else:
    pca = PCA(n_components=n)
    pca.fit(vecs)
    with open(f"{output_dir}/pca.pkl", "wb") as f:
        pickle.dump(pca,f)
print("fit pca done")

# get test
vecs = np.load(f"{data_dir}/test/s3/frame.npy").reshape(-1, 512*18) - neutral
new_vecs = pca.transform(vecs)
print("explained variance of top n=", n, " =", pca.explained_variance_ratio_)
print("sum variance =", np.sum(pca.explained_variance_ratio_))

inv_vecs = pca.inverse_transform(new_vecs)
print("check new vecs and inv_vecs abs diff=", np.sum(np.abs(vecs-inv_vecs)**2)/len(vecs))


# now create videos 
os.makedirs(f"./tmp/timit_inverse", exist_ok=True)
os.makedirs(f"./tmp/{prefix}_pca_inverse_n{n}", exist_ok=True)
os.system(f"cp {data_dir}/test/s3/audio.wav ./tmp/timit_inverse/")
os.system(f"cp {data_dir}/test/s3/audio.wav ./tmp/{prefix}_pca_inverse_n{n}/")
limit = 500
vecs = vecs[:limit]
new_vecs = new_vecs[:limit]
inv_vecs = inv_vecs[:limit]

for i,vec in enumerate(vecs):
    torch.save(torch.tensor(vec+neutral), f"./tmp/timit_inverse/{i:06d}.pt")
for i,vec in enumerate(inv_vecs):
    torch.save(torch.tensor(vec+neutral), f"./tmp/{prefix}_pca_inverse_n{n}/{i:06d}.pt")
 
new_neutral = pca.transform(neutral.reshape((1,-1))*0)
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
"""

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

