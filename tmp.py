import torch
from tqdm import tqdm
import os, sys
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from IPython import embed


prefix = "p_train"
n_frames=350
data_dir ="data/p_words/{prefix}"
#data_dir = "./tmp"
X = np.load(f'{data_dir}/wav2lip.npy')
y = np.load(f'{data_dir}/frame.npy')
y = y.reshape((y.shape[0], -1))
#neutral = np.load(f'{data_dir}/frame_latents/neutral.npy').reshape(-1)
neutral = y[0].reshape(-1)
#neutral=np.mean(y,axis=0)

# analyze PCA reduction
n = 10
vecs = y - neutral.reshape((1, -1))

if os.path.exists(f"./tmp/{prefix}_pca_inverse_n{n}/pca.pkl"):
    print("load")
    with open(f"./tmp/{prefix}_pca_inverse_n{n}/pca.pkl", "rb") as f:
        pca = pickle.load(f)
else:
    print("train a new pca")
    pca = PCA(n_components=n)
    pca.fit(vecs)

vecs = vecs[:n_frames]
new_vecs = pca.transform(vecs)

os.makedirs(f"./tmp/{prefix}_inverse", exist_ok=True)
os.makedirs(f"./tmp/{prefix}_pca_inverse_n{n}", exist_ok=True)
with open(f"./tmp/{prefix}_pca_inverse_n{n}/pca.pkl", "wb") as f:
    pickle.dump(pca, f)

print("explained variance of top n=", n, " =", pca.explained_variance_ratio_)
print("sum variance =", np.sum(pca.explained_variance_ratio_))

inv_vecs = pca.inverse_transform(new_vecs)
print("check new vecs and inv_vecs abs diff=", np.sum(np.abs(vecs-inv_vecs)))

os.system(f"cp {data_dir}/audio.wav ./tmp/{prefix}_inverse/")
os.system(f"cp {data_dir}/audio.wav ./tmp/{prefix}_pca_inverse_n{n}/")

for i,vec in enumerate(vecs):
    torch.save(torch.tensor(vec+neutral), f"./tmp/{prefix}_inverse/{i:06d}.pt")
for i,vec in enumerate(inv_vecs):
    torch.save(torch.tensor(vec+neutral), f"./tmp/{prefix}_pca_inverse_n{n}/{i:06d}.pt")
 
# analyze each dimension    
new_neutral = pca.transform(neutral.reshape((1,-1))*0)
for i in tqdm(range(n), "varying dims"):  # fix {i}-th dim
    ith_new_vecs = np.repeat(new_neutral, len(new_vecs), axis=0)
    assert ith_new_vecs.shape == new_vecs.shape
    ith_new_vecs[:, i] = new_vecs[:, i]
    inv_vecs = pca.inverse_transform(ith_new_vecs)
    os.makedirs(f"./tmp/{prefix}_pca_inverse_n{n}/{i}th", exist_ok=True)
    os.system(f"cp {data_dir}/audio.wav ./tmp/{prefix}_pca_inverse_n{n}/{i}th")
    for j,vec in enumerate(inv_vecs):
        torch.save(torch.tensor(vec+neutral), f"./tmp/{prefix}_pca_inverse_n{n}/{i}th/{j:06d}.pt")
"""
# pose only
name = "pose"
ith_new_vecs = np.repeat(new_neutral, len(new_vecs), axis=0)
assert ith_new_vecs.shape == new_vecs.shape
ith_new_vecs[:, 0] = new_vecs[:, 0]
ith_new_vecs[:, 3] = new_vecs[:, 3]
ith_new_vecs[:, 2] = new_vecs[:, 2]
inv_vecs = pca.inverse_transform(ith_new_vecs)
os.makedirs(f"./tmp/{prefix}_pca_inverse_n{n}/{name}", exist_ok=True)
os.system(f"cp {data_dir}/audio.wav ./tmp/{prefix}_pca_inverse_n{n}/{name}")
for j,vec in enumerate(inv_vecs):
    torch.save(torch.tensor(vec+neutral), f"./tmp/{prefix}_pca_inverse_n{n}/{name}/{j:06d}.pt")

name = "talking"
ith_new_vecs = np.repeat(new_neutral, len(new_vecs), axis=0)
assert ith_new_vecs.shape == new_vecs.shape
for i in [1,2,3,4,5,6,7,8,9]:
    ith_new_vecs[:, i] = new_vecs[:, i]
inv_vecs = pca.inverse_transform(ith_new_vecs)
os.makedirs(f"./tmp/{prefix}_pca_inverse_n{n}/{name}", exist_ok=True)
os.system(f"cp {data_dir}/audio.wav ./tmp/{prefix}_pca_inverse_n{n}/{name}")
for j,vec in enumerate(inv_vecs):
    torch.save(torch.tensor(vec+neutral), f"./tmp/{prefix}_pca_inverse_n{n}/{name}/{j:06d}.pt")
"""
"""

  



use_PCA = True
if use_PCA:
    yy = y[1:] - y[:-1]
    pca = PCA(n_components=100)
    yy = pca.fit_transform(yy)
    print("yy.shape=",yy.shape, " y.shape=" ,y.shape)
    print(pca.explained_variance_ratio_)
    print(np.sum(pca.explained_variance_ratio_))
    embed()
else:
    yy = y[1:] - y[:-1]

reg = LinearRegression().fit(X[:-1], yy)

predict_y=reg.predict(X[:-1])
mse_loss = ((predict_y-yy)**2).sum(axis=1).mean(axis=0)
print("mse loss=", mse_loss)

with open('./tmp/reg_diff.pkl', 'wb') as fw:
    pickle.dump(reg, fw, -1)

#with open('./tmp/reg_diff.pkl', 'rb') as fr:
#    reg = pickle.load(fr)

testdir = "eval-1"
os.makedirs(f"./tmp/inference/{testdir}", exist_ok=True)
os.system(f"cp eval_mlpmap/{testdir}/audio.wav ./tmp/inference/{testdir}/")


new_X = np.load(f'eval_mlpmap/{testdir}/test.npy')
pred_y = reg.predict(new_X)
if use_PCA:
    pred_y = pca.inverse_transform(pred_y)

new_y = np.zeros_like(pred_y)

# hack
#curr_y = np.mean(y, axis=0)
curr_y = y[0]

for i in range(len(new_y)):
    new_y[i] = curr_y
    curr_y += pred_y[i]

new_y = new_y.reshape((-1,18,512))

for i, latent in enumerate(new_y):
    torch.save(torch.tensor(latent), f'./tmp/inference/{testdir}/predict_{str(i).zfill(4)}.pt')

"""
