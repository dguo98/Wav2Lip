import torch
import os, sys
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from IPython import embed


X = np.load('./tmp/wav2lip.npy')
y = np.load('./tmp/frame.npy').reshape((2398, -1))

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

