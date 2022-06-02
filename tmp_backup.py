import torch
import os, sys
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression


X = np.load('data/p_words/p_train/wav2lip.npy')
y = np.load('data/p_words/p_train/frame.npy').reshape((-1, 512*18))

reg = LinearRegression().fit(X[:-1], y[1:] - y[:-1])

with open('./tmp/reg_diff.pkl', 'wb') as fw:
    pickle.dump(reg, fw, -1)

with open('./tmp/reg_diff.pkl', 'rb') as fr:
    reg = pickle.load(fr)

testdir = "p_test"
os.makedirs(f"./tmp/inference/{testdir}", exist_ok=True)
os.system(f"cp data/p_words/{testdir}/audio.wav ./tmp/inference/{testdir}/")


new_X = np.load(f'data/p_words/{testdir}/wav2lip.npy')
pred_y = reg.predict(new_X)
new_y = np.zeros_like(pred_y)
y_mean = np.mean(y, axis=0)
curr_y = y_mean
for i in range(len(new_y)):
    new_y[i] = curr_y
    curr_y = y_mean + pred_y[i]

new_y = new_y.reshape((-1,18,512))

for i, latent in enumerate(new_y):
    torch.save(torch.tensor(latent), f'./tmp/inference/{testdir}/predict_{str(i).zfill(6)}.pt')

