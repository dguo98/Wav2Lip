import torch
import os, sys
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression


X = np.load('./tmp/wav2lip.npy')
y = np.load('./tmp/frame.npy').reshape((2398, -1))

reg = LinearRegression().fit(X[:-1], y[1:] - y[:-1])

with open('./tmp/reg_diff.pkl', 'wb') as fw:
    pickle.dump(reg, fw, -1)

with open('./tmp/reg_diff.pkl', 'rb') as fr:
    reg = pickle.load(fr)

testdir = "eval-4-test"
os.makedirs(f"./tmp/inference/{testdir}", exist_ok=True)
os.system(f"cp eval_mlpmap/{testdir}/audio.wav ./tmp/inference/{testdir}/")


new_X = np.load(f'eval_mlpmap/{testdir}/test.npy')
pred_y = reg.predict(new_X)
new_y = np.zeros_like(pred_y)
y_mean = np.mean(y, axis=0)
curr_y = y_mean
for i in range(len(new_y)):
    new_y[i] = curr_y
    curr_y = y_mean + pred_y[i]

new_y = new_y.reshape((-1,18,512))

for i, latent in enumerate(new_y):
    torch.save(torch.tensor(latent), f'./tmp/inference/{testdir}/predict_{str(i).zfill(4)}.pt')

