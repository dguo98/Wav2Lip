import torch
import os, sys
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from IPython import embed


X = np.load('./tmp/wav2lip.npy')
y = np.load('./tmp/frame.npy').reshape((2398, -1))

# add previous
assert X.shape == (2398, 512)


#yy = y[1:] - y[:-1]
yy = y[1:]
XX = np.concatenate((X[1:], y[:-1]), axis=1)  # nb(demi): or X[:-1]?

reg = LinearRegression().fit(XX, yy)

predict_y=reg.predict(XX)
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
curr_y = np.mean(y, axis=0)
new_y = np.zeros((new_X.shape[0], 512*18))

for i in range(len(new_X)):
    new_XX = np.concatenate((new_X[i:i+1], curr_y.reshape(1,-1)),axis=1)
    pred_y = reg.predict(new_XX)
    assert pred_y.shape == (1, 512*18)
    curr_y =  pred_y.reshape(-1)
    new_y[i] = curr_y

new_y = new_y.reshape((-1,18,512))

for i, latent in enumerate(new_y):
    torch.save(torch.tensor(latent), f'./tmp/inference/{testdir}/predict_{str(i).zfill(4)}.pt')

