import torch
import os, sys
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from IPython import embed


import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Model 

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", from_tf=True)


def get_wav2vec(audio_file, data_len):
    new_audio_file = audio_file.replace("audio.wav", "audio_new.wav")
    feats_file = audio_file.replace("audio.wav", "wav2vec.npy")
    if os.path.exists(feats_file):
        feats = np.load(feats_file)
        return feats
    os.system(f"ffmpeg -y -i {audio_file} -ar 16000 -ac 1 {new_audio_file}")

    audio_input, sample_rate = sf.read(new_audio_file)
    print("read from :", new_audio_file, " shape=", audio_input.shape)
    input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
    with torch.no_grad():
        outputs = model(input_values)  
    feats = outputs.extract_features.permute((0,2,1))  # to [1, 512, length]
    feats = torch.nn.functional.interpolate(feats, size=data_len, mode="area")
    feats = feats.permute((0,2,1)).reshape(-1, 512)  # [1, length, 512]
    feats = feats.detach().cpu().numpy()
    np.save(feats_file, feats)
    return feats


X = get_wav2vec("data_mlpmap/synthesia-obama-trim/audio.wav", 2398)
#X = np.load('./tmp/wav2lip.npy')

y = np.load('./tmp/frame.npy').reshape((2398, -1))
yy = y[1:] - y[:-1]

reg = LinearRegression().fit(X[:-1], yy)

pred_y = reg.predict(X[:-1])
mse_loss = ((yy-pred_y)**2).sum(axis=1).mean(axis=0)
print("mse loss=", mse_loss)

with open('./tmp/reg_diff.pkl', 'wb') as fw:
    pickle.dump(reg, fw, -1)

#with open('./tmp/reg_diff.pkl', 'rb') as fr:
#    reg = pickle.load(fr)

testdir = "eval-3-train"
os.makedirs(f"./tmp/inference/{testdir}", exist_ok=True)
os.system(f"cp eval_mlpmap/{testdir}/audio.wav ./tmp/inference/{testdir}/")

old_X = np.load(f'eval_mlpmap/{testdir}/test.npy')
assert old_X.shape[1] == 512
new_X = get_wav2vec(f"eval_mlpmap/{testdir}/audio.wav", old_X.shape[0])
print("new_X.shape=", new_X.shape)

pred_y = reg.predict(new_X)
new_y = np.zeros_like(pred_y)
curr_y = np.mean(y, axis=0)
for i in range(len(new_y)):
    new_y[i] = curr_y
    curr_y += pred_y[i]

new_y = new_y.reshape((-1,18,512))

for i, latent in enumerate(new_y):
    torch.save(torch.tensor(latent), f'./tmp/inference/{testdir}/predict_{str(i).zfill(4)}.pt')

