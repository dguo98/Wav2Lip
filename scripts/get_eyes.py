import os.path
import numpy as np
import PIL.Image
import scipy
from tqdm import tqdm
import pickle
import cv2


root_dir = "data/timit_v3"
save_dir = "/u/nlp/data/timit_videos_v2"


debug=False

eye_close_left_avg=None
eye_close_right_avg=None
for si in tqdm(range(32), desc="extracting eyes"):
    lm_path = f"{root_dir}/landmarks/s{si}.pkl"
    if not os.path.exists(lm_path):
        print("lmk not exist")
        continue
    with open(lm_path, "rb") as f:
        lm = np.array(pickle.load(f))

    eyes = lm[:, 36:48]

    eye_close_left = (lm[:, 37] + lm[:, 38] - lm[:, 41] - lm[:, 40]) / 2 
    eye_close_right = (lm[:, 43] + lm[:, 44] - lm[:, 47] - lm[:, 46]) / 2
    eye_close_left = eye_close_left[:, 1]
    eye_close_right = eye_close_right[:, 1]
    if eye_close_left_avg is None:
        eye_close_left_avg = np.mean(eye_close_left, axis=0)
        eye_close_right_avg = np.mean(eye_close_right, axis=0)
        assert eye_close_left_avg.shape == ()

    eye_close_left = eye_close_left - eye_close_left_avg
    eye_close_right = eye_close_right - eye_close_right_avg
    
    n = lm.shape[0]
    eye_info = np.concatenate([eye_close_left.reshape(-1,1), eye_close_right.reshape(-1,1), eyes.reshape(n, 12*2)], axis=1)
    assert eye_info.shape == (lm.shape[0], 26)

    split = "train"
    if si in [3]:
        split="test"
    if si in [30, 4]:
        split="val"
    np.save(f"{save_dir}/videos/{split}/s{si}/eyes.npy", eye_info)
    
