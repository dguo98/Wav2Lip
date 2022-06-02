import os.path
import numpy as np
import PIL.Image
import scipy
from tqdm import tqdm
import pickle
import cv2


root_dir = "data/timit_v3"

debug=False

for si in tqdm(range(32), desc="aligning"):
    lm_path = f"{root_dir}/landmarks/s{si}.pkl"
    if not os.path.exists(lm_path):
        print("lmk not exist")
        continue
    dump_root = f"{root_dir}/noaligned/s{si}"
    if os.path.exists(dump_root):
        print("dump exist")
        continue
    os.makedirs(dump_root)
    with open(lm_path, "rb") as f:
        lm = np.array(pickle.load(f))


    for i in tqdm(range(len(lm))):
        # read image
        image_path = f"{root_dir}/frames/s{si}/{i+1:04d}.png"
        

        # read image
        img = PIL.Image.open(image_path)
        cx=0
        cy=630
        s=660
        img = np.asarray(img)

        img = img[cx:cx+s, cy:cy+s]
        img = PIL.Image.fromarray(img)

        output_size = 256
        transform_size = 256
        enable_padding = True

        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        img.save(f"{root_dir}/noaligned/s{si}/{i:04d}.jpg")
          

