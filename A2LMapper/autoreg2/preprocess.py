import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from glob import glob
from PIL import Image

def preprocess_images(args):
    data_folders = glob(f"{args.train_path}/train/*")
    data_folders.extend(glob(f"{args.train_path}/val/*")
    data_folders.append(f"{args.test_path}")
    

    for data_folder in data_folders:
        output_path = f"{data_folder}/{args.image_type}_images.npy"









                        if self.image_type == "gt":

                        img_paths = sorted(glob(f"{folder}/frames/*aligned.jpg"))
                    elif self.image_type == "gan":
                        img_paths = sorted(glob(f"{folder}/gan_aligned/*.jpg"))
                    else:
                        raise NotImplementedError

                    assert len(self.audio_vecs_list[-1]) == len(img_paths)

                    folder_images = []
                    for img_path in tqdm(img_paths, desc="load images"):
                        img = Image.open(f"{img_path}").resize((self.image_size, self.image_size))
                        img = np.array(img)
                        if args.image_mouth == 1:
                            img = img[x1:x2, y1:y2]
                        folder_images.append(img)
                    folder_images = np.stack(folder_images, axis=0)
                    print("folder_images.shape=", folder_images.shape)
                    assert folder_images.shape[0] == len(self.audio_vecs_list[-1])
                    assert np.max(folder_images) <= 255 and np.min(folder_images) >= 0

        

