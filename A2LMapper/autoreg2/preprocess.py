import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from glob import glob
from PIL import Image

def preprocess_images(args):
    data_folders = glob(f"{args.train_path}/train/*")
    data_folders.extend(glob(f"{args.train_path}/val/*"))
    data_folders.append(f"{args.test_path}")
    

    for i, folder in enumerate(data_folders):
        output_path = f"{folder}/{args.image_type}_images_r{args.image_size}.npy"
        
        if not os.path.exists(output_path):  # preprocess

            if args.image_type == "gt":
                img_paths = sorted(glob(f"{folder}/frames/*aligned.jpg"))
            elif args.image_type == "gan":
                img_paths = sorted(glob(f"{folder}/gan_aligned/*.jpg"))
            else:
                raise NotImplementedError


            folder_images = []
            for img_path in tqdm(img_paths, desc=f"load images for folder {i} out of {len(data_folders)}"):
                img = Image.open(f"{img_path}").resize((args.image_size, args.image_size))
                folder_images.append(np.array(img))
            folder_images = np.stack(folder_images, axis=0)
            assert np.max(folder_images) <= 255 and np.min(folder_images) >= 0
            np.save(output_path, folder_images)


