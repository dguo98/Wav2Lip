print("pp start impot")
import os
import sys
import numpy as np
import torch
import cv2
from tqdm import tqdm
from glob import glob
from PIL import Image

print("pp importsecond")
from criterion import images_to_lmks
from visualize import viz_lmk
print("pp finish import")

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


def preprocess_lmks(args, hgnet, xycoords):
    # assume image already processed
    data_folders = glob(f"{args.train_path}/train/*")
    data_folders.extend(glob(f"{args.train_path}/val/*"))
    data_folders.append(f"{args.test_path}")

    for i, folder in enumerate(data_folders):
        output_path = f"{folder}/{args.image_type}_lmks_r{args.image_size}.npy"

        if os.path.exists(output_path):
            continue

        img_path = f"{folder}/{args.image_type}_images_r{args.image_size}.npy"
        images = np.load(img_path)
        bsz = 32

        all_lmks = []
        for l in tqdm(range(0, len(images), bsz), desc=f"extracting landmarks for folder {i} out of {len(data_folders)}"):
            r = min(l+bsz, len(images))
            
            batch_images = images[l:r].astype(np.float32)/255.
            batch_images = torch.from_numpy(batch_images).cuda()
            
            batch_lmks = images_to_lmks(args, hgnet, xycoords, batch_images)
            assert batch_lmks.shape == (r-l, 131, 2)

            viz_image = viz_lmk(args, batch_images[0], batch_lmks[0])
            cv2.imwrite("preprocess_lmk.jpg", viz_image)

            all_lmks.append(batch_lmks.detach().cpu().numpy())
        all_lmks = np.concatenate(all_lmks, axis=0)

        assert all_lmks.shape[0] == images.shape[0]
        np.save(output_path, all_lmks)


 
