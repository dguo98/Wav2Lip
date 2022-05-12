import numpy as np
import os
import sys
import torch
from PIL import Image
import cv2

def convert_image(img):
    img = np.clip(img, 0, 1)
    img = (img*255).astype(np.uint8)
    return Image.fromarray(img)

def visualize(args, viz_info, viz_path_prefix):
    for key in viz_info:
        if "_img" in key:
            img = convert_image(viz_info[key])
            img.save(f"{viz_path_prefix}_{key}.jpg")
    return
