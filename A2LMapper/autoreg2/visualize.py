import numpy as np
import os
import sys
import torch
from PIL import Image
import cv2
import json

sys.path.append(".")
from HGNet.common.global_constants import IMAGE_SIZE_256

def convert_image(img):
    img = np.clip(img, 0, 1)
    img = (img*255).astype(np.uint8)
    return Image.fromarray(img)

def viz_lmk(args, pimg, plmk, gtlmk=None):
    pimg = pimg[:, :, [2,1,0]]*255
    pimg = pimg.detach().cpu().numpy().astype(np.uint8)
    x1,y1,x2,y2 = args.face_box
    
    plmk = plmk.detach().cpu().numpy()
    plmk = (x1, y1) + (x2-x1, y2-y1) * plmk / IMAGE_SIZE_256
    for i in range(plmk.shape[0]):
        x,y = plmk[i,0], plmk[i, 1]
        cv2.circle(pimg, (int(x), int(y)), 1, (0,255,0),-1)  # green, predict
    
    if gtlmk is not None:
        gtlmk = gtlmk.detach().cpu().numpy()
        gtlmk = (x1, y1) + (x2-x1, y2-y1) * gtlmk / IMAGE_SIZE_256
        for i in range(gtlmk.shape[0]):
            x,y = gtlmk[i,0], gtlmk[i, 1]
            cv2.circle(pimg, (int(x), int(y)), 1, (0,0, 255),-1)  # red, original

    return pimg

def log_metrics(args, epoch, n_iter, split, viz_info, log_path):
    log_dir = {"epoch": epoch, "n_iter": n_iter, "type": split}
    for key in viz_info:
        if "loss" in key:
            log_dir[key] = viz_info[key]
    with open(log_path, "a") as f:
        json.dump(log_dir, f)
        f.write("\n")

def visualize(args, viz_info, viz_path_prefix):
    for key in viz_info:
        if "_img" in key:
            img = convert_image(viz_info[key])
            img.save(f"{viz_path_prefix}_{key}.jpg")
        elif "_cv2" in key:
            cv2.imwrite(f"{viz_path_prefix}_{key.replace('_cv2','')}.jpg", viz_info[key])
    return
