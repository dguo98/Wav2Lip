import os.path
import numpy as np
import PIL.Image
import scipy
from tqdm import tqdm
import pickle
import cv2


root_dir = "/u/nlp/data/timit_v3"
#root_dir="data/timit_v3"

debug=False

for si in tqdm(range(32), desc="aligning"):
    lm_path = f"{root_dir}/landmarks/s{si}.pkl"
    if not os.path.exists(lm_path):
        print("lmk not exist")
        continue
    dump_root = f"{root_dir}/aligned/s{si}"
    if os.path.exists(dump_root):
        print("dump exist")
        continue
    os.makedirs(dump_root)
    with open(lm_path, "rb") as f:
        lm = np.array(pickle.load(f))

    lm_nostrils = lm[:, 31: 36]  # top-down
    lm_eye_left_bottom = lm[:, [36, 39, 40, 41]]
    lm_eye_right_bottom = lm[:, [42, 45, 46, 47]]
    eye_bottom_left = np.mean(lm_eye_left_bottom, axis=1)
    eye_bottom_right = np.mean(lm_eye_right_bottom, axis=1)
    eye_bottom_avg = (eye_bottom_left + eye_bottom_right) * 0.5
    eye_to_eye_bottom = eye_bottom_right - eye_bottom_left
    nostrils_avg = np.mean(lm_nostrils, axis=1)

    eye_avg = eye_bottom_avg + np.array([-0.16190288, -5.43143092])
    eye_to_eye = eye_to_eye_bottom + np.array([-0.53272205, -0.0035519])
    mouth_avg = nostrils_avg + np.array([2.66506988, 44.71003433])
    eye_to_mouth = mouth_avg - eye_avg

    eye_close = ((lm[:, 37] + lm[:, 38] - lm[:, 41] - lm[:, 40]) / 2 + \
        (lm[:, 43] + lm[:, 44] - lm[:, 47] - lm[:, 46]) / 2) / 2
    eye_close = eye_close[:, 1]  # only look at 1-axis for now
    eye_close_avg = np.mean(eye_close, axis=0)
	

    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    eye_avg[:, 0] = np.convolve(eye_avg[:, 0], kernel, mode='same')
    eye_avg[:, 1] = np.convolve(eye_avg[:, 1], kernel, mode='same')
    eye_to_eye[:, 0] = np.convolve(eye_to_eye[:, 0], kernel, mode='same')
    eye_to_eye[:, 1] = np.convolve(eye_to_eye[:, 1], kernel, mode='same')
    eye_to_mouth[:, 0] = np.convolve(eye_to_mouth[:, 0], kernel, mode='same')
    eye_to_mouth[:, 1] = np.convolve(eye_to_mouth[:, 1], kernel, mode='same')

    for i in tqdm(range(len(lm))):
       # Choose oriented crop rectangle.
        x = eye_to_eye[i] - np.flipud(eye_to_mouth[i]) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye[i]) * 2.0, np.hypot(*eye_to_mouth[i]) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg[i] + eye_to_mouth[i] * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2


        # read image
        image_path = f"{root_dir}/frames/s{si}/{i+1:04d}.png"
        

        # read image
        img = PIL.Image.open(image_path)

        output_size=256
        transform_size=256
        #output_size = 1024
        #transform_size = 1024
        enable_padding = True

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                              1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(),
                            PIL.Image.BILINEAR)
        
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        img.save(f"{root_dir}/aligned/s{si}/{i:04d}.jpg")
        if i == 2:  # HACK(demi): overwrite first two frames with third frame
            img.save(f"{root_dir}/aligned/s{si}/{0:04d}.jpg")
            img.save(f"{root_dir}/aligned/s{si}/{1:04d}.jpg")
           

