import torch
import numpy as np
import os
import sys
from PIL import Image
import cv2

def convert_image(img):
    img = np.clip(img, 0, 1)
    img = (img*255).astype(np.uint8)
    return Image.fromarray(img)

def images_to_lmks(args, hgnet, images):
    raise NotImplementedError

def get_loss(args, predict_tgt, tgt, src_mask, imgs, aux_models, viz=False):
    bsz, seq_len, latent_dim = tgt.shape
    neutral_vec = torch.from_numpy(args.neutral_vec).cuda().reshape(-1, 18, 512)  # now, train and test has to be the same face

    loss_mask = src_mask.reshape(bsz, seq_len, 1).float()
    latent_loss = torch.nn.MSELoss(reduction="none")(predict_tgt, tgt) 
    latent_loss = torch.sum(latent_loss * loss_mask) / torch.sum(loss_mask)  # average across bsz, seq_len, sum over latent dims

    # NB(demi): revisit scaling regime
    image_loss = 0
    lmk_loss = 0
    perceptual_loss = 0
    viz_info = {}

    if args.image_type != "none":
        # get images
        generator = aux_models["generator"]
        latents = predict_tgt.reshape(-1, 18, 512) + neutral_vec
        predict_imgs, _ = generator([latents], input_is_latent=True, randomize_noise=False, return_latents=True) 

        # predict_imgs: [3, H, W,], values in [-1, 1]
        assert predict_imgs.shape[:2] == (bsz*seq_len, 3)
        predict_imgs = predict_imgs.permute((0, 2,3,1))
        assert predict_imgs.shape[0] == bsz*seq_len and predict_imgs.shape[-1] == 3
        predict_imgs = (predict_imgs+1)/2
        # todo(demi): optionally, clamp to (0,1)
        if predict_imgs.shape[1] != args.image_size:
            viz_info["org_predict_img"] = predict_imgs[0].detach().cpu().numpy()
            # predict_imgs: [bsz*seq_len, H,W,3]
            predict_imgs = predict_imgs.permute((0,3,1,2))
            predict_imgs =  torch.nn.functional.interpolate(predict_imgs, size=(args.image_size, args.image_size), mode='bilinear')
            # predict_imgs: [bsz*seq_len, 3, H, W] 
            predict_imgs = predict_imgs.permute((0, 2,3,1))
            assert predict_imgs.shape == (bsz*seq_len, args.image_size, args.image_size, 3)
        predict_imgs = predict_imgs.reshape(bsz, seq_len, args.image_size, args.image_size, 3)

        viz_info["predict_img"] = predict_imgs[0][0].detach().cpu().numpy()
        viz_info["tgt_img"] = imgs[0][0].detach().cpu().numpy()

        if args.img_loss > 0.0:
            if args.image_mouth == 1:
                x1, y1, x2, y2 = args.mouth_box
               
                # visualize predict img with mask, and tgt img with 
                blue_img = np.ones((args.image_size, args.image_size, 3), np.float32)
                blue_img[:, :, 0] = 0
                mouth_mask = np.zeros((args.image_size, args.image_size, 3), np.float32)
                mouth_mask[x1:x2, y1:y2] = 1

                predict_img = viz_info["predict_img"]
                comp_predict_img = predict_img * (1-mouth_mask) + blue_img * mouth_mask
                viz_info["mouth_predict_img"] = predict_img[x1:x2, y1:y2]
                viz_info["comp_predict_img"] = comp_predict_img

                # for loss
                predict_imgs = predict_imgs[:, :, x1:x2, y1:y2]
                assert predict_imgs.shape == imgs.shape
 
            image_loss = torch.nn.MSELoss(reduction="none")(predict_imgs, imgs) 
            image_loss = torch.sum(image_loss * loss_mask.reshape(bsz, seq_len, 1, 1, 1)) / torch.sum(loss_mask)
            image_loss = image_loss / np.prod(predict_imgs[0][0].shape) * latent_dim  # rescale
            assert args.image_mouth == 1 or np.prod(predict_imgs[0][0].shape) == 3*(args.image_size**2)
            
            # difference image
            diff_img = (predict_imgs[0][0] - imgs[0][0]).detach().cpu().numpy()
            diff_img = np.max(np.abs(diff_img), axis=-1)
            viz_info["diff_img"] = diff_img


            viz_info["image_loss"] = image_loss.item()

        if args.lmk_loss > 0.0:
            raise NotImplementedError

        if args.perceptual_loss > 0.0:
            raise NotImplementedError
    viz_info["latent_loss"] = latent_loss.item()

    loss = (1-args.img_loss-args.lmk_loss-args.perceptual_loss) * latent_loss + \
        args.img_loss * image_loss + \
        args.lmk_loss * lmk_loss + \
        args.perceptual_loss * perceptual_loss
    return loss, viz_info
