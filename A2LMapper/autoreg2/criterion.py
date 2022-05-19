import torch
import numpy as np
import os
import sys
from PIL import Image
import cv2
from IPython import embed

sys.path.append(".")
from visualize import viz_lmk
from HGNet.networks import HGNet, FaceScoreNet, heatmap_to_landmarks, compute_xycoords
from HGNet.common.global_constants import *
from HGNet.convert_weights import DsConvTF, load_tf_vars, ConvTF
from HGNet.common.auxiliary_ftns import box_from_landmarks

def convert_image(img):
    img = np.clip(img, 0, 1)
    img = (img*255).astype(np.uint8)
    return Image.fromarray(img)

def inverse_tensor2im(args, img):
    img = img.reshape(-1, args.image_size, args.image_size, 3)  # [B, H, W, 3], value in [0,1)
    img = img.permute((0, 3, 1, 2))
    assert img.shape[1] == 3
    img = (img * 2) - 1
    return img

def images_to_lmks(args, hgnet, xycoords, images):
    orig_image = images[:,:,:,[2,1,0]]*255

    # tied to size 256
    y2=256
    x1,y1,x2,y2 = args.face_box
    face = orig_image[:, y1:y2, x1:x2]  # [H',W',3]

    face = face.permute((0,3,1,2))  # [bsz, 3, H, W]
    resized_face = torch.nn.functional.interpolate(face, size=(IMAGE_SIZE_256, IMAGE_SIZE_256),mode="bilinear")
    
    # now output: rezied_face: [1, 3, H, W]
    resized_face = torch.permute(resized_face, (0,2,3,1))
    
    input_img = (resized_face[:,:, :, [2,1,0]] / 255.0 - 0.5) * np.sqrt(2.0)
    
    pt_in = torch.permute(input_img, (0, 3, 1, 2))
    pt_out = hgnet(pt_in)
    torch_pred_ldmk = heatmap_to_landmarks(pt_out[-1], IMAGE_SIZE_256, xycoords)
   
    return torch_pred_ldmk

def get_loss(args, predict_tgt, tgt, src_mask, imgs, lmks, aux_models, viz=False):
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
            #viz_info["org_predict_img"] = predict_imgs[0].detach().cpu().numpy()
            # predict_imgs: [bsz*seq_len, H,W,3]
            predict_imgs = predict_imgs.permute((0,3,1,2))
            predict_imgs =  torch.nn.functional.interpolate(predict_imgs, size=(args.image_size, args.image_size), mode='bilinear')
            # predict_imgs: [bsz*seq_len, 3, H, W] 
            predict_imgs = predict_imgs.permute((0, 2,3,1))
            assert predict_imgs.shape == (bsz*seq_len, args.image_size, args.image_size, 3)
        predict_imgs = predict_imgs.reshape(bsz, seq_len, args.image_size, args.image_size, 3)

        viz_info["predict_img"] = predict_imgs[0][0].detach().cpu().numpy()
        if args.img_loss + args.perceptual_loss > 0:
            viz_info["tgt_img"] = imgs[0][0].detach().cpu().numpy()

        if args.lmk_loss > 0.0:
            predict_lmks = images_to_lmks(args, aux_models["hgnet"], aux_models["xycoords"], predict_imgs.reshape(-1, args.image_size, args.image_size,3))
            predict_lmks = predict_lmks.reshape(bsz, seq_len, 131, 2) 
            assert predict_lmks.shape == lmks.shape
            
            if args.image_mouth == 1:
                predict_lmks = predict_lmks[:, :, 66:105]
                lmks = lmks[:, :, 66:105]
            
            # get loss
            l2_dist = torch.sqrt(torch.sum((predict_lmks-lmks)**2, dim=3))
            assert l2_dist.shape[:2] == (bsz, seq_len)
            lmk_loss = torch.sum(l2_dist * loss_mask.reshape(bsz,seq_len,1))/torch.sum(loss_mask)
            lmk_loss = lmk_loss / l2_dist.shape[2] * 10  # rescale, 10 is magic number
 
            # get visualize
            viz_info["predict_lmk_cv2"] = viz_lmk(args, predict_imgs[0][0], predict_lmks[0][0], lmks[0][0])
            viz_info["lmk_loss"] = lmk_loss.item()

        if args.perceptual_loss > 0.0:
            # only select valid images and average 
            bool_mask = loss_mask.reshape(bsz, seq_len).bool()
            perceptual_loss = aux_models["lpips_loss"](inverse_tensor2im(args, predict_imgs[bool_mask]), inverse_tensor2im(args, imgs[bool_mask]))  # average across bsz, seq_len on valid mask
            perceptual_loss = perceptual_loss * 100  # rescale, magic number
            viz_info["perceptual_loss"] = perceptual_loss.item()

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
                #viz_info["mouth_predict_img"] = predict_img[x1:x2, y1:y2]
                #viz_info["comp_predict_img"] = comp_predict_img

                # for loss
                imgs = imgs[:, :, x1:x2, y1:y2]
                predict_imgs = predict_imgs[:, :, x1:x2, y1:y2]
                assert predict_imgs.shape == imgs.shape
 
            image_loss = torch.nn.MSELoss(reduction="none")(predict_imgs, imgs) 
            image_loss = torch.sum(image_loss * loss_mask.reshape(bsz, seq_len, 1, 1, 1)) / torch.sum(loss_mask)
            image_loss = image_loss / np.prod(predict_imgs[0][0].shape) * latent_dim  # rescale
            assert args.image_mouth == 1 or np.prod(predict_imgs[0][0].shape) == 3*(args.image_size**2)
            
            # difference image
            diff_img = (predict_imgs[0][0] - imgs[0][0]).detach().cpu().numpy()
            diff_img = np.max(np.abs(diff_img), axis=-1)
            #viz_info["diff_img"] = diff_img


            viz_info["image_loss"] = image_loss.item()

    viz_info["latent_loss"] = latent_loss.item()

    loss = (1-args.img_loss-args.lmk_loss-args.perceptual_loss) * latent_loss + \
        args.img_loss * image_loss + \
        args.lmk_loss * lmk_loss + \
        args.perceptual_loss * perceptual_loss
    return loss, viz_info
