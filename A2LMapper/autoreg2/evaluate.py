import os
import json
import sys
import argparse
import pickle
import cv2
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import numpy as np
from IPython import embed
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch import optim
from torch.utils import data as data_utils

sys.path.append(".")
from dataset import Audio2FrameDataset
from model import Transformer, LinearMapper, AutoRegMLPMapper
from optim import NoamOpt
from criterion import get_loss
from visualize import visualize, log_metrics

# auxiliary models
from utils.model_utils import setup_model
from HGNet.networks import HGNet, FaceScoreNet, heatmap_to_landmarks, compute_xycoords
from HGNet.common.global_constants import *
from HGNet.convert_weights import DsConvTF, load_tf_vars, ConvTF
from HGNet.common.auxiliary_ftns import box_from_landmarks
from stylegan2_criteria.lpips.lpips import LPIPS
from wav2lip_models import SyncNet_color as SyncNet

# preprocessing
from preprocess import preprocess_images, preprocess_lmks

 

def validate(args, data_loader, model, generator=None, save_path=None):
    model.eval()

    total = len(data_loader)

    losses = []
    latent_losses = []
    perc_losses = []
    img_losses = []
    sync_losses = []
    lmk_losses = []

    for n_iter, (ids, src, prev_tgt, tgt, src_mask, tgt_mask, imgs, lmks, mels) in tqdm(enumerate(data_loader),total=total, desc="validate"):

        bsz = src.size(0)
        seq_len = src.size(1)
        assert src.shape == (bsz, seq_len, model.input_dim)

        if args.sync_loss > 0.0:  # HACK(demi): reshape back, syncnet_T=5
            assert seq_len == 5
            src = src.reshape(bsz*5, 1, model.input_dim)
            prev_tgt = prev_tgt.reshape(bsz*5, 1, model.output_dim)
            tgt = tgt.reshape(bsz*5, 1, model.output_dim)
            
            # HACK(demi): since seq_len=1, mask should be all 1s
            if torch.sum(src_mask).item() != np.prod(src_mask.shape):  # last frames
                # HACK(demi): continue for now
                print("drop n<5 frames, continue")
                continue 
            assert torch.sum(src_mask).item() == np.prod(src_mask.shape)
            assert torch.sum(tgt_mask[:,:1,:1]).item() == np.prod(tgt_mask[:,:1,:1].shape)
            assert src_mask.shape == (bsz, 1, 5) and tgt_mask.shape == (bsz, 5, 5)
            src_mask = torch.ones(bsz*5, 1, 1).type(src_mask.dtype)
            tgt_mask = torch.ones(bsz*5, 1, 1).type(tgt_mask.dtype)
            
            if imgs.shape == (bsz, seq_len):
                imgs = imgs.reshape(bsz*5, 1)
            else:
                assert imgs.shape == (bsz, seq_len, args.image_size, args.image_size, 3)
                imgs = imgs.reshape(bsz*5, 1, args.image_size, args.image_size, 3)
            if lmks.shape == (bsz, seq_len):
                lmks = lmks.reshape(bsz*5, 1)
            else:
                lmks = lmks.reshape(bsz*5, 1, lmks.shape[2], lmks.shape[3])
            mels = mels.reshape(bsz*5, 1, 80, 16)

            bsz = src.shape[0]
            seq_len = 1  # HACK(demi): enforced for sync loss
        
        src = src.cuda()
        prev_tgt = prev_tgt.cuda()
        tgt = tgt.cuda()
        src_mask = src_mask.cuda()
        tgt_mask = tgt_mask.cuda()
        imgs = imgs.cuda()
        imgs = imgs.type(torch.float32) / 255.  # convert to float
        lmks = lmks.cuda()
        mels = mels.cuda()
        
        predict_tgt = model(src, prev_tgt, src_mask, tgt_mask)
        assert predict_tgt.shape == tgt.shape
        assert predict_tgt.shape == (bsz, seq_len, model.output_dim)

        loss, viz_info = get_loss(args, predict_tgt, tgt, src_mask, imgs, lmks, mels, aux_models, viz=True)

        losses.append(loss.item())
        latent_losses.append(viz_info["latent_loss"])
        perc_losses.append(viz_info.get("perceptual_loss", 0))
        img_losses.append(viz_info.get("image_loss", 0))
        sync_losses.append(viz_info.get("sync_loss", 0))
        lmk_losses.append(viz_info.get("lmk_loss",0))

    
    eval_dict = {"type": "scaled",
        "loss": np.mean(losses),
        "latent_loss": np.mean(latent_losses),  # scaled 512*18
        "perc_loss": np.mean(perc_losses),   # scaled 100
        "img_loss": np.mean(img_losses),  # scaled 512*18
        "sync_loss": np.mean(sync_losses),
        "lmk_loss": np.mean(lmk_losses) }  # scaled 10
    
    with open(save_path, "w") as f:
        json.dump(eval_dict, f)
 

def inference(args, input_dim, output_dim, neutral_vec, model, mode="autoreg", save_path=None):
    assert mode in ["autoreg", "tf"]
    assert args.pca is None  # deprecated, to simplify code
    model.eval()
    neutral_vec = neutral_vec.reshape(1, -1).type(torch.float32)
    losses = []
    #output_dim = model.output_dim
    #org_output_dim = model.org_output_dim
    cur_vec = neutral_vec.reshape(-1) - neutral_vec.reshape(-1)  
    
    counter = 0
    new_vecs_list = []

    def subsequent_mask(size):
        mask = 1-np.triu(np.ones((1,size,size)),k=1)
        return torch.from_numpy(mask).cuda()

    # NB(demi): first generate all predictions
    original_seq_len = args.seq_len
    
    args.seq_len = original_seq_len
    test_dataset = Audio2FrameDataset(args, args.test_path, "test", sample="sparse", input_dim=input_dim, output_dim=output_dim)
    data_loader = data_utils.DataLoader(
        test_dataset, batch_size=1,
        num_workers=args.num_workers,
        shuffle=False)  # HACK(demi): use batch size = seq len
    
    predict_tgts = []
    total=len(data_loader)
    for n_iter, (ids, src, prev_tgt, tgt, src_mask, tgt_mask, imgs, lmks, mels) in tqdm(enumerate(data_loader),total=total, desc="inference: get predictions"):
        
        bsz = src.size(0)
        seq_len = src.size(1)
        assert bsz == 1

        src = src.cuda()
        prev_tgt = prev_tgt.cuda()  # not actually used
        tgt = tgt.cuda()  # not actually used
        src_mask = src_mask.cuda()
        tgt_mask = tgt_mask.cuda()  # not actually used
        
        memory = model.encode(src, src_mask)
        prev_tgt = cur_vec.reshape(1, 1, model.output_dim)
        for i in range(seq_len):
            tgt_mask = subsequent_mask(prev_tgt.size(1))
            out = model.decode(memory, src_mask, prev_tgt, tgt_mask, gen=False)
            assert out.shape[:2] == (bsz, prev_tgt.size(1))
            predict_tgt = model.generate(out[:, -1])
            
            predict_tgts.append(predict_tgt.detach().cpu().numpy())
            counter += 1
            
            if mode == "tf":
                predict_tgt = tgt[0, i]  # teacher forcing
            prev_tgt = torch.cat([prev_tgt, predict_tgt.reshape(1,1,model.output_dim)], dim=1)
            cur_vec = predict_tgt
            assert prev_tgt.shape == (bsz, i+2, model.output_dim)
    
    predict_tgts = np.stack(predict_tgts, axis=0)


    # now evaluate loss
    args.seq_len = 1
    test_dataset = Audio2FrameDataset(args, args.test_path, "test", sample="sparse", input_dim=input_dim, output_dim=output_dim)
    # NB(demi): batch size = 5, each batch for sync loss
    data_loader = data_utils.DataLoader(
        test_dataset, batch_size=10,
        num_workers=args.num_workers,
        shuffle=False)     # each k

    counter = 0
    losses = []
    latent_losses = []
    perc_losses = []
    img_losses = []
    sync_losses = []
    lmk_losses = []
    total=len(data_loader)
    for n_iter, (ids, src, prev_tgt, tgt, src_mask, tgt_mask, imgs, lmks, mels) in tqdm(enumerate(data_loader),total=total, desc="inference: get losses"):
        
        bsz = src.size(0)
        seq_len = src.size(1)

        src = src.cuda()
        prev_tgt = prev_tgt.cuda()  # not actually used
        tgt = tgt.cuda()  # not actually used
        src_mask = src_mask.cuda()
        tgt_mask = tgt_mask.cuda()  # not actually used
        imgs = imgs.cuda()
        imgs = imgs.type(torch.float32) / 255.  # convert to float
        lmks = lmks.cuda()
        mels = mels.cuda()
 
        # get predictions
        if bsz < 10:
            print("last batch pass")  # HACK(demi)
            continue
        for t in range(10):  # double check
            assert ids[t][0].item() == counter + t
        predict_tgt = torch.Tensor(predict_tgts[counter:counter+10]).type(tgt.dtype).cuda()
        assert tgt.shape == predict_tgt.shape
        
        loss, viz_info = get_loss(args, predict_tgt, tgt, src_mask, imgs, lmks, mels, aux_models, viz=True)
        
        # log losses
        losses.append(loss.item())
        latent_losses.append(viz_info["latent_loss"])
        perc_losses.append(viz_info.get("perceptual_loss", 0))
        img_losses.append(viz_info.get("image_loss", 0))
        sync_losses.append(viz_info.get("sync_loss", 0))
        lmk_losses.append(viz_info.get("lmk_loss",0))

 
        counter += 10
    

    eval_dict = {"type": "scaled",
        "loss": np.mean(losses),
        "latent_loss": np.mean(latent_losses),  # scaled 512*18
        "perc_loss": np.mean(perc_losses),   # scaled 100
        "img_loss": np.mean(img_losses),  # scaled 512*18
        "sync_loss": np.mean(sync_losses),
        "lmk_loss": np.mean(lmk_losses) }  # scaled 10
 
    with open(save_path, "w") as f:
        json.dump(eval_dict, f)
 

    eval_dict = {"type": "noscale",
        "loss": np.mean(losses),
        "latent_loss": np.mean(latent_losses) / (512*18.),  # scaled 512*18
        "perc_loss": np.mean(perc_losses) / 100.,   # scaled 100
        "img_loss": np.mean(img_losses) / (512*18.),  # scaled 512*18
        "sync_loss": np.mean(sync_losses) / 10.,
        "lmk_loss": np.mean(lmk_losses) / 10.}  # scaled 10
 
    with open(save_path, "a") as f:
        f.write("\n")
        json.dump(eval_dict, f)
 

if __name__ == "__main__":
    print("now parser")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/timit/videos")  # parent folder of train video folders
    parser.add_argument("--test_path", type=str, default="data/timit/videos/test/s3")  # only support one test video for now, path to the specific test video folder
    parser.add_argument("--neutral_path", type=str, default=None)
    parser.add_argument("--output", type=str, default="output/debug")
    parser.add_argument("--load_ckpt", type=str, default=None)  # NB(demi): currently, not used for evaluation
    parser.add_argument("--load_optim", type=int, default=0)


    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mode", type=str, default="default", help="support: [default]")
    parser.add_argument("--latent_type", type=str, default="w+", help="latent types: [w+, stylespace]")

    # image loss
    parser.add_argument("--image_type", type=str, default="none", help="[none, gt, gan]")
    parser.add_argument("--image_size", type=int, default=1024, help="must be squre, dataloader first resize image to image size")
    parser.add_argument("--image_mouth", type=int, default=0, help="mouth loss or not ")  # currently only support all vs mouth only

    parser.add_argument("--img_loss", type=float, default=0.0)
    parser.add_argument("--mouth_box", type=int, nargs="+", default=None)
    parser.add_argument("--lmk_loss", type=float, default=0.0)
    parser.add_argument("--face_box", type=int, nargs="+", default=[25,63,231,256])  # set for img_size=256, (x1,y1,x2,y2)
    parser.add_argument("--perceptual_loss", type=float, default=0.0)
    parser.add_argument("--sync_loss", type=float, default=0.0)
    
    # model
    parser.add_argument("--pca", type=str, default=None)
    parser.add_argument("--pca_dims", type=int, nargs="+", default=None)
    parser.add_argument("--model", type=str, default="transformer")
    parser.add_argument("--use_pose", type=int, default=0)

    # MLP parameters
    parser.add_argument("--nlayer", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    
    # transformer parameters
    parser.add_argument("--h", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--N", type=int, default=5)


    # optimization
    parser.add_argument("--optim",  type=str, default="noam")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=320)

    # NoamOpt
    parser.add_argument("--warmup", type=int, default=4000)
    
    # Adam
    parser.add_argument("--lr", type=float, default=1e-3)  # only for adam
    parser.add_argument("--lr_reduce", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--wd", type=float, default=0)

    # logging and visualize
    parser.add_argument("--viz_every", type=int, default=300)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=10000)

    args = parser.parse_args()
    
    # NB(demi): force all loss to be > 0
    args.lmk_loss = 0.1
    args.img_loss = 0.1
    args.sync_loss = 0.1
    args.perceptual_loss = 0.1
    args.image_mouth = 1  # only on mouth for now
    args.image_type = "gt"

    
    print("args=",args)
        
    args.viz_dir = f"{args.output}/viz" 
    os.makedirs(args.viz_dir, exist_ok=True)
    
    # check conflicts
    assert args.model in ["transformer", "mlp", "linear"]
    if args.model == "mlp":
        if args.seq_len != 1:
            print("warning: seq_len is forced to set to 1")
            args.seq_len = 1
    

    # load pca
    if args.neutral_path is None:
        assert args.pca is None, "pca need to specify corresponding neutral path"
        args.neutral_vec = np.load(f"{args.train_path}/neutral_{args.latent_type}.npy").reshape(1,-1)
    else:
        args.neutral_vec = np.load(args.neutral_path).reshape(1, -1)

    device = torch.device("cuda")
    args.device = device
    if args.pca is not None:
        with open(args.pca, "rb") as f:
            args.pca = pickle.load(f)
        args.pca_neutral_vec = args.pca.transform(args.neutral_vec*0)  # need to subtract neutral vec (so resulting 0 vector)
        
        if args.pca_dims is None:
            args.pca_dims = list(range(args.pca.n_components_))

    input_dim = 512
    if args.latent_type == "w+":
        output_dim=512*18
    elif args.latent_type == "stylespace":
        output_dim=9088
    else:
        raise NotImplementedError
    org_output_dim = output_dim
    if args.use_pose == 1:
        input_dim = input_dim + 6
    elif args.use_pose == 2:
        input_dim = input_dim + 8
    elif args.use_pose == 3:
        input_dim = input_dim + 6 + 2 + 24


    # load image-loss related models
    aux_models = {}
    print("load models")
    if args.image_type != "none":
        code_path = "A2LMapper/autoreg2"
        # load stylegan2
        stylegan2_path = "A2LMapper/autoreg2/checkpoints/e4e_ffhq_encode.pt"
        net, opts = setup_model(stylegan2_path, "cuda")
        generator = net.decoder
        generator.eval()
        aux_models["generator"] = generator
        for p in generator.parameters():
            p.requires_grad = False
        
        # load landmark detector
        if args.lmk_loss > 0.0:
            # currently don't use face predictor, use predefined boxes (from face predictor for fix size aligned image)
            hgnet_model_path = "A2LMapper/autoreg2/HGNet/model_410400.pt"
            checkpoint = torch.load(hgnet_model_path)
            if "model_type" not in checkpoint or checkpoint["model_type"] == "hgnet":
                hgnet = HGNet(INNER_NUM_LANDMARKS).to(device)
            elif checkpoint["model_type"] == "hgnet_tf":
                hgnet = HGNet(INNER_NUM_LANDMARKS, conv=DsConvTF).to(device)
            else:
                raise Exception(f"{checkpoint['model_type']} is not supported.")

            hgnet.load_state_dict(checkpoint['model_state_dict'])
            hgnet.eval()
            for p in hgnet.parameters():
                p.requires_grad = False
            aux_models["hgnet"] = hgnet

            xycoords = compute_xycoords(IMAGE_SIZE_256 // 4, INNER_NUM_LANDMARKS)
            xycoords = torch.Tensor(xycoords).to(device)
            aux_models["xycoords"] = xycoords

        # load perceptual loss
        if args.perceptual_loss > 0.0:
           lpips_loss = LPIPS(net_type="alex").cuda()  # NB(demi): use alex for now 
           aux_models["lpips_loss"] = lpips_loss

        # load sync loss model
        if args.sync_loss > 0.0:
            syncnet = SyncNet().cuda()
            for p in syncnet.parameters():
                p.requires_grad = False  # Q(demi): is it necessary? how about other models? we didn't specify it to be differentiable in optimizer?
            aux_models["syncnet"] = syncnet


    print("preprocess")
    # preprocess dataset
    if args.image_type != "none":
        if args.img_loss > 0.0:
            preprocess_images(args)
        if args.lmk_loss > 0.0:
            preprocess_lmks(args, aux_models["hgnet"], aux_models["xycoords"])

    # datasets
    print("loading datasets")
    train_dataset = Audio2FrameDataset(args, args.train_path, "train", sample="dense", input_dim=input_dim, output_dim=output_dim) 
    val_dataset = Audio2FrameDataset(args, args.train_path, "val", sample="sparse", input_dim=input_dim, output_dim=output_dim)
    
  
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers)
    val_data_loader = data_utils.DataLoader(
        val_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False)

    if not os.path.exists(f"{args.output}/inference"):
        os.makedirs(f"{args.output}/inference")
    
     
    # models and optimizers
    print("loading models")
    
    # transform to model's input dim and output dim
    if args.pca is not None:
        output_dim = len(args.pca_dims)
    if args.model == "transformer":
        model = Transformer(args, input_dim=input_dim, output_dim=output_dim)
        print("loaded transformer on cpu")
        model = model.to(device)
        d_model = model.model.src_embed[0].d_model
    elif args.model == "linear":
        model = LinearMapper(args, input_dim=input_dim, output_dim=output_dim).to(device)
        d_model = 512
    else:
        model = AutoRegMLPMapper(args, input_dim=input_dim, output_dim=output_dim).to(device)
        d_model = args.hidden_dim

    model.org_output_dim = org_output_dim

   
    
    neutral_vec = torch.from_numpy(args.neutral_vec).to(device)
    
    os.makedirs(f"{args.output}/evaluation", exist_ok=True)

    # inference evaluation
    checkpoints = glob(f"{args.output}/best_val*.pt")
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    print(f"loading checkpoint from {checkpoint}")
    model.load_state_dict(torch.load(f"{checkpoint}")["model_state_dict"])

    with torch.no_grad(): 
        inference(args, input_dim, output_dim, neutral_vec, model, mode="autoreg",
            save_path=f"{args.output}/evaluation/best_val_inference.json")
        inference(args, input_dim, output_dim, neutral_vec, model, mode="tf",
            save_path=f"{args.output}/evaluation/tf_best_val_inference.json")


     
    # validate all epoch checkpoints
    checkpoints = glob(f"{args.output}/epoch*.pt")
    for checkpoint in checkpoints:
        epoch = int(checkpoint.split("/")[-1].split("_")[0].replace("epoch",""))
        prefix = f"epoch{epoch:02d}"  # reformat to 02d so sorting is accurate
        print(f"loading checkpoint from {checkpoint}")
        model.load_state_dict(torch.load(f"{checkpoint}")["model_state_dict"])
        with torch.no_grad():
            validate(args, val_data_loader, model, save_path=f"{args.output}/evaluation/{prefix}_validation.json")
    
    # TODO(demi): plot loss curve over epochs
    # plot validation loss curves
    losses = []
    latent_losses = []
    perc_losses = []
    img_losses = []
    lmk_losses = []
    sync_losses = []
    eval_paths = sorted(glob(f"{args.output}/evaluation/*validation.json"))
    print("eval_paths=",eval_paths)

    for eval_path in eval_paths:
        json_obj = json.load(open(eval_path))
        
        losses.append(json_obj["loss"])
        latent_losses.append(json_obj["latent_loss"])
        perc_losses.append(json_obj["perc_loss"])
        img_losses.append(json_obj["img_loss"])
        lmk_losses.append(json_obj["lmk_loss"])
        sync_losses.append(json_obj["sync_loss"])

    x = list(range(len(losses)))
    plt.plot(x, losses, label="total")
    plt.plot(x, latent_losses, label="latent")
    plt.plot(x, perc_losses, label="perc")
    plt.plot(x, img_losses, label="img")
    plt.plot(x, sync_losses, label="sync")
    plt.plot(x, lmk_losses, label="lmk")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(f"{args.output}/evaluation/validation_curve.png")
       
    

