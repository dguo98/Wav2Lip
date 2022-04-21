import os
import json
import sys
import argparse
import pickle
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
from model import Transformer, LinearMapper
from optim import NoamOpt

def train(args, data_loader, model, opt):
    model.train()

    total = len(data_loader)

    losses = []
    for n_iter, (ids, src, prev_tgt, tgt, src_mask, tgt_mask) in tqdm(enumerate(data_loader),total=total, desc="train"):

        opt.zero_grad()

        bsz = src.size(0)
        seq_len = src.size(1)
        assert src.shape == (bsz, seq_len, 512)
        
        src = src.cuda()
        prev_tgt = prev_tgt.cuda()
        tgt = tgt.cuda()
        src_mask = src_mask.cuda()
        tgt_mask = tgt_mask.cuda()
        
        predict_tgt = model(src, prev_tgt, src_mask, tgt_mask)
        assert predict_tgt.shape == tgt.shape
        
        loss_mask = src_mask.reshape(bsz, seq_len, 1).float()
        loss = torch.nn.MSELoss(reduction="none")(predict_tgt, tgt) 
        assert loss.shape == (bsz, seq_len, model.output_dim)
        
        loss = torch.sum(loss * loss_mask) / torch.sum(loss_mask) #.type(dtype=loss.dtype)
        loss = loss * model.output_dim  # only average across batch, and sequence, but not dimensions

        # MSE Loss for now
        losses.append(loss.item())
        
        if n_iter % 100 == 0 and n_iter > 0:
           print("loss avg=", np.mean(losses[-100:]))

        loss.backward()
        opt.step()
    return np.mean(losses)
        

def validate(args, data_loader, model):
    model.eval()

    total = len(data_loader)

    losses = []
    for n_iter, (ids, src, prev_tgt, tgt, src_mask, tgt_mask) in tqdm(enumerate(data_loader),total=total, desc="validate"):

        bsz = src.size(0)
        seq_len = src.size(1)
        assert src.shape == (bsz, seq_len, 512)
        
        src = src.cuda()
        prev_tgt = prev_tgt.cuda()
        tgt = tgt.cuda()
        src_mask = src_mask.cuda()
        tgt_mask = tgt_mask.cuda()
        
        predict_tgt = model(src, prev_tgt, src_mask, tgt_mask)
        assert predict_tgt.shape == tgt.shape
        assert predict_tgt.shape == (bsz, seq_len, model.output_dim)

        loss_mask = src_mask.reshape(bsz, seq_len, 1).float()
        loss = torch.nn.MSELoss(reduction="none")(predict_tgt, tgt) 
        assert loss.shape == (bsz, seq_len, model.output_dim)
        loss = torch.sum(loss * loss_mask) / torch.sum(loss_mask) #.type(loss.dtype)
        loss = loss * model.output_dim  # only average across batch, and sequence, but not dimensions

        # MSE Loss for now
        losses.append(loss.item())

    return np.mean(losses)
 

def inference(args, data_loader, model, infer_dir, neutral_vec):
    model.eval()

    total = len(data_loader)
    neutral_vec = neutral_vec.reshape(1, -1).type(torch.float32)

    losses = []
    
    output_dim = model.output_dim
    cur_vec = neutral_vec.reshape(-1)
    counter = 0
    # todo(demi): support rolling/overlapping source context

    def subsequent_mask(size):
        mask = 1-np.triu(np.ones((1,size,size)),k=1)
        return torch.from_numpy(mask).cuda()

    for n_iter, (ids, src, prev_tgt, tgt, src_mask, tgt_mask) in tqdm(enumerate(data_loader),total=total, desc="inference"):
        
        bsz = src.size(0)
        seq_len = src.size(1)
        assert bsz == 1

        src = src.cuda()
        prev_tgt = prev_tgt.cuda()  # not actually used
        tgt = tgt.cuda()  # not actually used
        src_mask = src_mask.cuda()
        tgt_mask = tgt_mask.cuda()  # not actually used
        
        memory = model.encode(src, src_mask)
        prev_tgt = cur_vec.reshape(1, 1, output_dim)
        for i in range(seq_len):
            tgt_mask = subsequent_mask(prev_tgt.size(1))
            out = model.decode(memory, src_mask, prev_tgt, tgt_mask, gen=False)
            assert out.shape[:2] == (bsz, prev_tgt.size(1))
            predict_tgt = model.generate(out[:, -1])
            
            torch.save(predict_tgt, f"{infer_dir}/predict_{counter:06d}.pt")
            counter += 1

            prev_tgt = torch.cat([prev_tgt, predict_tgt.reshape(1,1,output_dim)], dim=1)
            assert prev_tgt.shape == (bsz, i+2, output_dim)

    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data_mlpmap/synthesia-obama-trim")
    parser.add_argument("--test_path", type=str, default="eval_mlpmap/eval-1")
    parser.add_argument("--output", type=str, default="output/debug")


    parser.add_argument("--num_workers", type=int, default=0)
    
    parser.add_argument("--mode", type=str, default="default", help="support: [default]")
    # parser.add_argument("--pca", type=str, default=None)
    # parser.add_argument("--pca_dims", type=int, nargs="+", default=None)
    parser.add_argument("--model", type=str, default="transformer")

    # MLP parameters
    # parser.add_argument("--nlayer", type=int, default=2)
    # parser.add_argument("--hidden_dim", type=int, default=1024)
    
    # transformer parameters
    parser.add_argument("--h", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--N", type=int, default=5)


    # optimization
    parser.add_argument("--optim",  type=str, default="naom")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=320)
    parser.add_argument("--warmup", type=int, default=4000)

    parser.add_argument("--lr", type=float, default=1e-3)  # only for adam
    #parser.add_argument("--lr_reduce", type=float, default=0.5)
    #parser.add_argument("--lr_patience", type=int, default=5)
    #parser.add_argument("--wd", type=float, default=0)


    args = parser.parse_args()


    
    # load pca
    args.neutral_vec = np.load(f"{args.train_path}/frame_latents/neutral.npy").reshape(1, 18*512)
    device = torch.device("cuda")
    args.device = device

    # datasets
    print("loading datasets")
    train_dataset = Audio2FrameDataset(args, args.train_path, "train") 
    val_dataset = Audio2FrameDataset(args, args.train_path, "val")
    test_dataset = Audio2FrameDataset(args, args.test_path, "test")

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers)
    val_data_loader = data_utils.DataLoader(
        val_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False)
    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=1,
        num_workers=args.num_workers,
        shuffle=False)  # HACK(demi): use batch size = seq len

    if not os.path.exists(f"{args.output}/inference"):
        os.makedirs(f"{args.output}/inference")
    
     
    # models and optimizers
    print("loading models and optimizers")
    if args.model == "transformer":
        model = Transformer(args).to(device)
        d_model = model.model.src_embed[0].d_model
    else:
        assert args.model == "linear"
        model = LinearMapper(args).to(device)
        d_model = 512
    
    if args.optim == "naom":
        optimizer = NoamOpt(d_model, 2, args.warmup,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9,0.98), eps=1e-9))
    else:
        assert args.optim == "adam"
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
       
    # training and inference
    print("start training")
    neutral_vec = torch.from_numpy(args.neutral_vec).to(device)
    train_losses = []
    val_losses = []
    
    best_val = None
    for epoch in range(args.epochs):
        train_loss = train(args, train_data_loader, model, optimizer)
        print(f"EPOCH[{epoch}] | Train Loss : {train_loss:.3f}")
        val_loss = validate(args, val_data_loader, model)
        print(f"EPOCH[{epoch}] | Val Loss : {val_loss:.3f}")
        #scheduler.step(val_loss)
        #print("LR:", optimizer.param_groups[0]['lr'])

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if best_val is None or val_loss < best_val:
            torch.save({"model_state_dict": model.state_dict()}, f"{args.output}/best_val_checkpoint.pt")
       

    torch.save({"model_state_dict": model.state_dict()}, f"{args.output}/last_checkpoint.pt")
    model.load_state_dict(torch.load(f"{args.output}/best_val_checkpoint.pt")["model_state_dict"])
        
    infer_dir = f"{args.output}/inference"
    if not os.path.exists(infer_dir):
        os.makedirs(infer_dir)
    os.system(f"cp {args.test_path}/audio.wav {infer_dir}/")
    neutral_vec = torch.from_numpy(args.neutral_vec).to(device)  # now, train and test has to be the same face
    inference(args, test_data_loader, model, infer_dir, neutral_vec)

    x = list(range(args.epochs))
    plt.plot(x, train_losses, label="train")
    plt.plot(x, val_losses, label="val")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(f"{args.output}/loss_curve.png")


    with open(f"{args.output}/metrics.json", "w") as f:
        json.dump({"best_val_loss": best_val}, f)
    

        
