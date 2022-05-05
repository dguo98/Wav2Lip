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
from model import Transformer, LinearMapper, AutoRegMLPMapper
from optim import NoamOpt

def train(args, data_loader, model, opt):
    model.train()

    total = len(data_loader)

    losses = []
    for n_iter, (ids, src, prev_tgt, tgt, src_mask, tgt_mask) in tqdm(enumerate(data_loader),total=total, desc="train"):

        opt.zero_grad()

        bsz = src.size(0)
        seq_len = src.size(1)
        assert src.shape == (bsz, seq_len, model.input_dim)
        
        src = src.cuda()
        prev_tgt = prev_tgt.cuda()
        tgt = tgt.cuda()
        src_mask = src_mask.cuda()
        tgt_mask = tgt_mask.cuda()
        
        predict_tgt = model(src, prev_tgt, src_mask, tgt_mask)
        assert predict_tgt.shape == tgt.shape
        
        if args.seq_len == 0:
            loss = torch.nn.MSELoss(reduction="mean")(predict_tgt,tgt)
            loss = loss*model.output_dim
        else:
            loss_mask = src_mask.reshape(bsz, seq_len, 1).float()
            loss = torch.nn.MSELoss(reduction="none")(predict_tgt, tgt) 
            #print("mean=", torch.nn.MSELoss(reduction="mean")(predict_tgt,tgt))
            #print("try to run sum (loss*loss_mask) / sum(loss_mask)")
            #embed()
            assert loss.shape == (bsz, seq_len, model.output_dim)
            
            loss = torch.sum(loss * loss_mask) / torch.sum(loss_mask) #.type(dtype=loss.dtype)
            #loss = loss * model.output_dim  # only average across batch, and sequence, but not dimensions

        # MSE Loss for now
        losses.append(loss.item())
       
        if n_iter % 100 == 0 and n_iter > 0 and args.mode == "debug":
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
        assert src.shape == (bsz, seq_len, model.input_dim)
        
        src = src.cuda()
        prev_tgt = prev_tgt.cuda()
        tgt = tgt.cuda()
        src_mask = src_mask.cuda()
        tgt_mask = tgt_mask.cuda()
        
        predict_tgt = model(src, prev_tgt, src_mask, tgt_mask)
        if args.mode == "debug":
            predict_tgt = predict_tgt * 0
        assert predict_tgt.shape == tgt.shape
        assert predict_tgt.shape == (bsz, seq_len, model.output_dim)

        loss_mask = src_mask.reshape(bsz, seq_len, 1).float()
        loss = torch.nn.MSELoss(reduction="none")(predict_tgt, tgt) 
        assert loss.shape == (bsz, seq_len, model.output_dim)
        loss = torch.sum(loss * loss_mask) / torch.sum(loss_mask) #.type(loss.dtype)
        #loss = loss * model.output_dim  # only average across batch, and sequence, but not dimensions

        # MSE Loss for now
        losses.append(loss.item())

    return np.mean(losses)
 
def inference(args, data_loader, model, infer_dir, neutral_vec, mode="autoreg"):
    assert mode in ["autoreg", "tf"]
    model.eval()

    total = len(data_loader)
    neutral_vec = neutral_vec.reshape(1, -1).type(torch.float32)

    losses = []
    
    output_dim = model.output_dim
    org_output_dim = model.org_output_dim

    cur_vec = neutral_vec.reshape(-1) - neutral_vec.reshape(-1)  
    if args.pca is not None:  
        # quite hacky now -- in theory we can rewrite pca on torch
        cur_vec = torch.zeros_like(neutral_vec).cpu().numpy().reshape(1, org_output_dim)
        cur_vec = args.pca.transform(cur_vec)[:, args.pca_dims]
        cur_vec = torch.from_numpy(cur_vec).cuda()
         
        
    
    counter = 0
    # todo(demi): support rolling/overlapping source context

    def subsequent_mask(size):
        mask = 1-np.triu(np.ones((1,size,size)),k=1)
        return torch.from_numpy(mask).cuda()
    
    new_vecs_list = []

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
            
            if args.pca is not None:
                vecs = predict_tgt.detach().cpu().numpy().reshape(1,output_dim)
                new_vecs = np.repeat(args.pca_neutral_vec, len(vecs), axis=0)
                new_vecs[:, args.pca_dims] = vecs
                new_vecs = args.pca.inverse_transform(new_vecs)
                new_vecs = new_vecs + args.neutral_vec
                new_vecs_list.append(new_vecs.reshape(-1))
                new_vecs = torch.from_numpy(new_vecs).reshape(-1)
                torch.save(new_vecs, f"{infer_dir}/predict_{counter:06d}.pt")
            else:
                new_vecs_list.append((predict_tgt+neutral_vec).reshape(-1).detach().cpu().numpy())
                torch.save(predict_tgt.reshape(-1) + neutral_vec.reshape(-1), f"{infer_dir}/predict_{counter:06d}.pt")
            counter += 1
            
            if mode == "tf":
                predict_tgt = tgt[0, i]  # teacher forcing
            prev_tgt = torch.cat([prev_tgt, predict_tgt.reshape(1,1,output_dim)], dim=1)
            cur_vec = predict_tgt
            assert prev_tgt.shape == (bsz, i+2, output_dim)
    
    final_vecs = np.stack(new_vecs_list, axis=0)
    if args.latent_type == "stylespace":
        np.save(f"{infer_dir}/predict_stylespace.npy", final_vecs)
        os.system(f"rm {infer_dir}/predict_*.pt")
    assert final_vecs.shape == (len(data_loader), org_output_dim)
    return 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/timit/videos")  # parent folder of train video folders
    parser.add_argument("--test_path", type=str, default="data/timit/videos/test/s3")  # only support one test video for now, path to the specific test video folder
    parser.add_argument("--neutral_path", type=str, default=None)
    parser.add_argument("--output", type=str, default="output/debug")
    parser.add_argument("--load_ckpt", type=str, default=None)
    parser.add_argument("--load_optim", type=int, default=0)


    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mode", type=str, default="default", help="support: [default]")
    parser.add_argument("--latent_type", type=str, default="w+", help="latent types: [w+, stylespace]")

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

    args = parser.parse_args()
    
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

    
    # datasets
    print("loading datasets")
    train_dataset = Audio2FrameDataset(args, args.train_path, "train", sample="dense", input_dim=input_dim, output_dim=output_dim) 
    val_dataset = Audio2FrameDataset(args, args.train_path, "val", sample="sparse", input_dim=input_dim, output_dim=output_dim)
    test_dataset = Audio2FrameDataset(args, args.test_path, "test", sample="sparse", input_dim=input_dim, output_dim=output_dim)
    
    if args.mode == "sanity_check":
        train_dataset.data_len = 100
        val_dataset.data_len=100
        test_dataset.data_len=50

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

    if args.load_ckpt is not None:
        print(f"loading checkpoint from  {args.load_ckpt}")
        model.load_state_dict(torch.load(f"{args.load_ckpt}")["model_state_dict"])

    print("loading optimizer") 
    if args.optim == "noam":
        optimizer = NoamOpt(d_model, 2, args.warmup,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9,0.98), eps=1e-9))
    else:
        assert args.optim == "adam"
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_reduce, patience=args.lr_patience)
    
    if args.load_ckpt is not None and args.load_optim:
        assert args.optim == "noam"
        steps = 20 * len(train_data_loader)
        optimizer._step = steps
        print("now learning rate=", optimizer.rate())
        #embed()
    # training and inference
    print("start training")
    neutral_vec = torch.from_numpy(args.neutral_vec).to(device)
    train_losses = []
    val_losses = []
    
    best_val = None
    last_val = None
    last_train = None
    for epoch in range(args.epochs):
        train_loss = train(args, train_data_loader, model, optimizer)
        print(f"EPOCH[{epoch}] | Train Loss : {train_loss:.3f}")
        val_loss = validate(args, val_data_loader, model)
        print(f"EPOCH[{epoch}] | Val Loss : {val_loss:.3f}")

        if args.optim == "adam":
            scheduler.step(val_loss)
            print("LR:", optimizer.param_groups[0]['lr'])

        train_losses.append(train_loss)
        val_losses.append(val_loss)


        if best_val is None or val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model.state_dict()}, f"{args.output}/best_val_checkpoint.pt")
        last_val = val_loss 
        last_train = train_loss

    torch.save({"model_state_dict": model.state_dict()}, f"{args.output}/last_checkpoint.pt")
    model.load_state_dict(torch.load(f"{args.output}/best_val_checkpoint.pt")["model_state_dict"])


    neutral_vec = torch.from_numpy(args.neutral_vec).to(device)  # now, train and test has to be the same face
       
    val_loss = validate(args, val_data_loader, model)
    test_loss = validate(args, test_data_loader, model)
    print("load model val loss=", val_loss)
    print("load model test loss=", test_loss)


    infer_dir = f"{args.output}/tf_inference"
    if not os.path.exists(infer_dir):
        os.makedirs(infer_dir)
    os.system(f"cp {args.test_path}/audio.wav {infer_dir}/")
    inference(args, test_data_loader, model, infer_dir, neutral_vec, mode="tf")

    infer_dir = f"{args.output}/inference"
    if not os.path.exists(infer_dir):
        os.makedirs(infer_dir)
    os.system(f"cp {args.test_path}/audio.wav {infer_dir}/")
    inference(args, test_data_loader, model, infer_dir, neutral_vec, mode="autoreg")


    x = list(range(args.epochs))
    plt.plot(x, train_losses, label="train")
    plt.plot(x, val_losses, label="val")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(f"{args.output}/loss_curve.png")

    with open(f"{args.output}/hparams.pkl", "wb") as f:
        pickle.dump(args, f)
    with open(f"{args.output}/metrics.json", "w") as f:
        json.dump({"best_val_loss": best_val, "last_val_loss": last_val, "last_train_loss": last_train}, f)

