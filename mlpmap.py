import os
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

class Mapper(torch.nn.Module):
    def __init__(self, args, input_dim=512, output_dim=512*18):
        super(Mapper, self).__init__()
        
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = args.hidden_dim
        self.nlayer = args.nlayer
        self.device = device
        self.mode = args.mode

        if self.args.pca is not None:
            self.output_dim = len(self.args.pca_dims)
        
        assert self.nlayer >= 1
        
        self.layers = nn.ModuleList()
        if self.nlayer == 1:
            self.layers.append(nn.Linear(self.input_dim, self.output_dim))
        else:
            self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            for i in range(self.nlayer-2):
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))
   
    def forward(self, audio_vec):
        x = audio_vec
        for layer in self.layers:
            x = layer(x)
        return x


class Audio2FrameDataset(object):
    def __init__(self, args, path, split, load_img=False):
        self.args = args
        self.split = split
        self.path = path

        self.audio_vecs = np.load(f"{path}/wav2lip.npy")
        
        if split == "test":
            self.latent_vecs = np.zeros((len(self.audio_vecs), 18*512))
        else:
            self.latent_vecs = np.load(f"{path}/frame.npy").reshape(-1, 18*512)
        assert len(self.audio_vecs) == len(self.latent_vecs)

        self.mode = args.mode
        assert load_img is False, "loading image not supported yet"

        split_ratio = 0.9
        split_num = int(split_ratio * len(self.audio_vecs))
        if split == "train":
            self.audio_vecs = self.audio_vecs[:split_num]
            self.latent_vecs = self.latent_vecs[:split_num]
        elif split == "val":
            self.audio_vecs = self.audio_vecs[split_num:]
            self.latent_vecs = self.latent_vecs[split_num:]
        else:
            assert split == "test"
            pass
        self.data_len = len(self.audio_vecs)

        assert self.mode == "default"
    
    def get_neutral(self):
        return self.args.neutral_vec

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        latent_vec = self.latent_vecs[idx:idx+1] - self.args.neutral_vec
        if self.args.pca is not None:
            latent_vec = self.args.pca.transform(latent_vec)[:, self.args.pca_dims]
        return np.array([idx]), self.audio_vecs[idx], latent_vec.reshape(-1)
            

def decode(args, vecs):
    """ from model vecs to original latent vecs """
    assert len(vecs.shape) == 2
    assert args.neutral_vec.shape[0] == 1
    if args.pca is not None:
        new_vecs = np.repeat(args.pca_neutral_vec, len(vecs), axis=0)
        new_vecs[:, args.pca_dims] = vecs
        new_vecs = args.pca.inverse_transform(new_vecs)
        new_vecs = new_vecs + args.neutral_vec
        return new_vecs
    else: 
        vecs = vecs + args.neutral_vec
        return vecs
        

def train(args, data_loader, model, optimizer, neutral_vec):
    model.train()

    total = len(data_loader)
    neutral_vec = neutral_vec.reshape(1, -1)

    losses = []
    for n_iter, (ids, audio_vec, latent_vec) in tqdm(enumerate(data_loader),total=total, desc="train"):
        optimizer.zero_grad()

        bsz = audio_vec.size(0)

        audio_vec = audio_vec.cuda()
        latent_vec = latent_vec.cuda()

        predict_vec = model(audio_vec)
        assert predict_vec.size() == (bsz, model.output_dim)


        # MSE Loss for now
        loss = torch.nn.MSELoss(reduction="mean")(predict_vec, latent_vec) 
        loss = loss * model.output_dim  # only average across batch, not items
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
    return np.mean(losses)
        

def validate(args, data_loader, model, neutral_vec):
    model.eval()

    total = len(data_loader)
    neutral_vec = neutral_vec.reshape(1, -1)

    losses = []
    for n_iter, (ids, audio_vec, latent_vec) in tqdm(enumerate(data_loader),total=total, desc="eval"):
        bsz = audio_vec.size(0)

        audio_vec = audio_vec.cuda()
        latent_vec = latent_vec.cuda()

        predict_vec = model(audio_vec)
        assert predict_vec.size() == (bsz, model.output_dim)


        # MSE Loss for now
        loss = torch.nn.MSELoss(reduction="mean")(predict_vec, latent_vec) 
        loss = loss * model.output_dim  # only average across batch, not items
        losses.append(loss.item())

    return np.mean(losses)
        

def inference(args, data_loader, model, infer_dir, neutral_vec):
    model.eval()

    total = len(data_loader)
    neutral_vec = neutral_vec.reshape(1, -1).type(torch.float32)

    losses = []
    
    cur_vec = neutral_vec.reshape(-1)
    counter = 0
    for n_iter, (ids, audio_vec, latent_vec) in tqdm(enumerate(data_loader),total=total, desc="inference"):
        bsz = audio_vec.size(0)

        audio_vec = audio_vec.cuda()
        latent_vec = latent_vec.cuda()
        ids = ids.cuda().reshape(bsz)

        predict_vec = model(audio_vec)
        assert predict_vec.size() == (bsz, model.output_dim)
        
        predict_vec = decode(args, predict_vec.detach().cpu().numpy())
        predict_vec = torch.from_numpy(predict_vec)  # todo(demi): save in torch files are bad

        for i in range(bsz):
            idx = ids[i].item()

            assert idx == counter
            counter += 1
            
            cur_vec = predict_vec[i].reshape(-1)
            torch.save(cur_vec, f"{infer_dir}/predict_{idx:06d}.pt")

    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data_mlpmap/synthesia-obama-trim")
    parser.add_argument("--test_path", type=str, default="eval_mlpmap/eval-1")
    parser.add_argument("--output", type=str, default="output/debug")


    parser.add_argument("--num_workers", type=int, default=0)
    
    parser.add_argument("--mode", type=str, default="default", help="support: [default]")
    parser.add_argument("--nlayer", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--pca", type=str, default=None)
    parser.add_argument("--pca_dims", type=int, nargs="+", default=None)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr_reduce", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=5)

    args = parser.parse_args()

    
    # load pca
    args.neutral_vec = np.load(f"{args.train_path}/frame_latents/neutral.npy").reshape(1, 18*512)
    device = torch.device("cuda")
    args.device = device
    if args.pca is not None:
        with open(args.pca, "rb") as f:
            args.pca = pickle.load(f)
        args.pca_neutral_vec = args.pca.transform(args.neutral_vec)
        
        if args.pca_dims is None:
            args.pca_dims = list(range(args.pca.n_components_))


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
        test_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False)

    if not os.path.exists(f"{args.output}/inference"):
        os.makedirs(f"{args.output}/inference")
    
     
    # models and optimizers
    print("loading models and optimizers")
    model = Mapper(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_reduce, patience=args.lr_patience)
   
    # training and inference
    print("start training")
    neutral_vec = torch.from_numpy(args.neutral_vec).to(device)
    train_losses = []
    val_losses = []
    for epoch in range(args.epochs):
        train_loss = train(args, train_data_loader, model, optimizer, neutral_vec)
        print(f"EPOCH[{epoch}] | Train Loss : {train_loss:.3f}")
        val_loss = validate(args, val_data_loader, model, neutral_vec)
        print(f"EPOCH[{epoch}] | Val Loss : {val_loss:.3f}")
        scheduler.step(val_loss)
        print("LR:", optimizer.param_groups[0]['lr'])

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    torch.save({"model_state_dict": model.state_dict()}, f"{args.output}/last_checkpoint.pt")
        
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

    

        
