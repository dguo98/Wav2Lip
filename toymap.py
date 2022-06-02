import os
import sys
import argparse
import matplotlib as plt
from glob import glob
from tqdm import tqdm
import numpy as np
from IPython import embed

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
        
        # todo(demi): variable layers
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
    
    def forward(self, audio_vec):
        return self.layers(audio_vec)
        

class Audio2FrameDataset(object):
    def __init__(self, args, path, split, load_img=False, cache=True):
        self.args = args
        self.split = split
        self.path = path
        self.audio_files = sorted(glob(f"{path}/wav2lip_latents/wav2lip_audio*"))
        self.latent_files = sorted(glob(f"{path}/frame_latents/latents_frame_[0123456789]*"))

        assert len(self.audio_files) == len(self.latent_files) or split == "test"

        assert load_img is False, "loading image not supported yet"
        split_ratio = 0.9
        split_num = int(split_ratio * len(self.audio_files))
        if split == "train":
            self.audio_files = self.audio_files[:split_num]
            self.latent_files = self.latent_files[:split_num]
        elif split == "val":
            self.audio_files = self.audio_files[split_num:]
            self.latent_files = self.latent_files[split_num:]
        else:
            assert split == "test"
            pass

        self.data_len = len(self.audio_files)

        self.cache = cache

        start_id = 0
        if split == "val":
            start_id = split_num
        if self.cache:
            self.audio_vecs = []
            for i, f in tqdm(enumerate(self.audio_files), desc="load audio vecs"):
                v = torch.load(f).reshape(-1).detach().cpu().numpy()
                self.audio_vecs.append(v)

                assert f"{i+start_id:04d}" in f
            self.audio_vecs = np.stack(self.audio_vecs, axis=0)
            assert self.audio_vecs.shape == (self.data_len, 512)

            if self.split is not "test":
                self.latent_vecs = []
                for i, f in tqdm(enumerate(self.latent_files), desc="load latent vecs"):
                    v = np.load(f).reshape(-1)
                    self.latent_vecs.append(v)

                    assert f"{i+start_id:04d}" in f
                self.latent_vecs = np.stack(self.latent_vecs, axis=0)
                assert self.latent_vecs.shape == (self.data_len, 512*18)
            else:
                self.latent_vecs = np.zeros((self.data_len, 512*18))

        assert self.cache is True
    
    def get_neutral(self):
        neutral_vec = np.load(f"{self.path}/frame_latents/neutral.npy")
        return neutral_vec

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return np.array([idx]), self.audio_vecs[idx], self.latent_vecs[idx]


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
        assert predict_vec.size() == (bsz, 512*18)
        predict_vec = predict_vec + neutral_vec  # only predict difference


        # MSE Loss for now
        loss = torch.nn.MSELoss(reduction="mean")(predict_vec, latent_vec) 
        loss = loss * 512 * 18  # only average across batch, not items
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
        assert predict_vec.size() == (bsz, 512*18)
        predict_vec = predict_vec + neutral_vec  # only predict difference


        # MSE Loss for now
        loss = torch.nn.MSELoss(reduction="mean")(predict_vec, latent_vec) 
        loss = loss * 512 * 18  # only average across batch, not items
        losses.append(loss.item())

    return np.mean(losses)
        

def inference(args, data_loader, model, infer_dir, neutral_vec):
    model.eval()

    total = len(data_loader)
    neutral_vec = neutral_vec.reshape(1, -1)

    losses = []
    for n_iter, (ids, audio_vec, latent_vec) in tqdm(enumerate(data_loader),total=total, desc="inference"):
        bsz = audio_vec.size(0)

        audio_vec = audio_vec.cuda()
        latent_vec = latent_vec.cuda()
        ids = ids.cuda().reshape(bsz)

        predict_vec = model(audio_vec)
        assert predict_vec.size() == (bsz, 512*18)
        predict_vec = predict_vec + neutral_vec  # only predict difference

        for i in range(bsz):
            idx = ids[i].item()
            torch.save(predict_vec[i], f"{infer_dir}/predict_{idx:04d}.pt")

    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data_mlpmap/synthesia-obama-trim")
    parser.add_argument("--test_path", type=str, default="eval_mlpmap/eval-1")
    parser.add_argument("--output", type=str, default="output/debug")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)

    args = parser.parse_args()
    
    device = torch.device("cuda")
    model = Mapper(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    
    # datasets
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
    
    neutral_vec = torch.from_numpy(train_dataset.get_neutral()).to(device)
    # training and inference
    for epoch in range(args.epochs):
        train_loss = train(args, train_data_loader, model, optimizer, neutral_vec)
        print(f"EPOCH[{epoch}] | Train Loss : {train_loss:.3f}")
        val_loss = validate(args, val_data_loader, model, neutral_vec)
        print(f"EPOCH[{epoch}] | Val Loss : {val_loss:.3f}")
        scheduler.step(val_loss)

    torch.save({"model_state_dict": model.state_dict()}, f"{args.output}/last_checkpoint.pt")
        
    infer_dir = f"{args.output}/inference"
    if not os.path.exists(infer_dir):
        os.makedirs(infer_dir)
    os.system(f"cp {args.test_path}/audio.wav {infer_dir}/")
    neutral_vec = torch.from_numpy(test_dataset.get_neutral()).to(device)
    inference(args, test_data_loader, model, infer_dir, neutral_vec)


        
