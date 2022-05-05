import torch
import os
import sys
import numpy as np
from glob import glob
from tqdm import tqdm

class Audio2FrameDataset(object):
    def __init__(self, args, path, split, load_img=False, split_ratio=0.9, input_dim=512, output_dim=512*18, sample="dense"):
        self.args = args
        self.split = split
        self.path = path
        self.seq_len = args.seq_len
        self.split_ratio = split_ratio
        self.sample = sample  # sample strategy: [dense, sparse]
        self.use_pose = args.use_pose == 1
        self.latent_type = args.latent_type

        self.input_dim=input_dim
        self.output_dim=output_dim
        
        # get data folders
        if split in ["train", "val"]:
            self.data_folders = glob(f"{path}/{split}/*")
        else:
            assert split == "test"
            # NB(demi): only support 1 single test video folder
            self.data_folders = [path]
        

        self.counts = []
        self.audio_vecs_list = []
        self.latent_vecs_list = []
        self.pose_vecs_list = []
        for folder in tqdm(self.data_folders, desc=f"loading {split} dataset vectors"):
            self.counts.append(self.get_interval_count(folder))
            self.audio_vecs_list.append(np.load(f"{folder}/wav2lip.npy"))
            
            if self.latent_type == "w+":
                self.latent_vecs_list.append(np.load(f"{folder}/frame.npy").reshape(-1,18*512))  # NB(demi): even test needs to have frame.npy for now
            elif self.latent_type == "stylespace":
                self.latent_vecs_list.append(np.load(f"{folder}/frame_stylespace.npy").reshape(-1,9088)) 
            else:
                raise NotImplementedError
            if self.use_pose:
                self.pose_vecs_list.append(np.load(f"{folder}/openface.npy")[:, 2:8].astype(np.float32))
            else:
                self.pose_vecs_list.append(np.zeros((len(self.audio_vecs_list[-1]), 6), dtype=np.float32))
        assert len(self.counts) <= 1000, "currently iterate over all folders -- cannot support many folders"
        
        self.data_len = np.sum(np.array(self.counts))
        self.mode = args.mode
        assert load_img is False, "loading image not supported yet"
    

    def get_interval_count(self, folder):
        audio_vec = np.load(f"{folder}/wav2lip.npy")
        folder_len = len(audio_vec)
        
        if self.sample == "dense":
            return folder_len
        else:
            assert self.sample == "sparse"
            if folder_len % self.seq_len == 0:
                return folder_len // self.seq_len
            else:
                return folder_len // self.seq_len + 1

    def get_interval_i(self, folder_id, idx):
        folder = self.data_folders[folder_id]
        audio_vecs = self.audio_vecs_list[folder_id]
        latent_vecs = self.latent_vecs_list[folder_id]
        pose_vecs = self.pose_vecs_list[folder_id]
        folder_len = len(audio_vecs)

        if self.sample == "sparse":
            idx = idx * self.seq_len

        r_idx = min(folder_len, idx + self.seq_len)
        idxs = np.array(range(idx, r_idx))
        src = audio_vecs[idx: r_idx]

        if self.use_pose:
            src = np.concatenate([src, pose_vecs[idx:r_idx]], axis=1)
            assert src.shape == (r_idx-idx, len(audio_vecs[0])+len(pose_vecs[0]))

        if idx == 0:
            prev_tgt = latent_vecs[0:r_idx-1]
            prev_tgt = np.concatenate([self.args.neutral_vec.reshape(1,-1), prev_tgt], axis=0)
        else:
            prev_tgt = latent_vecs[idx-1:r_idx-1]  
        tgt = latent_vecs[idx: r_idx]
        
        # predict minus neutral
        prev_tgt = prev_tgt - self.args.neutral_vec.reshape(1,-1)
        tgt = tgt - self.args.neutral_vec.reshape(1, -1)

        assert prev_tgt.shape == tgt.shape
        assert src.shape[0] == tgt.shape[0]
        src_mask = np.ones(src.shape[0])
        tgt_mask = np.ones(src.shape[0])

        if idx + self.seq_len > folder_len:  
            assert self.seq_len != 1
            idxs = np.arange(idx, idx+self.seq_len)
            new_len = idx + self.seq_len - folder_len
            src = np.concatenate([src, np.zeros((new_len,self.input_dim),dtype=src.dtype)],axis=0)
            assert src.shape == (self.seq_len, self.input_dim)
            prev_tgt = np.concatenate([prev_tgt, np.zeros((new_len, self.output_dim),dtype=prev_tgt.dtype)],axis=0)
            assert prev_tgt.shape == (self.seq_len, self.output_dim)
            tgt = np.concatenate([tgt, np.zeros((new_len,self.output_dim),dtype=tgt.dtype)],axis=0)
            assert tgt.shape == (self.seq_len, self.output_dim)
            
            src_mask = np.concatenate([src_mask, np.zeros(new_len,dtype=src_mask.dtype)],axis=0)
            tgt_mask = np.concatenate([tgt_mask, np.zeros(new_len, dtype=tgt_mask.dtype)],axis=0)

        src_mask = src_mask.reshape(1, self.seq_len)
        tgt_mask = tgt_mask.reshape(1, self.seq_len).repeat(self.seq_len, axis=0)
        
        
        assert src.shape[0]  == self.seq_len
        assert tgt.shape[0] == self.seq_len
        tgt_mask = tgt_mask * (1-np.triu(np.ones((self.seq_len,self.seq_len)),k=1))

        # NB(demi): if PCA, need to convert tgt, prev_tgt
        if self.args.pca is not None:
            tgt = self.args.pca.transform(tgt)[:, self.args.pca_dims]
            prev_tgt = self.args.pca.transform(prev_tgt)[:, self.args.pca_dims]
                
        return idxs, src, prev_tgt, tgt, src_mask, tgt_mask

    def get_neutral(self):
        return self.args.neutral_vec

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        
        for i in range(len(self.counts)):
            if idx < self.counts[i]:
                return self.get_interval_i(i, idx)
            else:
                idx = idx - self.counts[i]

        assert False  # must found idx
                

