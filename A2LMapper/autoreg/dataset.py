import torch
import os
import sys
import numpy as np

class Audio2FrameDataset(object):
    def __init__(self, args, path, split, load_img=False, split_ratio=0.9, input_dim=512, output_dim=512*18):
        self.args = args
        self.split = split
        self.path = path
        self.seq_len = args.seq_len
        self.split_ratio = split_ratio
        self.model = args.model

        self.input_dim=input_dim
        self.output_dim=output_dim

        self.audio_vecs = np.load(f"{path}/wav2lip.npy")
        
        if os.path.exists(f"{path}/frame.npy"):
            self.latent_vecs = np.load(f"{path}/frame.npy").reshape(-1, 18*512)
        else:
            self.latent_vecs = np.zeros((len(self.audio_vecs), 18*512))
        assert len(self.audio_vecs) == len(self.latent_vecs)
        

        self.mode = args.mode
        assert load_img is False, "loading image not supported yet"

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

    
    def get_neutral(self):
        return self.args.neutral_vec

    def __len__(self):
        if self.split == "train":
            if self.model == "conv":
                return self.data_len - self.seq_len + 1
            else:
                return self.data_len
        else:
            if self.data_len % self.seq_len == 0 or self.model == "conv":
                # NB(demi): for now, even for test, it will ignore the rest 
                return self.data_len // self.seq_len
            else:
                return self.data_len // self.seq_len + 1

    def __getitem__(self, idx):
        """
        if self.seq_len == 1:  # hack: debug
            idxs = np.arange(idx,idx+1)
            src = self.audio_vecs[idx:idx+1]
            tgt = self.latent_vecs[idx:idx+1]
            prev_tgt = tgt
            src_mask = np.ones((1,1))
            tgt_mask = np.ones((1,1))
            return idxs,src,prev_tgt,tgt,src_mask,tgt_mask
        """
            
        if self.split != "train":
            idx = idx * self.seq_len

        r_idx = min(self.data_len, idx + self.seq_len)

        idxs = np.array(range(idx, r_idx))
        src = self.audio_vecs[idx: r_idx]
        if idx == 0:
            prev_tgt = self.latent_vecs[0:r_idx-1]
            prev_tgt = np.concatenate([self.args.neutral_vec.reshape(1,-1), prev_tgt], axis=0)
        else:
            prev_tgt = self.latent_vecs[idx-1:r_idx-1]  
        tgt = self.latent_vecs[idx: r_idx]
        
        # predict minus neutral
        prev_tgt = prev_tgt - self.args.neutral_vec.reshape(1,-1)
        tgt = tgt - self.args.neutral_vec.reshape(1, -1)

        assert prev_tgt.shape == tgt.shape
        assert src.shape[0] == tgt.shape[0]
        src_mask = np.ones(src.shape[0])
        tgt_mask = np.ones(src.shape[0])

        if idx + self.seq_len > self.data_len:  
            assert self.seq_len != 1
            idxs = np.arange(idx, idx+self.seq_len)
            new_len = idx + self.seq_len - self.data_len
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
        
        return idxs, src, prev_tgt, tgt, src_mask, tgt_mask


            
