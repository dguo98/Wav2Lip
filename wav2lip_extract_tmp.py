from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
from models import Wav2Lip as Wav2Lip
import audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob
from IPython import embed

import os, random, cv2, argparse
from hparams import hparams, get_image_list

parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model without the visual quality discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True, type=str)
parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default=None, type=str)

args = parser.parse_args()


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16

class Dataset(object):
    def __init__(self, split):
        self.all_videos = [split]
        self.wavpath = f"{split}/audio.wav"
        self.frames = sorted(glob(f"{split}/frames/frame_*[0123456789].jpg"))

        # HACK(demi): for debugging
        #self.frames = self.frames[:42]
        #print("self.frames=",self.frames)

        print("split=",split)
        #print("self.frames=",self.frames)
        #print("frames=",self.frames)
        self.num_frames = len(self.frames)
        self.data_len = self.num_frames // syncnet_T
        if self.data_len * syncnet_T < self.num_frames:
            self.data_len += 1

        print("numframes=", self.num_frames, " datalen=", self.data_len)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0].replace("frame_",""))

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, f'frame_{frame_id:06d}.jpg')
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame) # 0-indexing ---> 1-indexing
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
        end_idx = start_idx + syncnet_mel_step_size
        
        # HACK(demi): pad
        if end_idx > len(spec):
            start_idx = len(spec) - syncnet_mel_step_size
            end_idx = len(spec) 

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        # if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, max(0, i - 2))  # HACK(demi): pad 

            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        #print("in loading data idx=", idx)

        img_names = self.frames

        img_name = self.frames[idx * syncnet_T]
        #print("idx=", idx,"img_name=", img_name)


        if idx*syncnet_T + syncnet_T -1 >= self.num_frames:
            # hack for last window
            img_name = self.frames[self.num_frames-syncnet_T]
            #print("idx=", idx, "new img_name=", img_name)
        #print("get item idx=", idx, " img name=", img_name)

        #if len(img_names) <= 3 * syncnet_T:
        #    continue
        

        wrong_img_name = random.choice(img_names[:-syncnet_T])
        while wrong_img_name == img_name:
            wrong_img_name = random.choice(img_names)

        window_fnames = self.get_window(img_name)  # get filenames within the window for lipsync expert

        window_ids = []
        #print("window_fnames=", window_fnames)
        for fname in window_fnames:
            window_ids.append(self.get_frame_id(fname))

        wrong_window_fnames = self.get_window(wrong_img_name)
        #print("window fnames=",window_fnames, " wrong window fnames=", wrong_window_fnames)
        if window_fnames is None or wrong_window_fnames is None:
            print("window_fnames or wrong_window_fnames is none")
            assert False
        
        window = self.read_window(window_fnames)  # read images from filenames
        assert window is not None

        wrong_window = self.read_window(wrong_window_fnames)
        assert wrong_window is not None
        
        #print("finish reading window")

        wavpath = self.wavpath
        wav = audio.load_wav(wavpath, hparams.sample_rate)
        orig_mel = audio.melspectrogram(wav).T
        
        #print("before crop audio window")
        mel = self.crop_audio_window(orig_mel.copy(), img_name)
        #print("cropped mel=", mel)
        #print("got mel")
        
        #if (mel.shape[0] != syncnet_mel_step_size):
        #    print("bad line 165")
        #    assert False
        
        #print("get segmented mels")
        indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
        if indiv_mels is None:
            print("bad indiv mels")
        assert indiv_mels is not None
        #print("got segmetn4ed mels")
        
        #print("start prepare window")
        window = self.prepare_window(window)
        y = window.copy()
        window[:, :, window.shape[2]//2:] = 0.

        wrong_window = self.prepare_window(wrong_window)
        x = np.concatenate([window, wrong_window], axis=0)
        #print("after prepare window")

        x = torch.FloatTensor(x)
        mel = torch.FloatTensor(mel.T).unsqueeze(0)
        indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
        y = torch.FloatTensor(y)

        y_ids = torch.from_numpy(np.array(window_ids))
        #print("idx=",idx,"y_ids=",y_ids)

        #print("data iterate: x, mel, indiv_mels, y, y_ids")
        return x, indiv_mels, mel, y, y_ids, torch.Tensor([idx])

def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

device = torch.device("cuda" if use_cuda else "cpu")
syncnet = SyncNet().to(device)
for p in syncnet.parameters():
    p.requires_grad = False

recon_loss = nn.L1Loss()
def get_sync_loss(mel, g):
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)

def train(video_folder, device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):

    global global_step, global_epoch
    resumed_step = global_step
    
    global_epoch = 0
    indiv_mels_list = []
    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_sync_loss, running_l1_loss = 0., 0.
        #prog_bar = tqdm(enumerate(train_data_loader), desc="extracting")
        total = len(train_data_loader)
        print("total=",total)
        for step, (x, indiv_mels, mel, gt, gt_ids, idxs) in tqdm(enumerate(train_data_loader), desc="extracting", total=total):
            #print("load first batch")
            model.train()
            optimizer.zero_grad()

            # Move data to CUDA device
            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)
            
            #print("in training loop: x, mel, indiv_mels, gt, gt_ids, idxs")
            #audio, context, final_ids = model.extract(indiv_mels, x, gt_ids)
            
            indiv_mels = indiv_mels.reshape(-1, 80,16)
            assert indiv_mels.shape[0] == len(gt_ids.reshape(-1))

            for i in range(len(indiv_mels)):
                #torch.save(audio[i].reshape(-1), f"{video_folder}/wav2lip_latents/wav2lip_audio_{idx:06d}.pt")
                #torch.save(context[i].reshape(-1), f"{video_folder}/wav2lip_latents/wav2lip_context_{idx:06d}.pt")
                indiv_mels_list.append(indiv_mels[i].reshape(80,16).detach().cpu().numpy())


        global_epoch += 1          
        indiv_mels_list = np.stack(indiv_mels_list, axis=0)
        print("indiv_mels_list.shape=",indiv_mels_list.shape)

        np.save(f"{video_folder}/wav2lip_mels.npy", indiv_mels_list)
        break

       
def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model

if __name__ == "__main__":
    video_folders = [args.data_root]

    print("video_folders=",video_folders)
    # Model
    model = Wav2Lip().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    
    device = torch.device("cuda" if use_cuda else "cpu")
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate)
    load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)

    for video_folder in video_folders:
        if ".mp4" in video_folder:
            continue
        #os.makedirs(f"{video_folder}/wav2lip_latents")
        train_dataset = Dataset(video_folder)
        """ 
        for i in tqdm(range(400), desc="go through dataset"):
            try:
                data = train_dataset[i]
            except:
                print(f"train dataset[{i}] failed")
        """

        train_data_loader = data_utils.DataLoader(
            train_dataset, batch_size=hparams.batch_size, shuffle=False,
            num_workers=hparams.num_workers, drop_last=False)

            
        train(video_folder, device, model, train_data_loader, None, optimizer,
                  checkpoint_dir=None,
                  checkpoint_interval=hparams.checkpoint_interval,
                  nepochs=1)
