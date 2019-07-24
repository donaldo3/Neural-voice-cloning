'''
Train speaker encoder with speaker_embeddings from seq2seq TTS as ground truth.

usage: train_speaker_encoder.py [options]

options:
    --vctk-root=<dir>               Directory for vctk root
    --data-root=<dir>               Directory contains preprocessed features.
    --cmp-root=<dir>                Directory contains world cmp features.
    --checkpoint-dir=<dir>          Directory where to save model checkpoints [default: checkpoints].
    --hparams=<parmas>              Hyper parameters [default: ].
    --preset=<json>                 Path of preset parameters (json).
    --checkpoint=<path>             Restore model from checkpoint path if given.
    --checkpoint-seq2seq=<path>     Restore seq2seq model from checkpoint path.
    --checkpoint-postnet=<path>     Restore postnet model from checkpoint path.
    --checkpoint-multispeaker-tts=<path>  Restore speaker embedding of mutlispeaker TTS from checkpoint path
    --train-seq2seq-only            Train only seq2seq model.
    --train-postnet-only            Train only postnet model.
    --restore-parts=<path>          Restore part of the model.
    --log-event-path=<name>         Log event path.
    --reset-optimizer               Reset optimizer.
    --load-embedding=<path>         Load embedding from checkpoint.
    --speaker-id=<N>                Use specific speaker of data in case for multi-speaker datasets.
    -h, --help                      Show this help message and exit

'''
from train import collate_fn as collate_fn_sub
from train import PartialyRandomizedSimilarTimeLengthSampler
from docopt import docopt
from deepvoice3_pytorch.modules import Embedding
from nnmnkwii.datasets import vctk
from torch.utils.data import Dataset
import sys
import gc
import platform
from os.path import dirname, join
from tqdm import tqdm, trange
from datetime import datetime
from deepvoice3_pytorch import frontend, builder
import audio
import lrschedule
from train import MelSpecDataSource
import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from torch.utils.data.sampler import Sampler
import numpy as np
from numba import jit

from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from os.path import join, expanduser
import random

import librosa.display
from matplotlib import pyplot as plt
import sys
import os
from tensorboardX import SummaryWriter
from matplotlib import cm
from warnings import warn
from hparams import hparams, hparams_debug_string

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False

def _pad(seq, max_len, constant_values=0):
    return np.pad(seq, ((0, max_len - len(seq)), (0, 0)),
                  mode='constant', constant_values=constant_values)

def train(device, model, data_loader, optimizer, writer,
          init_lr = 0.002,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,
          clip_thresh=1.0):
    current_lr = init_lr
    MSELoss = nn.MSELoss()

    global globa_step, global_epoch
    while global_epoch < nepochs:
        running_loss = 0.
        for step, (mel, speaker_ids) in tqdm(enumerate(data_loader)):
            model.train()
            ismultispeaker = speaker_ids is not None

            # Learning rate schedule
            if hparams.lr_schedule is not None:
                lr_schedule_f = getattr(lrschedule, hparams.lr_schedule)
                current_lr = lr_schedule_f(init_lr, global_step, **hparams.lr_schedule_kwargs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            optimizer.zero_grad()

            if downsample_step > 1:
                mel = mel[:, 0::downsample_step, :].contiguous()

            # L2 Loss with embeddings from generative model
            loss = MSELoss(pred_embedding, tts_embedding)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

'''
SpeakerDataSource has MelSpecDataSource of every VCTK speaker except for the speakers on black list
'''
class SpeakerDataSet(Dataset):

    '''
    Create list or dictionary of MelSpecDataSource
    '''
    def __init__(self, vctk_root, data_root):
        self.data_root = data_root
        self.train_speaker_id_list = []
        self.mel_spec_datasource_list = []

        speakers = vctk.available_speakers
        td = vctk.TranscriptionDataSource(vctk_root, speakers=speakers)
        transcriptions = td.collect_files()
        speaker_ids = td.labels # All speakers except for p315
        self.speaker_to_speaker_id = td.labelmap

        # Create lists of training speaker_ids
        black_list_speaker = hparams.not_for_train_speaker.split(", ")
        black_list_speaker_ids = [self.speaker_to_speaker_id[i] for i in black_list_speaker]
        speaker_ids = np.unique(speaker_ids).tolist()
        for i in black_list_speaker_ids:
            speaker_ids.remove(i)
        self.train_speaker_id_list = speaker_ids

        # Assuming self.speaker_list contains speaker_ids of training speakers
        for spkr_id in self.train_speaker_id_list:
            Mel = FileSourceDataset(MelSpecDataSource(data_root, speaker_id=spkr_id))
            self.mel_spec_datasource_list.append(Mel)

    '''
    Return MelSpecDataSource of desired speaker idx
    '''
    def __getitem__(self, idx):
        return self.mel_spec_datasource_list[idx]

    def __len__(self):
        return len(self.mel_spec_datasource_list)

# Return mel spectrogram (b, T, F) from x[0]
def collate_fn_sub(batch):
    # Try not to downsample or anything at first implementation. Refrain from using tricks
    # Speaker encoder is non-causal filter. No autoregressive decoder.
    # Thus no position, no pad at the beginning, no alignment
    # Pad to match the max target length before stacking samples into a batch ndarray
    target_lengths = [len(x) for x in batch] # Frame length
    max_target_len = max(target_lengths)
    b = np.array([_pad(x, max_target_len) for x in batch])
    mel_batch = torch.FloatTensor(b)
    return mel_batch

'''
Return 4D mel sectrogram and speaker embedding from TTS 
'''
def collate_fn(batch):
    # Randomly sample N_sample audios from x[0]
    # Assume single MelDataSource is given and try to sample.
    # TODO: frame length padding
    mels = []
    for x in batch:
        dataset = x[0] # MelSpecDataSource

        # Call MelSpecDataSource.__getitem__() and randomly sample batch size of hparams.cloning_sample_size
        # without creating DataLoader object
        # Or what if DataLoader does not create worker thread for num_workders=0?
        frame_lengths = dataset.file_data_source.frame_lengths
        sampler = PartialyRandomizedSimilarTimeLengthSampler(frame_lengths, batch_size=hparams.cloning_sample_size)

        data_loader = data_utils.DataLoader(
            dataset, batch_size=hparams.cloning_sample_size,
            num_workers=0, sampler=sampler,
            collate_fn=collate_fn_sub, pin_memory=False
        )

        # Create list
        cloning_samples = []
        for i, audio in enumerate(data_loader):
            audio = audio.numpy()
            break

        mels.append(audio)

    # Match the length among speakers
    target_lengths = [len(x[0]) for x in mels]
    max_target_len = max(target_lengths)
    padded_mels = []
    # Match the max length for each speaker batch
    for i in range(len(mels)):
        padded_mels_sub = []
        for j in range(len(mels[i])):
            sample = _pad(mels[i][j], max_target_len)
            padded_mels_sub.append(sample)
        padded_mels_sub = np.stack(padded_mels_sub, axis=0)
        padded_mels.append(padded_mels_sub)
    padded_mels = np.stack(padded_mels, axis=0)
    padded_mels = torch.FloatTensor(padded_mels)
    # TODO: 3D or 2D for speaker embedding?
    speaker_embeds = torch.stack([x[1] for x in batch])
    speaker_embeds = speaker_embeds.detach()
    return padded_mels, speaker_embeds

class PyTorchDataset(Dataset):
    def __init__(self, speaker_data_source, tts_checkpoint_path):
        self.speaker_data_source = speaker_data_source
        self.speaker_embedding = Embedding(num_embeddings=hparams.n_speakers,
        embedding_dim=hparams.speaker_embed_dim,padding_idx=None,
                std=hparams.speaker_embedding_weight_std)
        state = _load(tts_checkpoint_path)["state_dict"]
        key = "embed_speakers.weight"
        pretrained_embedding = state[key]
        self.speaker_embedding.weight.data.copy_(pretrained_embedding)
        # Expand speaker_embedding into 3D and save as list of tensors.

    '''
    Return FileSourceDatas of n_batch speakers and speaker embedding from TTS(2D: batch, d_emb)
    '''
    def __getitem__(self, idx):
        idx_tensor = torch.tensor([idx], dtype=torch.long)
        emb = self.speaker_embedding(idx_tensor)
        return self.speaker_data_source[idx], emb

    def __len__(self):
        return len(self.speaker_data_source.mel_spec_datasource_list)

if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    log_event_path = args["--log-event-path"]
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path = args["--checkpoint"]
    checkpoint_multispeaker_tts = args["--checkpoint-multispeaker-tts"]
    load_embedding = args["--load-embedding"]
    preset = args["--preset"]
    vctk_root = args["--vctk-root"]
    data_root = args["--data-root"]
    preset = args["--preset"]
    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    # What if cmp is also used as feature to extract speaker embedding? I think i'd like that

    print("Training speaker encoder model")

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])

    os.makedirs(checkpoint_dir, exist_ok=True)

    speaker_dataset = SpeakerDataSet(vctk_root, data_root)
    dataset = PyTorchDataset(speaker_dataset, checkpoint_multispeaker_tts)
    data_loader = data_utils.DataLoader(
        dataset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )
    device = torch.device("cuda" if use_cuda else "cpu")
    model = SpeakerEncoder(hparams.encoder_channels, hparams.kernel_size)
    optimizer = optim.Adam(model.get_trainable_parameters(),
                           lr=hparams.initial_learning_rate, betas=(
        hparams.adam_beta1, hparams.adam_beta2),
        eps=hparams.adam_eps, weight_decay=hparams.weight_decay,
        amsgrad=hparams.amsgrad)
    if log_event_path is None:
        if platform.system() == "Windows":
            log_event_path = "log/run-test" + \
                str(datetime.now()).replace(" ", "_").replace(":", "_")
        else:
            log_event_path = "log/run-test" + str(datetime.now()).replace(" ", "_")
    writer = SummaryWriter(log_dir=log_event_path)
    # Train
    try:
        train(device, model, data_loader, optimizer, writer,
              init_lr=hparams.initial_learning_rate
              )
    except:
        pass
