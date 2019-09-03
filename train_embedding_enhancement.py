'''
Train speaker encoder with speaker_embeddings from seq2seq TTS as ground truth.

usage: train_speaker_encoder.py [options]

options:
    --vctk-root=<dir>               Directory for vctk root
    --clean-data-root=<dir>         Directory contains preprocessed features.
    --noisy-data-root=<dir>         Directory contains preprocessed features.
    --cmp-root=<dir>                Directory contains world cmp features.
    --checkpoint-dir=<dir>          Directory where to save model checkpoints [default: checkpoints].
    --hparams=<parmas>              Hyper parameters [default: ].
    --preset=<json>                 Path of preset parameters (json).
    --checkpoint-speaker-encoder=<path>    Restore speaker encoder model from checkpoint path if given.
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
'''
 TODO: clean-data-root and noisy-data-root now points to preprocessed mel spectrogram directory.
 It will be comfier to make it point to wav directory and include feature extraction to Dataset.__init__() 
'''
from embedding_enhancement import EmbeddingEnhancement
from train import collate_fn as collate_fn_sub, load_checkpoint
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
from speaker_encoder.speaker_encoder3 import SpeakerEncoder
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

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    suffix = "_speaker_encoder"
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}{}.pth".format(global_step, suffix)
    )
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def train(device, speaker_encoder, model, data_loader, optimizer, writer,
          init_lr = 0.002,
          checkpoint_dir=None, checkpoint_interval=10000, nepochs=None,
          clip_thresh=1.0):
    current_lr = init_lr
    MSELoss = nn.MSELoss()

    global global_step, global_epoch
    while global_epoch < nepochs:
        running_loss = 0.

        # (b, N, T, F)
        for step, (padded_clean_mels, padded_noisy_mels) in tqdm(enumerate(data_loader)):
            # to device
            speaker_encoder.cuda()
            speaker_encoder.eval()
            model.cuda()
            model.train()
            clean_mels = padded_clean_mels.to(device)
            noisy_mels = padded_noisy_mels.to(device)

            # Generate clean embeddings (b, d_s)
            size = clean_mels.size()
            clean_mels = clean_mels.view(size[0] * size[1], size[2], size[3])
            clean_embeddings = speaker_encoder(clean_mels)
            clean_embeddings = clean_embeddings.detach()


            # Generate noisy embeddings (b, d_s)
            noisy_mels = noisy_mels.view(size[0] * size[1], size[2], size[3])
            noisy_embeddings = speaker_encoder(noisy_mels)
            noisy_embeddings = noisy_embeddings.detach()

            # Learning rate schedule
            if hparams.lr_schedule is not None:
                lr_schedule_f = getattr(lrschedule, hparams.lr_schedule)
                current_lr = lr_schedule_f(init_lr, global_step, **hparams.lr_schedule_kwargs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            optimizer.zero_grad()

            # Run embedding enhancement model
            pred_embeddings = model(noisy_embeddings)

            # L2 Loss with embeddings from generative model
            loss = MSELoss(pred_embeddings, clean_embeddings)

            # Update
            loss.backward()
            if clip_thresh > 0:
                grad_norm = torch. nn.utils.clip_grad_norm_(
                    speaker_encoder.parameters(), clip_thresh
                )
                optimizer.step()

            # Logs
            writer.add_scalar("loss", float(loss.item()), global_step)

            if clip_thresh > 0:
                writer.add_scalar("gradient norm", grad_norm, global_step)
            writer.add_scalar("learning rate", current_lr, global_step)

            global_step += 1
            running_loss += loss.item()

            # Save checkpoint
            # TODO: check checkpoint saving indentation
            if global_step > 0 and global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch
                )


        averaged_loss = running_loss / (len(data_loader))
        writer.add_scalar("loss (per epoch)", averaged_loss, global_epoch)
        print("Loss: {}".format(running_loss/(len(data_loader))))
        global_epoch += 1

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

class _DoubleNPYDataSource(FileDataSource):
    def __init__(self, clean_data_root, noisy_data_root, speaker_id=None):
        self.clean_data_root = clean_data_root
        self.noisy_data_root = noisy_data_root
        self.frame_lengths = []
        self.speaker_id = speaker_id

    def collect_files(self):
        clean_meta = join(self.clean_data_root, "train.txt")
        with open(clean_meta, "rb") as f:
            lines = f.readlines()
        l = lines[0].decode("utf-8").split("|")
        assert len(l) == 4 or len(l) == 5
        multi_speaker = len(l) == 5
        self.frame_lengths = list(
            map(lambda l: int(l.decode("utf-8").split("|")[2]), lines))

        paths = list(map(lambda l:l.decode("utf-8").split("|")[1], lines))
        paths = list(map(lambda f:join(self.clean_data_root, f), paths))

        if multi_speaker and self.speaker_id is not None:
            speaker_ids = list(map(lambda l: int(l.decode("utf-8").split("|")[-1]), lines))
            # Filter by speaker_id
            # using multi-speaker dataset as a single speaker dataset
            indices = np.array(speaker_ids) == self.speaker_id
            paths = list(np.array(paths)[indices])
            self.frame_lengths = list(np.array(self.frame_lengths)[indices])
            # aha, need to cast numpy.int64 to int
            self.frame_lengths = list(map(int, self.frame_lengths))

        return paths

    def collect_features(self, path):
        base = os.path.basename(path)
        noisy_path = os.path.join(self.noisy_data_root, base)
        return np.load(path), np.load(noisy_path)

class DoubleMelSpecDataSource(_DoubleNPYDataSource):
    def __init__(self, clean_data_root, noisy_data_root, speaker_id=None):
        super(DoubleMelSpecDataSource, self).__init__(clean_data_root, noisy_data_root, speaker_id=speaker_id)

'''
SpeakerDataSource has MelSpecDataSource of every VCTK speaker except for the speakers on black list
'''
class SpeakerDataSet(Dataset):

    '''
    Create list or dictionary of MelSpecDataSource
    '''
    def __init__(self, vctk_root, clean_data_root, noisy_data_root):
        self.clean_data_root = clean_data_root
        self.noisy_data_root = noisy_data_root
        self.train_speaker_id_list = []
        self.mel_spec_datasource_list = []

        speakers = vctk.available_speakers
        td = vctk.TranscriptionDataSource(vctk_root, speakers=speakers)
        transcriptions = td.collect_files()
        speaker_ids = td.labels # All speakers except for p315
        self.speaker_to_speaker_id = td.labelmap

        # Create lists of training speaker_ids
        black_list_speaker = hparams.not_for_train_speaker.split(", ")
        self.black_list_speaker_ids = [self.speaker_to_speaker_id[i] for i in black_list_speaker]
        speaker_ids = np.unique(speaker_ids).tolist()
        for i in self.black_list_speaker_ids:
            speaker_ids.remove(i)
        self.train_speaker_id_list = speaker_ids

        # Assuming self.speaker_list contains speaker_ids of training speakers
        for spkr_id in self.train_speaker_id_list:
            Mel = FileSourceDataset(DoubleMelSpecDataSource(clean_data_root, noisy_data_root, speaker_id=spkr_id))
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
    target_lengths = [len(x[0]) for x in batch] # Frame length
    max_target_len = max(target_lengths)
    clean_mel = np.array([_pad(x[0], max_target_len) for x in batch])
    clean_mel_batch = torch.FloatTensor(clean_mel)

    noisy_mel = np.array([_pad(x[1], max_target_len) for x in batch])
    noisy_mel_batch = torch.FloatTensor(noisy_mel)
    return clean_mel_batch, noisy_mel_batch

'''
Return 4D mel sectrogram and speaker embedding from TTS 
'''
def collate_fn(batch):
    # Randomly sample N_sample audios from x[0]
    # Assume single MelDataSource is given and try to sample.
    # TODO: frame length padding
    clean_mels = []
    noisy_mels = []
    for x in batch:
        dataset = x # MelSpecDataSource

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
        for i, audio in enumerate(data_loader):
            clean, noisy = audio
            clean = clean.numpy()
            noisy = noisy.numpy()
            break
        clean_mels.append(clean)
        noisy_mels.append(noisy)



    # Match the length among speakers
    target_lengths = [len(x[0]) for x in clean_mels]
    max_target_len = max(target_lengths)
    padded_clean_mels = []
    padded_noisy_mels = []
    # Match the max length for each speaker batch
    for i in range(len(clean_mels)):
        padded_clean_mels_sub = []
        padded_noisy_mels_sub = []
        for j in range(len(clean_mels[i])):
            sample = _pad(clean_mels[i][j], max_target_len)
            padded_clean_mels_sub.append(sample)
            sample = _pad(noisy_mels[i][j], max_target_len)
            padded_noisy_mels_sub.append(sample)
        padded_clean_mels_sub = np.stack(padded_clean_mels_sub, axis=0)
        padded_clean_mels.append(padded_clean_mels_sub)

        padded_noisy_mels_sub = np.stack(padded_noisy_mels_sub, axis=0)
        padded_noisy_mels.append(padded_noisy_mels_sub)
    padded_clean_mels = np.stack(padded_clean_mels, axis=0)
    padded_clean_mels = torch.FloatTensor(padded_clean_mels)

    padded_noisy_mels = np.stack(padded_noisy_mels, axis=0)
    padded_noisy_mels = torch.FloatTensor(padded_noisy_mels)
    return padded_clean_mels, padded_noisy_mels

class PyTorchDataset(Dataset):
    def __init__(self, speaker_data_source):
        self.speaker_data_source = speaker_data_source




    '''
    Return FileSourceDatas of n_batch speakers and speaker embedding from TTS(2D: batch, d_emb)
    '''
    def __getitem__(self, idx):
        return self.speaker_data_source[idx]

    def __len__(self):
        return len(self.speaker_data_source.mel_spec_datasource_list)

if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    log_event_path = args["--log-event-path"]
    checkpoint_dir = args["--checkpoint-dir"]
    speaker_encoder_checkpoint_path = args["--checkpoint-speaker-encoder"]
    checkpoint_multispeaker_tts = args["--checkpoint-multispeaker-tts"]
    checkpoint = args["--checkpoint"]
    load_embedding = args["--load-embedding"]
    preset = args["--preset"]
    vctk_root = args["--vctk-root"]
    clean_data_root = args["--clean-data-root"]
    noisy_data_root = args["--noisy-data-root"]
    preset = args["--preset"]
    reset_optimizer = args["--reset-optimizer"]

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    # What if cmp is also used as feature to extract speaker embedding? I think i'd like that

    print("Training speaker encoder model")

    os.makedirs(checkpoint_dir, exist_ok=True)

    speaker_dataset = SpeakerDataSet(vctk_root, clean_data_root, noisy_data_root)
    dataset = PyTorchDataset(speaker_dataset)
    data_loader = data_utils.DataLoader(
        speaker_dataset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )
    device = torch.device("cuda" if use_cuda else "cpu")
    speaker_encoder_model = SpeakerEncoder(hparams.num_mels, hparams.encoder_channels, hparams.kernel_size,
                                           hparams.f_mapped, hparams.speaker_embed_dim,
                                           hparams.speaker_encoder_attention_num_heads,
                                           hparams.speaker_encoder_attention_dim,
                                           hparams.dropout,
                                           hparams.batch_size,
                                           hparams.cloning_sample_size)
    model = EmbeddingEnhancement()
    optimizer = optim.Adam(speaker_encoder_model.parameters(),
                           lr=hparams.initial_learning_rate, betas=(
        hparams.adam_beta1, hparams.adam_beta2),
                           eps=hparams.adam_eps, weight_decay=hparams.weight_decay,
                           amsgrad=hparams.amsgrad)

    if device.type != "cpu":
        speaker_encoder_model.cuda()

    if speaker_encoder_checkpoint_path is not None:
        speaker_encoder_model = load_checkpoint(speaker_encoder_checkpoint_path, speaker_encoder_model, optimizer, reset_optimizer)

    if checkpoint is not None:
        model = load_checkpoint(checkpoint, model, optimizer, reset_optimizer)

    if log_event_path is None:
        if platform.system() == "Windows":
            log_event_path = "log/run-test" + \
                str(datetime.now()).replace(" ", "_").replace(":", "_")
        else:
            log_event_path = "log/run-test" + str(datetime.now()).replace(" ", "_")
    writer = SummaryWriter(log_dir=log_event_path)
    # Train
    # TODO: add embedding enhancement model as argument to train()
    try:
        train(device, speaker_encoder_model, model, data_loader, optimizer, writer,
              init_lr=hparams.initial_learning_rate,
              nepochs=hparams.nepochs,
              checkpoint_interval=hparams.speaker_encoder_checkpoint_interval,
              checkpoint_dir=checkpoint_dir
              )
    except KeyboardInterrupt:
        save_checkpoint(speaker_encoder_model, optimizer, global_step, checkpoint_dir, global_epoch)
