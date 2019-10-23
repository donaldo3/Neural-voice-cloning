'''
Train speaker encoder with speaker_embeddings from seq2seq TTS as ground truth.

usage: train_speaker_encoder.py [options]

options:
    --device=<name>                 Which GPU to use
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

def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model
def train(device, model, data_loader, optimizer, writer,
          init_lr = 0.002,
          checkpoint_dir=None, checkpoint_interval=10000, nepochs=None,
          clip_thresh=1.0):
    current_lr = init_lr
    mse_loss = nn.MSELoss()

    global global_step, global_epoch
    while global_epoch < nepochs:
        running_loss = 0.
        for step, (mel, tts_embeddings) in tqdm(enumerate(data_loader)):
            model.train()

            # Learning rate schedule
            if hparams.lr_schedule is not None:
                lr_schedule_f = getattr(lrschedule, hparams.lr_schedule)
                current_lr = lr_schedule_f(init_lr, global_step, **hparams.lr_schedule_kwargs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            optimizer.zero_grad()

            # Downsample degrades temporal resolution
            # if hparams.downsample_step > 1:
            #     mel = mel[:, :, 0::hparams.downsample_step,:].contiguous()

            # Change into 3D
            size = mel.size()
            mel = mel.view(size[0] * size[1], size[2], size[3])


            # Transform data to CUDA device
            mel = mel.to(device)
            tts_embeddings = tts_embeddings.to(device)

            # Apply model
            pred_speaker_embeddings = model(mel)

            # L1 Loss with embeddings from generative model
            loss = mse_loss(pred_speaker_embeddings, tts_embeddings.squeeze())

            # Update
            loss.backward()
            if clip_thresh > 0:
                grad_norm = torch. nn.utils.clip_grad_norm_(
                    model.parameters(), clip_thresh
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
        self.black_list_speaker_ids = []
        if vctk_root is not None:
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
        else:
            self.speaker_to_speaker_id = {}
            meta_file = os.path.join(data_root, 'speaker_id_to_speaker.txt')
            ids = []
            with open(meta_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                id = int(line.split('|')[0])
                speaker = line.split('|')[1]
                ids.append(id)
                self.speaker_to_speaker_id[speaker] = id

        # Assuming self.speaker_list contains speaker_ids of training speakers
        for spkr_id in ids:
            speaker = self.speaker_to_speaker_id
            Mel = FileSourceDataset(MelSpecDataSource(data_root, speaker_id=spkr_id))

            if len(Mel.file_data_source.frame_lengths) < hparams.cloning_sample_size:
                self.black_list_speaker_ids.append(spkr_id)
                continue
            else:
                self.mel_spec_datasource_list.append(Mel)
        self.train_speaker_id_list = [x for x in ids if x not in self.black_list_speaker_ids]
        meta_file2 = os.path.join(data_root, 'speaker_ids_excluded_from_training.txt')
        with open(meta_file2, 'w', encoding='utf-8') as f:
            for x in self.black_list_speaker_ids:
                f.write(str(x)+'\n')


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

        state = _load(tts_checkpoint_path)["state_dict"]
        key = "embed_speakers.weight"
        pretrained_embedding = state[key]

        # Create embedding tensor with black list embeddings removed
        black_list_id = speaker_data_source.black_list_speaker_ids
        dict_embedding = {}
        for i in range(len(pretrained_embedding)):
            dict_embedding[i] = pretrained_embedding[i]
            #pretrained_embedding.tolist()

        # start = black_list_id[0]
        # end = black_list_id[-1] +1
        #
        # del list_embedding[start:end] # TODO: black list must be continuously selected
        #

        for id in black_list_id:
            del dict_embedding[id]
        list_embedding = []
        for key in sorted(dict_embedding):
            list_embedding.append(dict_embedding[key])

        tensor_embedding = torch.stack(list_embedding)
        assert len(tensor_embedding) == len(speaker_data_source.mel_spec_datasource_list)

        # Create speaker_embedding and copy the tensor from above

        self.speaker_embedding = Embedding(num_embeddings=tensor_embedding.size()[0],
                                           embedding_dim=hparams.speaker_embed_dim, padding_idx=None,
                                           std=hparams.speaker_embedding_weight_std
                                           )
        self.speaker_embedding.weight.data.copy_(tensor_embedding)
        # self.speaker_embedding = Embedding(num_embeddings=pretrained_embedding.size()[0],
        #                                    embedding_dim=hparams.speaker_embed_dim, padding_idx=None,
        #                                    std=hparams.speaker_embedding_weight_std)
        # self.speaker_embedding.weight.data.copy_(pretrained_embedding)

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
    device = args["--device"]
    log_event_path = args["--log-event-path"]
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path = args["--checkpoint"]
    checkpoint_multispeaker_tts = args["--checkpoint-multispeaker-tts"]
    checkpoint = args["--checkpoint"]
    load_embedding = args["--load-embedding"]
    preset = args["--preset"]
    vctk_root = args["--vctk-root"]
    data_root = args["--data-root"]
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

    speaker_dataset = SpeakerDataSet(vctk_root, data_root)
    dataset = PyTorchDataset(speaker_dataset, checkpoint_multispeaker_tts)
    data_loader = data_utils.DataLoader(
        dataset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )
    if device is None:
        device = torch.device("cuda" if use_cuda else "cpu")
    else:
        device = torch.device(int(device))
    model = SpeakerEncoder(hparams.num_mels, hparams.encoder_channels, hparams.kernel_size,
                           hparams.f_mapped, hparams.speaker_embed_dim,
                           hparams.speaker_encoder_attention_num_heads,
                           hparams.speaker_encoder_attention_dim,
                           hparams.dropout,
                           hparams.batch_size,
                           hparams.cloning_sample_size)
    optimizer = optim.Adam(model.parameters(),
                           lr=hparams.initial_learning_rate, betas=(
        hparams.adam_beta1, hparams.adam_beta2),
        eps=hparams.adam_eps, weight_decay=hparams.weight_decay,
        amsgrad=hparams.amsgrad)

    if device.type != "cpu":
        model.cuda()

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
    try:
        train(device, model, data_loader, optimizer, writer,
              init_lr=hparams.initial_learning_rate,
              nepochs=hparams.nepochs,
              checkpoint_interval=hparams.speaker_encoder_checkpoint_interval,
              checkpoint_dir=checkpoint_dir
              )
    except KeyboardInterrupt:
        save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch)
