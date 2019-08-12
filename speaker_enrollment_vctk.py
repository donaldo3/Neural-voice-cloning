"""
Generate speaker embedding from new speaker and save it at the LUT
This LUT is going to be loaded at synthesis2.py

usage: speaker_enrollment_vctk.py [options] <checkpoint-speaker-encoder> <checkpoint-multispeaker-tts>
<checkpoint-speaker-embedding> <vctk-root> <data-root>

options:
    --hparams=<parmas>                    Hyper parameters [default: ].
    --preset=<json>                       Path of preset parameters (json).
    --checkpoint-seq2seq=<path>           Load seq2seq model from checkpoint path.
    --checkpoint-postnet=<path>           Load postnet model from checkpoint path.
    --checkpoint-speaker-encoder=<path>   Load speaker encoder model from checkpoint path.
    --file-name-suffix=<s>                File name suffix [default: ].
    --max-decoder-steps=<N>               Max decoder steps [default: 500].
    --replace_pronunciation_prob=<N>      Prob [default: 0.0].
    --speaker_id=<id>                     Speaker ID (for multi-speaker model).
    --output-html                         Output html for blog post.

    -h, --help               Show help message.
"""
import os

import numpy as np
import torch
from docopt import docopt
from nnmnkwii.datasets import FileSourceDataset
from nnmnkwii.datasets import vctk
from torch.utils import data as data_utils
from torch.utils.data import Dataset

# The deepvoice3 model
from deepvoice3_pytorch import frontend
from deepvoice3_pytorch.modules import Embedding
from hparams import hparams
from speaker_encoder import SpeakerEncoder
from train import MelSpecDataSource
from train import PartialyRandomizedSimilarTimeLengthSampler

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#TODO: Make an interface for giving this. Hard coding is bad.
cloning_sample_id = [0, 1, 2, 3, 4, 5]

class SpeakerDataSet(Dataset):

    '''
    Create list or dictionary of MelSpecDataSource
    '''
    def __init__(self, vctk_root, data_root):
        self.data_root = data_root
        self.mel_spec_datasource_list = []

        speakers = vctk.available_speakers
        td = vctk.TranscriptionDataSource(vctk_root, speakers=speakers)
        transcriptions = td.collect_files()
        speaker_ids = td.labels # All speakers except for p315
        self.speaker_to_speaker_id = td.labelmap

        # Create lists of training speaker_ids
        black_list_speaker = hparams.not_for_train_speaker.split(", ")
        self.cloning_speaker_ids = [self.speaker_to_speaker_id[i] for i in black_list_speaker]

        # Assuming self.speaker_list contains speaker_ids of training speakers
        for spkr_id in self.cloning_speaker_ids:
            Mel = FileSourceDataset(MelSpecDataSource(data_root, speaker_id=spkr_id))
            self.mel_spec_datasource_list.append(Mel)

    '''
    Return MelSpecDataSource of desired speaker idx
    '''
    def __getitem__(self, idx):
        return self.mel_spec_datasource_list[idx]

    def __len__(self):
        return len(self.mel_spec_datasource_list)

def run(model):
    model = model.to(device)
    model.eval()

def _pad(seq, max_len, constant_values=0):
    return np.pad(seq, ((0, max_len - len(seq)), (0, 0)),
                  mode='constant', constant_values=constant_values)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

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
    return padded_mels
'''
Speaker id of cloned speakers are going to be printed on the screen
Speaker ids are increased from the last speaker ID of Multispeaker TTS.
Even though some speakers are not used and some speaker ids are not trained, that space is not 
going to be filled at this stage and be left with dummy numbers.
That is because appending new speaker embedding at the end of LUT has more scalability 
in case speakers other than VCTK are used(eg. Obama voice from Internet)

'''
if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_speaker_encoder = args["<checkpoint-speaker-encoder>"]
    checkpoint_tts = args["<checkpoint-multispeaker-tts>"]
    checkpoint_speaker_embedding = args["<checkpoint-speaker-embedding>"]
    vctk_root = args["<vctk-root>"]
    data_root = args["<data-root>"] # Directory where preprocessed mel-spectrograms are.

    preset = args["--preset"]
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())

    _frontend = getattr(frontend, hparams.frontend)

    '''
    Load speaker encoder model
    '''
    checkpoint =_load(checkpoint_speaker_encoder)
    speaker_encoder_model = SpeakerEncoder(hparams.num_mels, hparams.encoder_channels, hparams.kernel_size,
                           hparams.f_mapped, hparams.speaker_embed_dim,
                           hparams.speaker_encoder_attention_num_heads,
                           hparams.speaker_encoder_attention_dim,
                           hparams.dropout,
                           hparams.batch_size,
                           hparams.cloning_sample_size)
    speaker_encoder_model.load_state_dict(checkpoint["state_dict"])

    '''
    Load mutli-speaker TTS speaker embedding LUT
    '''
    state_dict = _load(checkpoint_tts)["state_dict"]
    key = "embed_speakers.weight"
    pretrained_embedding = state_dict[key]

    '''
    Generate input to speaker encoder with batch size equal to the number of speaker_id 
    '''
    # Create mel batch of size [B x N, T, F]
    speaker_dataset = SpeakerDataSet(vctk_root, data_root)
    sds_list = []
    for i in range(len(speaker_dataset)):
        sds_list.append(speaker_dataset.__getitem__(i))

    # Create mel batch with designated sample id
    cloning_samples = []
    # cloning_samples has B x N samples
    for speaker in sds_list:
        for i in cloning_sample_id:
            cloning_samples.append(speaker.__getitem__(i))

    # Match the length of frames in the batch
    cloning_samples = collate_fn_sub(cloning_samples)

    '''
    Run speaker encoder and generate new speaker embedding
    '''
    cloning_samples = cloning_samples.to(device)
    speaker_encoder_model.cuda()
    pred_speaker_embeddings = speaker_encoder_model(cloning_samples)

    '''
    Create new LUT to include new speaker embeddings.
    Create new tensor with copied old embeddings and new embeddings 
    Copy that tensor into LUT weight.
    Store that LUT as checkpoint
    '''
    new_embedding = Embedding(num_embeddings=pretrained_embedding.data.size()[0] + hparams.batch_size,
                              embedding_dim=hparams.speaker_embed_dim, padding_idx=None,
                              std=hparams.speaker_embedding_weight_std
                              )
    new_tensor = torch.cat((pretrained_embedding, pred_speaker_embeddings), 0)
    new_embedding.weight.data.copy_(new_tensor)

    # Save new_embedding
    dir = checkpoint_speaker_embedding
    checkpoint_path = os.path.join(dir, "speaker_embedding_with_cloning_speakers.tar")
    torch.save(
        {"state_dict": new_embedding.state_dict()}, checkpoint_path
    )
    print("Lookup table including cloned speaker embeddings has been saved at {}"
          .format(checkpoint_path))

    print("Mapping between newly added speaker ids and speaker names are:")
    starting_id = pretrained_embedding.data.size()[0]
    ending_id = starting_id + pred_speaker_embeddings.data.size()[0]

    ids = range(starting_id, ending_id)
    speakers = hparams.not_for_train_speaker.split(", ")
    print("{:>5}, {:>5}".format("id", "speaker"))
    for id, speaker in zip(ids, speakers):
        print("{:>5}, {:>5}".format(id, speaker))




















