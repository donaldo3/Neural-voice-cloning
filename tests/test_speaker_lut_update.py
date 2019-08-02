'''
To test speaker embedding LUT is modifiable
1. modify the value of certain row
2. append

usage: test_speaker_lut_update.py [options]

options:
    --preset=<json>                 Path of preset parameters (json).

'''
import os
from docopt import docopt
import torch
from hparams import hparams
from deepvoice3_pytorch.modules import Embedding
import numpy as np
use_cuda = True

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


if __name__ == "__main__":
    args = docopt(__doc__)
    preset = args["--preset"]
    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    tts_checkpoint_path = "190709/checkpoint_step004510000.pth"
    speaker_embedding = Embedding(num_embeddings=hparams.n_speakers,
                                  embedding_dim=hparams.speaker_embed_dim, padding_idx=None,
                                  std=hparams.speaker_embedding_weight_std)


    # 1. Modify
    # Load LUT from TTS
    state = _load(tts_checkpoint_path)["state_dict"]
    key = "embed_speakers.weight"
    pretrained_embedding = state[key]
    speaker_embedding.weight.data.copy_(pretrained_embedding)
    # Modify LUT
    zeros = torch.zeros(1, 16)
    speaker_embedding.weight.data[69] = zeros
    # Save LUT
    dir = "tests/saved_lut"
    os.makedirs(dir, exist_ok=True)
    checkpoint_path = os.path.join(dir, "lut_modified.tar")
    torch.save({
        "state_dict": speaker_embedding.state_dict()
    }, checkpoint_path)

    # Reload LUT
    state = _load(checkpoint_path)["state_dict"]
    key = "weight"
    reloaded_embedding = state[key]
    reloaded_embedding = reloaded_embedding.numpy()

    # 2. Append
    # Load LUT
    state = _load(tts_checkpoint_path)["state_dict"]
    key = "embed_speakers.weight"
    pretrained_embedding = state[key].cpu().numpy()

    # Append new row to LUT
    nd = pretrained_embedding[0]
    nd = np.expand_dims(nd, axis=0)
    nd_109 = np.concatenate((pretrained_embedding, nd), axis=0)
    embedding = torch.FloatTensor(nd_109)
    emb_109 = Embedding(num_embeddings=hparams.n_speakers + 1, embedding_dim=hparams.speaker_embed_dim, padding_idx=None,
                                  std=hparams.speaker_embedding_weight_std)
    emb_109.weight.data.copy_(embedding)

    # Replace LUT and save Multispeaker TTS checkpoint
    dir = "tests/saved_lut"
    checkpoint_path = os.path.join(dir, "emb_109.tar")
    torch.save(
        {"state_dict": emb_109.state_dict()}, checkpoint_path)
    # Test synth with previously generated TTS checkpoint