'''
Train speaker encoder with speaker_embeddings from seq2seq TTS as ground truth.

usage: test_4d_data_loader.py [options]

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
import torch
from torch.utils import data as data_utils
from train_speaker_encoder import SpeakerDataSet, PyTorchDataset, collate_fn
from hparams import hparams
from docopt import docopt
from tqdm import tqdm, trange
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

    #test
    idx=16
    idx_tensor = torch.tensor([idx], dtype=torch.long)

    speaker_dataset = SpeakerDataSet(vctk_root, data_root)
    dataset = PyTorchDataset(speaker_dataset, checkpoint_multispeaker_tts)
    data_loader = data_utils.DataLoader(
        dataset, batch_size=hparams.batch_size, collate_fn=collate_fn,
        num_workers=hparams.num_workers,
        pin_memory=False
    )

    for step, (mel, speaker_embeds) in tqdm(enumerate(data_loader)):
        print("")