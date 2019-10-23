# coding: utf-8
"""
Separate checkpoints into seq2seq and postnet

usage: extract_postnet.py [options] <checkpoint>

options:
    --hparams=<parmas>                Hyper parameters [default: ].
    --preset=<json>                   Path of preset parameters (json).
    --checkpoint-seq2seq=<path>       Load seq2seq model from checkpoint path.
    --checkpoint-postnet=<path>       Load postnet model from checkpoint path.
    --file-name-suffix=<s>            File name suffix [default: ].
    --max-decoder-steps=<N>           Max decoder steps [default: 500].
    --replace_pronunciation_prob=<N>  Prob [default: 0.0].
    --speaker_id=<id>                 Speaker ID (for multi-speaker model).
    --output-html                     Output html for blog post.
    --speaker-embedding-lut=<path>    Speaker embedding LUT that includes new speaker
    --test-id-list=<path>             Path for world test_id_list.scp
    -h, --help               Show help message.
"""
import torch
import os
from docopt import docopt
from hparams import hparams
from train import build_model
from deepvoice3_pytorch import frontend, builder
_frontend = getattr(frontend, hparams.frontend)
use_cuda = torch.cuda.is_available()

def build_model():
    model = getattr(builder, hparams.builder)(
        n_speakers=hparams.n_speakers,
        speaker_embed_dim=hparams.speaker_embed_dim,
        n_vocab=_frontend.n_vocab,
        embed_dim=hparams.text_embed_dim,
        mel_dim=hparams.num_mels,
        linear_dim=hparams.converter_dim,
        r=hparams.outputs_per_step,
        downsample_step=hparams.downsample_step,
        padding_idx=hparams.padding_idx,
        dropout=hparams.dropout,
        kernel_size=hparams.kernel_size,
        encoder_channels=hparams.encoder_channels,
        decoder_channels=hparams.decoder_channels,
        converter_channels=hparams.converter_channels,
        use_memory_mask=hparams.use_memory_mask,
        trainable_positional_encodings=hparams.trainable_positional_encodings,
        force_monotonic_attention=hparams.force_monotonic_attention,
        use_decoder_state_for_postnet_input=hparams.use_decoder_state_for_postnet_input,
        max_positions=hparams.max_positions,
        speaker_embedding_weight_std=hparams.speaker_embedding_weight_std,
        freeze_embedding=hparams.freeze_embedding,
        window_ahead=hparams.window_ahead,
        window_backward=hparams.window_backward,
        key_projection=hparams.key_projection,
        value_projection=hparams.value_projection,
        vocoder=hparams.vocoder
    )
    return model
def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def save_checkpoint(model, checkpoint_postnet_dir):
    name = "checkpoint_postnet.pth"
    m = model.postnet
    checkpoint_path = os.path.join(checkpoint_postnet_dir, name)
    torch.save({
        "state_dict": m.state_dict(),
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_path = args["<checkpoint>"]
    checkpoint_seq2seq_path = args["--checkpoint-seq2seq"]
    checkpoint_postnet_path = args["--checkpoint-postnet"]
    max_decoder_steps = int(args["--max-decoder-steps"])
    file_name_suffix = args["--file-name-suffix"]
    test_id_list = args["--test-id-list"]
    replace_pronunciation_prob = float(args["--replace_pronunciation_prob"])
    output_html = args["--output-html"]
    speaker_id = args["--speaker_id"]
    speaker_embedding_lut = args["--speaker-embedding-lut"]
    if speaker_id is not None:
        speaker_id = int(speaker_id)
    preset = args["--preset"]
    checkpoint = _load(checkpoint_path)
    params = checkpoint['state_dict']
    model = build_model().to(torch.device(0))
    save_checkpoint(model, checkpoint_postnet_path)
    print("hi")