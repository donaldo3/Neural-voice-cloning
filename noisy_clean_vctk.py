from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import audio
from nnmnkwii.datasets import vctk
from nnmnkwii.io import hts
from hparams import hparams
from os.path import exists
import librosa
import re
import glob

def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    _ignore_speaker = hparams.not_for_train_speaker.split(", ")

    wav_paths = glob.glob(os.path.join(in_dir, "p*", "*.wav"))
    for wav_path in wav_paths:
        file = os.path.basename(wav_path)
        speaker = file.split("_")[1]
        if speaker.replace('p', '') in _ignore_speaker:
            continue
        futures.append(executor.submit(
            partial(_process_utterance, in_dir, out_dir, speaker, wav_path)
        ))
    return [future.result() for future in tqdm(futures)]


def start_at(labels):
    has_silence = labels[0][-1] == "pau"
    if not has_silence:
        return labels[0][0]
    for i in range(1, len(labels)):
        if labels[i][-1] != "pau":
            return labels[i][0]
    assert False


def end_at(labels):
    has_silence = labels[-1][-1] == "pau"
    if not has_silence:
        return labels[-1][1]
    for i in range(len(labels) - 2, 0, -1):
        if labels[i][-1] != "pau":
            return labels[i][1]
    assert False

# Save mel spectrogram of clean and noisy VCTK
# Save meta file with framelength for each speaker in clean directory
def _process_utterance(in_dir, out_dir, speaker, wav_path):
    filename = wav_path.replace('.wav', '.npy')

    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)
    #wav, _ = librosa.effects.trim(wav, top_db=25)
    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
    n_frames = mel_spectrogram.shape[1]

    # Write the spectrograms to disk:
    mel_filename = filename.replace(in_dir, out_dir)
    dir = os.path.dirname(mel_filename)
    os.makedirs(dir, exist_ok=True)
    np.save(mel_filename, mel_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example:
    return (mel_filename, n_frames, speaker)
