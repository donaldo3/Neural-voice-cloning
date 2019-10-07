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
import sys
import wave

'''
audio: from pcm to wav of sample rate hparams.sr
text: use .pron
'''

def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    wav_paths = glob.glob(os.path.join(in_dir, "*", "raw", "*.pcm"))

    # Create dictionary of sequence speaker id and speaker
    speakers = os.listdir(in_dir)
    speakers.sort()
    speaker_id_to_speaker = {}
    speaker_to_speaker_id = {}
    for i in range(len(speakers)):
        speaker_id_to_speaker[i] = speakers[i]
        speaker_to_speaker_id[speakers[i]] = i
    meta_path1 = os.path.join(out_dir, "speaker_id_to_speaker.txt")
    with open(meta_path1, 'w', encoding='utf-8') as m:
        for i in range(len(speaker_id_to_speaker)):
            key = i
            m.write('{}|{}\n'.format(str(key), speaker_id_to_speaker[key]))

    for index, wav_path in enumerate(wav_paths):
        speaker = os.path.dirname(wav_path)
        speaker = os.path.dirname(speaker)
        speaker = os.path.basename(speaker)

        text_path = wav_path.replace(".pcm", ".pron").replace("raw", "script")
        with open(text_path, 'r') as f:
            text = f.readline().rstrip('\n')
        futures.append(executor.submit(
            partial(_process_utterance, out_dir, index + 1, speaker_to_speaker_id[speaker], wav_path, text)))
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


def _process_utterance(out_dir, index, speaker_id, pcm_path, text):
    sr = hparams.sample_rate
    filename = os.path.basename(pcm_path).replace('.pcm', '')

    with open(pcm_path, 'rb')as pcmfile:
        pcmdata = pcmfile.read()

    wav_path = pcm_path.replace('pcm', 'wav').replace('raw', 'wav_44100')
    if not os.path.isdir(os.path.dirname(wav_path)):
        os.makedirs(os.path.dirname(wav_path))

    with wave.open(wav_path, 'wb') as wavfile:
        wavfile.setparams((1, 2, 44100, 0, 'NONE', 'NONE'))
        wavfile.writeframes(pcmdata)

    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)
    os.remove(wav_path)
    # Librosa trim seems to cut off the ending part of speech
    wav, _ = librosa.effects.trim(wav, top_db=25, frame_length=2048, hop_length=512)

    if hparams.rescaling:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # Save trimmed wav
    if hparams.save_preprocessed_wav != "":
        save_wav_path = hparams.save_preprocessed_wav
        env_dir = os.path.dirname(wav_path)
        env = os.path.basename(env_dir)

        speaker = os.path.dirname(env_dir)
        speaker = os.path.basename(speaker)
        save_wav_path = os.path.join(save_wav_path, speaker, env, os.path.basename(wav_path))
        dir = os.path.dirname(save_wav_path)
        if not os.path.exists(dir):
            os.system('mkdir {} -p'.format(dir))
        audio.save_wav(wav, save_wav_path)

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

    # Write the spectrograms to disk:
    spectrogram_filename = '{}{}spec.npy'.format(filename, hparams.spec_cmp_separator)
    mel_filename = '{}{}mel.npy'.format(filename, hparams.spec_cmp_separator)
    np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example:
    return (spectrogram_filename, mel_filename, n_frames, text, speaker_id)
