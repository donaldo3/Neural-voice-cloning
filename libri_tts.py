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

    # speakers = vctk.available_speakers
    #
    # td = vctk.TranscriptionDataSource(in_dir, speakers=speakers)
    # transcriptions = td.collect_files()
    # speaker_ids = td.labels
    # speaker_ids_unique = np.unique(speaker_ids)
    # speaker_to_speaker_id = {}
    # for i, j in zip(speakers, speaker_ids_unique):
    #     speaker_to_speaker_id[i] = j
    # wav_paths = vctk.WavFileDataSource(
    #     in_dir, speakers=speakers).collect_files()
    #
    # _ignore_speaker = hparams.not_for_train_speaker.split(", ")
    # ignore_speaker = [speaker_to_speaker_id[i] for i in _ignore_speaker]
    wav_paths = glob.glob(os.path.join(in_dir, "*", "*", "*.wav"))

    # Create dictionary of sequence speaker id and original speaker id
    speakers = os.listdir(in_dir)
    speakers.sort()
    speaker_id_to_speaker = {}
    speaker_to_speaker_id = {}
    for i in range(len(speakers)):
        speaker_id_to_speaker[904 + i] = speakers[i]
        speaker_to_speaker_id[speakers[i]] = 904 + i
    meta_path1 = os.path.join(out_dir, "speaker_id_to_speaker.txt")
    with open(meta_path1, 'w', encoding='utf-8') as m:
        for i in range(len(speaker_id_to_speaker)):
            key = i + 904
            m.write('{}|{}\n'.format(str(key), speaker_id_to_speaker[key]))

    for index, wav_path in enumerate(wav_paths):
        speaker = os.path.dirname(wav_path)
        speaker = os.path.dirname(speaker)
        speaker = os.path.basename(speaker)

        text_path = wav_path.replace(".wav", ".normalized.txt")
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


def _process_utterance(out_dir, index, speaker_id, wav_path, text):
    sr = hparams.sample_rate
    filename = os.path.basename(wav_path).replace('.wav', '')

    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)

    # Librosa trim seems to cut off the ending part of speech
    wav, _ = librosa.effects.trim(wav, top_db=60, frame_length=2048, hop_length=512)

    if hparams.rescaling:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # Save trimmed wav
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
