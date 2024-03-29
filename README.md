Korean branch for emotiontts project
# Neural voice cloning by Sunghee Jung
## 1. How to train
You need to train 1) Multi-speaker tts and 2) Speaker encoder.
### 1. How to train multi-speaker tts
 In case you want to spare some VCTK speakers for evaluation set or test set, there is hparam option for that.
 Write down the speakers as follows.
 So far, only sequential speakers are supported to spare.
  ```
  "not_for_train_speaker": "300, 301, 302, 303, 304, 305"
 ```
 Rest of the process for training is identical to DeepVoice3 by r9y9.
 In case you would like WORLD vocoder support, refer to the readme in [https://github.com/hash2430/dv3_world]
 * Argument
 ```
--data-root=<path-to-preprocessed-vctk>
--cmp-root=<path-to-world-training-features>
--preset=presets/deepvoice3_vctk.json
--checkpoint-dir=<where-to-save-multispeaker-tts>
```
### 2. How to train speaker encoder
Aside from multispeaker tts, you will need a speaker encoder from which you can extract an embedding of new target speaker whose voice you want to clone.
```
python train_speaker_encoder.py
--data-root=<preprocessed-vctk>
--checkpoint-multispeaker-tts=<where-to-find-multispeaker-tts-model>
--checkpoint-dir=<where-to-save-speaker-encoder-model>
--vctk-root=<path-to-VCTK-Corpus>
--preset=presets/deepvoice3_vctk.json
```
## 2. Speaker enrollment
At this stage, speaker embedding lookup table generated for training speakers are loaded 
and newly inferenced embedding of cloned speaker is appended at the end.
### hparams configuration
* 'batch_size' represents the number of speakers to be enrolled at a time.
    * In case you are enrolling speakers who are not in VCTK, it supports only one speaker enrollment at a time.
* 'cloning_sample_size' must match the number of cloning samples you want to use for cloning.

### VCTK speaker enrollment
In case you want to extract speaker embeddings from the VCTK speakers who are not used for training,
there is 'speaker_enrollment_vctk.py'. This script extracts speaker embedding from mel spectrograms of VCTK speakers.
There are a couple of differences between how to use this module and general module
* You can enroll multiple speakers by setting batch_size equal to the number of speakers who are left out during training
* You can specify the which audio samples you are using to extract embedding.
* preprocess.py 
    * You must run preprocess with 'vctk_cloning' mode to extract mel spectrogram of target speakers (who are left out during preprocessing for training speakers with 'vctk' mode)
    * Run preprocess.py with the same options as preparing for 'vctk' except this time the mode is 'vctk_cloning' instead of  'vctk'
    * Preprocessing command
    ```
    python preprocess.py
    --preset=presets/deepvoice3_vctk.json vctk_cloning <vctk-root> <data-root>
    ```
* VCTK speaker enrollment command 
```
python speaker_enrollment_vctk.py [options] <checkpoint-speaker-encoder> <checkpoint-multispeaker-tts>
<checkpoint-speaker-embedding> <vctk-root> <data-root> 
```
### General speaker enrollment: whoever you wish to clone
* You can try enrolling some random speaker that are found online or maybe yourself.
* I recommed enhancing the speech with SEGAN before you use it to extract embedding and enroll that person to your lookup table.
* In order to enroll speaker who are not VCTK, use the 'speaker_enrollment_general.py' script.
    * Unlike 'speaker_enrollment_vctk', it extracts speaker embedding from wav files in the directory you provide.
    * Then it adds this new speaker embedding at the end of your lookup table which was trained during multispeaker-tts training step.

```
python speaker_enrollment_general.py [options] <checkpoint-speaker-encoder> <checkpoint-multispeaker-tts>
<checkpoint-speaker-embedding> <wav-root> <speaker-meta>
```
speaker-meta (which is given as program argument by you) and speaker-id (which is automatically decided according 
to the exsiting LUT size) are used to decided the name of speaker LUT checkpoint to be saved.
## 3. Synthesize with enrolled speaker
You need to specify where to load 1) multi-speaker tts and 2) speaker lookup table (which includes the result of speaker enrollment and trained speaker embeddings)
```
python synthesis2.py
<checkpoint-multi-speaker-tts>
text_list.txt
<path-to-save-output>
--preset=presets/deepvoice3_vctk.json
--speaker_id=<speaker_id_of_cloned_speaker_or_trained_speaker>
--speaker-embedding-lut=<path-to-lut-generated-from-enrollment-step>
```

## 4. Embedding enhancement
The idea of embedding enhancement is to experiment if it is possible to use wild low-quality DB for voice cloning purpose.
I have tested with speech enhancement + speaker encoder to generate cloned voice embedding. That worked fine.
However, how about the other way around? How about speaker encoder + embedding enhancement?
Embedding enhancement module is trained in a supervised manner in order to generate the embedding of clean speech given embedding of noisy speech as input.
The inspiration is from the paper "Predicting Expressive Speaking Style from Text in End-To-End Speech Synthesis".
This does not work by now. Maybe needs some more experiment or maybe it is an absurd idea. Anyways, below is how to try this idea.
### 4.1. Paring noisy and clean VCTK
You need to adjust hard-coded directories of following scripts to match yours.
* Download VCTK variations such as noisy VCTK, reverb VCTK, noisy-reverb VCTK, DR(Device-Record) VCTK.
* Run "embedding_enhancement_data_prepare/summ_vctk.py" to generate lists of speakers.
* Run "embedding_enhancement_data_prepare/pair_db.py" to pair wav files of noisy and clean VCTK with matching file names.
* Run "embedding_enhancement_data_prepare/pair_with_speaker.py" to create subdirectory of each speaker under noisy and clean VCTK wavs.
### 4.2. Preparing mel spectrogram
* Run "preprocess.py" **twice**: once for clean mel and once for noisy mel.
    * **This is where I made mistake.** I preprocessed it into 80 dim mel and tested with TTS and encoder that is trained on 60 dim mel.
```
--preset=presets/deepvoice3_vctk_80mel_256spk_emb.json
vctk
/past_projects/DB/VCTK_variants/all_VCTK_clean
/past_projects/DB/VCTK_variants/all_VCTK_clean_mel_60
```

### 4.3. Training embedding enhancement
* Train embedding enhancement
```
--speaker-list-file=/past_projects/DB/VCTK_variants/speakers_trainset_28spk.txt
--clean-data-root=/past_projects/DB/VCTK_variants/all_VCTK_clean_mel_60
--noisy-data-root=/past_projects/DB/VCTK_variants/all_VCTK_noisy_mel_60
--preset=presets/deepvoice3_vctk.json
--checkpoint-speaker-encoder=/past_projects/deepvoice3_world_converter/checkpoints/t2_190822_speaker_encoder/checkpoint_step000034000_speaker_encoder.pth
--checkpoint-dir=/past_projects/deepvoice3_world_converter/checkpoints/embedding_enhancement_190905
```
### 4.4. Test: Has not been done yet. Because training did not converge. 
#### 4.4.1. Create embedding from noisy speech
#### 4.4.2. Run embedding enhancement
#### 4.4.3. Save the enhanced embedding to a lookup table
#### 4.4.4. Load Multispeaker TTS and lookuptable using sysnthesis2.py 
## 5. Transfer learning with LJSpeech
### 5.1. Data preparation
* run preprocess and generate and save mel and linear spectrums as in original DeepVoice3.
    * preset: You can either use ljspeech or vctk preset as long as hop_size, sample_rate matches. But if you want separate rescaling configuration for both DB, use separate preset.
* Since I use WORLD as default, create cmp as explained in 
    * run  '03_prepare_acoustic_features.sh' instead of  '03_vctk_prepare_acoustic_features.sh'. That script is for multispeaker.
    * modify 'gen_file_id_list.sh' to fit your environment
# 6. Using LJSpeech and VCTK altogether at the same time for multi-speaker tts training
 - --force-multispeaker flag on
    - --data-root-single, --data-root-multi, --cmp-root-single, --cmp-root-multi must be given
    - The number of speakers must match the number of 'multispeaker speakers' + 1
    - The suffix(or separator) for preprocessed mel and linear spectrogram name for both single and multispeaker DB must match.


![alt text](assets/banner.jpg)

# Deepvoice3_pytorch

[![PyPI](https://img.shields.io/pypi/v/deepvoice3_pytorch.svg)](https://pypi.python.org/pypi/deepvoice3_pytorch)
[![Build Status](https://travis-ci.org/r9y9/deepvoice3_pytorch.svg?branch=master)](https://travis-ci.org/r9y9/deepvoice3_pytorch)
[![Build status](https://ci.appveyor.com/api/projects/status/8eurjakfaofbr24k?svg=true)](https://ci.appveyor.com/project/r9y9/deepvoice3-pytorch)
[![DOI](https://zenodo.org/badge/108992863.svg)](https://zenodo.org/badge/latestdoi/108992863)

PyTorch implementation of convolutional networks-based text-to-speech synthesis models:

1. [arXiv:1710.07654](https://arxiv.org/abs/1710.07654): Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning.
2. [arXiv:1710.08969](https://arxiv.org/abs/1710.08969): Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention.

Audio samples are available at https://r9y9.github.io/deepvoice3_pytorch/.

## Online TTS demo

Notebooks supposed to be executed on https://colab.research.google.com are available:

- [DeepVoice3: Multi-speaker text-to-speech demo](https://colab.research.google.com/github/r9y9/Colaboratory/blob/master/DeepVoice3_multi_speaker_TTS_en_demo.ipynb)
- [DeepVoice3: Single-speaker text-to-speech demo](https://colab.research.google.com/github/r9y9/Colaboratory/blob/master/DeepVoice3_single_speaker_TTS_en_demo.ipynb)

## Highlights

- Convolutional sequence-to-sequence model with attention for text-to-speech synthesis
- Multi-speaker and single speaker versions of DeepVoice3
- Audio samples and pre-trained models
- Preprocessor for [LJSpeech (en)](https://keithito.com/LJ-Speech-Dataset/), [JSUT (jp)](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) and [VCTK](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) datasets, as well as [carpedm20/multi-speaker-tacotron-tensorflow](https://github.com/carpedm20/multi-Speaker-tacotron-tensorflow) compatible custom dataset (in JSON format)
- Language-dependent frontend text processor for English and Japanese

### Samples

- [Ja Step000380000 Predicted](https://soundcloud.com/user-623907374/ja-step000380000-predicted)
- [Ja Step000370000 Predicted](https://soundcloud.com/user-623907374/ja-step000370000-predicted)
- [Ko_single Step000410000 Predicted](https://soundcloud.com/user-623907374/ko-step000410000-predicted)
- [Ko_single Step000400000 Predicted](https://soundcloud.com/user-623907374/ko-step000400000-predicted)
- [Ko_multi Step001680000 Predicted](https://soundcloud.com/user-623907374/step001680000-predicted)
- [Ko_multi Step001700000 Predicted](https://soundcloud.com/user-623907374/step001700000-predicted)

## Pretrained models

**NOTE**: pretrained models are not compatible to master. To be updated soon.

 | URL | Model      | Data     | Hyper paramters                                  | Git commit | Steps  |
 |-----|------------|----------|--------------------------------------------------|----------------------|--------|
 | [link](https://www.dropbox.com/s/5ucl9remrwy5oeg/20180505_deepvoice3_checkpoint_step000640000.pth?dl=0) | DeepVoice3 | LJSpeech | [link](https://www.dropbox.com/s/0ck82unm0bo0rxd/20180505_deepvoice3_ljspeech.json?dl=0) | [abf0a21](https://github.com/r9y9/deepvoice3_pytorch/tree/abf0a21f83aeb451b918f867bc23378f1e2e608b)| 640k |
 |  [link](https://www.dropbox.com/s/1y8bt6bnggbzzlp/20171129_nyanko_checkpoint_step000585000.pth?dl=0)   | Nyanko     | LJSpeech | `builder=nyanko,preset=nyanko_ljspeech`     | [ba59dc7](https://github.com/r9y9/deepvoice3_pytorch/tree/ba59dc75374ca3189281f6028201c15066830116) | 585k |
  |  [link](https://www.dropbox.com/s/uzmtzgcedyu531k/20171222_deepvoice3_vctk108_checkpoint_step000300000.pth?dl=0)   | Multi-speaker DeepVoice3     | VCTK | `builder=deepvoice3_multispeaker,preset=deepvoice3_vctk`     | [0421749](https://github.com/r9y9/deepvoice3_pytorch/tree/0421749af908905d181f089f06956fddd0982d47) | 300k + 300k |

To use pre-trained models, it's highly recommended that you are on the **specific git commit** noted above. i.e.,

```
git checkout ${commit_hash}
```

Then follow the "Synthesize from a checkpoint" section in the README of the specific git commit. Please notice that the latest development version of the repository may not work.

You could try for example:

```
# pretrained model (20180505_deepvoice3_checkpoint_step000640000.pth)
# hparams (20180505_deepvoice3_ljspeech.json)
git checkout 4357976
python synthesis.py --preset=20180505_deepvoice3_ljspeech.json \
  20180505_deepvoice3_checkpoint_step000640000.pth \
  sentences.txt \
  output_dir
```

## Notes on hyper parameters

- Default hyper parameters, used during preprocessing/training/synthesis stages, are turned for English TTS using LJSpeech dataset. You will have to change some of parameters if you want to try other datasets. See `hparams.py` for details.
- `builder` specifies which model you want to use. `deepvoice3`, `deepvoice3_multispeaker` [1] and `nyanko` [2] are surpprted.
- Hyper parameters described in DeepVoice3 paper for single speaker didn't work for LJSpeech dataset, so I changed a few things. Add dilated convolution, more channels, more layers and add guided attention loss, etc. See code for details. The changes are also applied for multi-speaker model.
- Multiple attention layers are hard to learn. Empirically, one or two (first and last) attention layers seems enough.
- With guided attention (see https://arxiv.org/abs/1710.08969), alignments get monotonic more quickly and reliably if we use multiple attention layers. With guided attention, I can confirm five attention layers get monotonic, though I cannot get speech quality improvements.
- Binary divergence (described in https://arxiv.org/abs/1710.08969) seems stabilizes training particularly for deep (> 10 layers) networks.
- Adam with step lr decay works. However, for deeper networks, I find Adam + noam's lr scheduler is more stable.

## Requirements

- Python 3
- CUDA >= 8.0
- PyTorch >= v0.4.0
- [nnmnkwii](https://github.com/r9y9/nnmnkwii) >= v0.0.11
- [MeCab](http://taku910.github.io/mecab/) (Japanese only)

## Installation

Please install packages listed above first, and then

```
git clone https://github.com/r9y9/deepvoice3_pytorch && cd deepvoice3_pytorch
pip install -e ".[bin]"
```

## Getting started

### Preset parameters

There are many hyper parameters to be turned depends on what model and data you are working on. For typical datasets and models, parameters that known to work good (**preset**) are provided in the repository. See `presets` directory for details. Notice that

1. `preprocess.py`
2. `train.py`
3. `synthesis.py`

accepts `--preset=<json>` optional parameter, which specifies where to load preset parameters. If you are going to use preset parameters, then you must use same `--preset=<json>` throughout preprocessing, training and evaluation. e.g.,

```
python preprocess.py --preset=presets/deepvoice3_ljspeech.json ljspeech ~/data/LJSpeech-1.0
python train.py --preset=presets/deepvoice3_ljspeech.json --data-root=./data/ljspeech
```

instead of

```
python preprocess.py ljspeech ~/data/LJSpeech-1.0
# warning! this may use different hyper parameters used at preprocessing stage
python train.py --preset=presets/deepvoice3_ljspeech.json --data-root=./data/ljspeech
```

### 0. Download dataset

- LJSpeech (en): https://keithito.com/LJ-Speech-Dataset/
- VCTK (en): http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
- JSUT (jp): https://sites.google.com/site/shinnosuketakamichi/publication/jsut
- NIKL (ko) (**Need korean cellphone number to access it**): http://www.korean.go.kr/front/board/boardStandardView.do?board_id=4&mn_id=17&b_seq=464

### 1. Preprocessing

Usage:

```
python preprocess.py ${dataset_name} ${dataset_path} ${out_dir} --preset=<json>
```

Supported `${dataset_name}`s are:

- `ljspeech` (en, single speaker)
- `vctk` (en, multi-speaker)
- `jsut` (jp, single speaker)
- `nikl_m` (ko, multi-speaker)
- `nikl_s` (ko, single speaker)

Assuming you use preset parameters known to work good for LJSpeech dataset / DeepVoice3 and have data in `~/data/LJSpeech-1.0`, then you can preprocess data by:

```
python preprocess.py --preset=presets/deepvoice3_ljspeech.json ljspeech ~/data/LJSpeech-1.0/ ./data/ljspeech
```

When this is done, you will see extracted features (mel-spectrograms and linear spectrograms) in `./data/ljspeech`.

#### 1-1. Building custom dataset. (using json_meta)
Building your own dataset, with metadata in JSON format (compatible with [carpedm20/multi-speaker-tacotron-tensorflow](https://github.com/carpedm20/multi-Speaker-tacotron-tensorflow)) is currently supported.
Usage:

```
python preprocess.py json_meta ${list-of-JSON-metadata-paths} ${out_dir} --preset=<json>
```
You may need to modify pre-existing preset JSON file, especially `n_speakers`. For english multispeaker, start with `presets/deepvoice3_vctk.json`.

Assuming you have dataset A (Speaker A) and dataset B (Speaker B), each described in the JSON metadata file `./datasets/datasetA/alignment.json` and `./datasets/datasetB/alignment.json`, then you can preprocess  data by:

```
python preprocess.py json_meta "./datasets/datasetA/alignment.json,./datasets/datasetB/alignment.json" "./datasets/processed_A+B" --preset=(path to preset json file)
```

#### 1-2. Preprocessing custom english datasets with long silence. (Based on [vctk_preprocess](vctk_preprocess/))

Some dataset, especially automatically generated dataset may include long silence and undesirable leading/trailing noises, undermining the char-level seq2seq model.
(e.g. VCTK, although this is covered in vctk_preprocess)

To deal with the problem, `gentle_web_align.py` will
- **Prepare phoneme alignments for all utterances**
- Cut silences during preprocessing

`gentle_web_align.py` uses [Gentle](https://github.com/lowerquality/gentle), a kaldi based speech-text alignment tool. This accesses web-served Gentle application, aligns given sound segments with transcripts and converts the result to HTK-style label files, to be processed in `preprocess.py`. Gentle can be run in Linux/Mac/Windows(via Docker).

Preliminary results show that while HTK/festival/merlin-based method in `vctk_preprocess/prepare_vctk_labels.py` works better on VCTK, Gentle is more stable with audio clips with ambient noise. (e.g. movie excerpts)

Usage:
(Assuming Gentle is running at `localhost:8567` (Default when not specified))
1. When sound file and transcript files are saved in separate folders. (e.g. sound files are at `datasetA/wavs` and transcripts are at `datasetA/txts`)
```
python gentle_web_align.py -w "datasetA/wavs/*.wav" -t "datasetA/txts/*.txt" --server_addr=localhost --port=8567
```

2. When sound file and transcript files are saved in nested structure. (e.g. `datasetB/speakerN/blahblah.wav` and `datasetB/speakerN/blahblah.txt`)
```
python gentle_web_align.py --nested-directories="datasetB" --server_addr=localhost --port=8567
```
**Once you have phoneme alignment for each utterance, you can extract features by running `preprocess.py`**

### 2. Training

Usage:

```
python train.py --data-root=${data-root} --preset=<json> --hparams="parameters you may want to override"
```

Suppose you build a DeepVoice3-style model using LJSpeech dataset, then you can train your model by:

```
python train.py --preset=presets/deepvoice3_ljspeech.json --data-root=./data/ljspeech/
```

Model checkpoints (.pth) and alignments (.png) are saved in `./checkpoints` directory per 10000 steps by default.

#### NIKL

Pleae check [this](https://github.com/homink/deepvoice3_pytorch/blob/master/nikl_preprocess/README.md) in advance and follow the commands below.

```
python preprocess.py nikl_s ${your_nikl_root_path} data/nikl_s --preset=presets/deepvoice3_nikls.json

python train.py --data-root=./data/nikl_s --checkpoint-dir checkpoint_nikl_s --preset=presets/deepvoice3_nikls.json
```

### 4. Monitor with Tensorboard

Logs are dumped in `./log` directory by default. You can monitor logs by tensorboard:

```
tensorboard --logdir=log
```

### 5. Synthesize from a checkpoint

Given a list of text, `synthesis.py` synthesize audio signals from trained model. Usage is:

```
python synthesis.py ${checkpoint_path} ${text_list.txt} ${output_dir} --preset=<json>
```

Example test_list.txt:

```
Generative adversarial network or variational auto-encoder.
Once upon a time there was a dear little girl who was loved by every one who looked at her, but most of all by her grandmother, and there was nothing that she would not have given to the child.
A text-to-speech synthesis system typically consists of multiple stages, such as a text analysis frontend, an acoustic model and an audio synthesis module.
```

## Advanced usage

### Multi-speaker model

VCTK and NIKL are supported dataset for building a multi-speaker model.

#### VCTK
Since some audio samples in VCTK have long silences that affect performance, it's recommended to do phoneme alignment and remove silences according to [vctk_preprocess](vctk_preprocess/).

Once you have phoneme alignment for each utterance, you can extract features by:

```
python preprocess.py vctk ${your_vctk_root_path} ./data/vctk
```

Now that you have data prepared, then you can train a multi-speaker version of DeepVoice3 by:

```
python train.py --data-root=./data/vctk --checkpoint-dir=checkpoints_vctk \
   --preset=presets/deepvoice3_vctk.json \
   --log-event-path=log/deepvoice3_multispeaker_vctk_preset
```

If you want to reuse learned embedding from other dataset, then you can do this instead by:

```
python train.py --data-root=./data/vctk --checkpoint-dir=checkpoints_vctk \
   --preset=presets/deepvoice3_vctk.json \
   --log-event-path=log/deepvoice3_multispeaker_vctk_preset \
   --load-embedding=20171213_deepvoice3_checkpoint_step000210000.pth
```

This may improve training speed a bit.

#### NIKL

You will be able to obtain cleaned-up audio samples in ../nikl_preprocoess. Details are found in [here](https://github.com/homink/speech.ko).


Once NIKL corpus is ready to use from the preprocessing, you can extract features by:

```
python preprocess.py nikl_m ${your_nikl_root_path} data/nikl_m
```

Now that you have data prepared, then you can train a multi-speaker version of DeepVoice3 by:

```
python train.py --data-root=./data/nikl_m  --checkpoint-dir checkpoint_nikl_m \
   --preset=presets/deepvoice3_niklm.json
```

### Speaker adaptation

If you have very limited data, then you can consider to try fine-turn pre-trained model. For example, using pre-trained model on LJSpeech, you can adapt it to data from VCTK speaker `p225` (30 mins) by the following command:

```
python train.py --data-root=./data/vctk --checkpoint-dir=checkpoints_vctk_adaptation \
    --preset=presets/deepvoice3_ljspeech.json \
    --log-event-path=log/deepvoice3_vctk_adaptation \
    --restore-parts="20171213_deepvoice3_checkpoint_step000210000.pth"
    --speaker-id=0
```

From my experience, it can get reasonable speech quality very quickly rather than training the model from scratch.

There are two important options used above:

- `--restore-parts=<N>`: It specifies where to load model parameters. The differences from the option `--checkpoint=<N>` are 1) `--restore-parts=<N>` ignores all invalid parameters, while `--checkpoint=<N>` doesn't. 2) `--restore-parts=<N>` tell trainer to start from 0-step, while `--checkpoint=<N>` tell trainer to continue from last step. `--checkpoint=<N>` should be ok if you are using exactly same model and continue to train, but it would be useful if you want to customize your model architecture and take advantages of pre-trained model.
- `--speaker-id=<N>`: It specifies what speaker of data is used for training. This should only be specified if you are using multi-speaker dataset. As for VCTK, speaker id is automatically assigned incrementally (0, 1, ..., 107) according to the `speaker_info.txt` in the dataset.

If you are training multi-speaker model, speaker adaptation will only work **when `n_speakers` is identical**.

## Trouble shooting

### [#5](https://github.com/r9y9/deepvoice3_pytorch/issues/5) RuntimeError: main thread is not in main loop


This may happen depending on backends you have for matplotlib. Try changing backend for matplotlib and see if it works as follows:

```
MPLBACKEND=Qt5Agg python train.py ${args...}
```

In [#78](https://github.com/r9y9/deepvoice3_pytorch/pull/78#issuecomment-385327057), engiecat reported that changing the backend of matplotlib from Tkinter(TkAgg) to PyQt5(Qt5Agg) fixed the problem.

## Acknowledgements

Part of code was adapted from the following projects:

- https://github.com/keithito/tacotron
- https://github.com/facebookresearch/fairseq-py

Banner and logo created by [@jraulhernandezi](https://github.com/jraulhernandezi) ([#76](https://github.com/r9y9/deepvoice3_pytorch/issues/76))

