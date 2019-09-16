import glob
import os
import shutil

# TODO: Do it with 56 spk
# I originally wanted to make it with 56 spk, but for fair comparison with 'SEGAN case', I try it with 28 spk first
prefix_=['n28_', 'nr28_', 'r28_', 'DR28_']
noisy_src_=['/past_projects/DB/VCTK_variants/noisy_VCTK/noisy_trainset_28spk_wav',
           '/past_projects/DB/VCTK_variants/noisyreverb_VCTK/noisyreverb_trainset_28spk_wav',
           '/past_projects/DB/VCTK_variants/reverb_VCTK/reverb_trainset_28spk_wav',
           '/past_projects/DB/VCTK_variants/DR-VCTK/device-recorded_trainset_wav_16k'
            ]

clean_src_=['/past_projects/DB/VCTK_variants/noisy_VCTK/clean_trainset_28spk_wav',
           '/past_projects/DB/VCTK_variants/noisy_VCTK/clean_trainset_28spk_wav',
           '/past_projects/DB/VCTK_variants/noisy_VCTK/clean_trainset_28spk_wav',
            '/past_projects/DB/VCTK_variants/DR-VCTK/clean_trainset_wav_16k']

noisy_dst = '/past_projects/DB/VCTK_variants/all_VCTK_noisy'
clean_dst = '/past_projects/DB/VCTK_variants/all_VCTK_clean'

for noisy_src, clean_src, prefix in zip(noisy_src_, clean_src_, prefix_):
    noisy_files = glob.glob(os.path.join(noisy_src, '*.wav'))
    clean_files = glob.glob(os.path.join(clean_src, '*.wav'))

    for wav in noisy_files:
        file = os.path.basename(wav)
        file = prefix + file
        savepath = os.path.join(noisy_dst, file)
        shutil.copy(wav, savepath)

    for wav in clean_files:
        file = os.path.basename(wav)
        file = prefix + file
        savepath = os.path.join(clean_dst, file)
        shutil.copy(wav, savepath)