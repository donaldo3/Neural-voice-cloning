# From noisy_dst and clean_dst
# Create directory for each speaker according to speaker_list_file
# Move each wav to corresponding speaker directory
import os
import glob
import shutil

if __name__ == '__main__':
    noisy_dst = '/past_projects/DB/VCTK_variants/all_VCTK_noisy'
    clean_dst = '/past_projects/DB/VCTK_variants/all_VCTK_clean'
    speaker_list_file = '/past_projects/DB/VCTK_variants/speakers_trainset_28spk.txt'
    with open(speaker_list_file) as f:
        speaker_list = f.readlines()

    for speaker in speaker_list:
        speaker = speaker.split("\n")[0]
        path = os.path.join(noisy_dst, speaker)
        os.makedirs(path, exist_ok=True)

        path = os.path.join(clean_dst, speaker)
        os.makedirs(path, exist_ok=True)

    noisy_file_list = glob.glob(os.path.join(noisy_dst, "*.wav"))
    clean_file_list = glob.glob(os.path.join(clean_dst, "*.wav"))

    for file in noisy_file_list:
        speaker = file.split("_")[5]
        path = os.path.join(noisy_dst, speaker)
        shutil.copy(file, path)

    for file in clean_file_list:
        speaker = file.split("_")[5]
        path = os.path.join(clean_dst, speaker)
        shutil.copy(file, path)


