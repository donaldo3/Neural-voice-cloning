'''
This script is for making list of speakers that are used in 28 trainset, 56 trainset and testset of noisy VCTK
'''


train28_meta = '/home/admin/Music/noisy_VCTK/log_trainset_28spk.txt'
train56_meta = '/home/admin/Music/noisy_VCTK/log_trainset_56spk.txt'
test_meta = '/home/admin/Music/noisy_VCTK/log_testset.txt'
log = [train28_meta, train56_meta, test_meta]

save_meta_train28 = '/home/admin/Music/noisy_VCTK/speakers_trainset_28spk.txt'
save_meta_train56 = '/home/admin/Music/noisy_VCTK/speakers_trainset_56spk.txt'
save_meta_test = '/home/admin/Music/noisy_VCTK/speakers_testset.txt'
save_meta = [save_meta_train28, save_meta_train56, save_meta_test]

for input, output in zip(log, save_meta):
    train_speaker_list = []
    with open(input, 'r') as f:
        lines = f.readlines()
    for line in lines:
        speaker = line.split("_")[0]
        train_speaker_list.append(speaker)
    train_speaker_list = list(set(train_speaker_list))
    train_speaker_list.sort()
    with open(output, 'w') as f:
        for speaker in train_speaker_list:
            f.write(speaker + '\n')

