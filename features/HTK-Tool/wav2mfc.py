import os
import pandas as pd
import re


def wav2mfc(PATH, files):
    raw_audio_list = [(PATH + files[j]) for j in range(len(files))]
    mfc_audio_list = [(PATH + (re.sub(r'.raw', '.mfc', files[i]))) for i in range(len(files))]
    wav_audio_list = [(PATH + (re.sub(r'.raw', '.wav', files[z])) + ' ') for z in range(len(files))]
    hcopy = 'HCopy -C '
    config_file = '/home/ferreiraa/Mestrado-PC/github/Conv1D/features/HTK-Tool/config '
    for x in range(len(raw_audio_list)):
        wav_file = wav_audio_list[x]
        mfc_file = mfc_audio_list[x]
        command = hcopy + config_file + wav_file + mfc_file
        os.system(command)
    return None


PATH = '/home/ferreiraa/dataSet/audio/agender_distribution/'
train_file_txt = 'trainSampleList_train.txt'
devel_file_txt = 'trainSampleList_devel.txt'

file_tr = pd.read_table(str(PATH + train_file_txt), delimiter=' ', header=None)
file_list_tr = file_tr[0]
file_dev = pd.read_table(str(PATH + devel_file_txt), delimiter=' ', header=None)
file_list_dev = file_dev[0]

wav2mfc(PATH, file_list_tr)
wav2mfc(PATH, file_list_dev)