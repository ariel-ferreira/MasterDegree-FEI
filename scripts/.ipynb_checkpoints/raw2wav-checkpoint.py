import wave
import pandas as pd
import re


def raw2wav(PATH, files):
    raw_audio_list = [(PATH + files[j]) for j in range(len(files))]
    wav_audio_list = [(PATH + (re.sub(r'.raw', '.wav', files[i]))) for i in range(len(files))]
    for x in range(len(raw_audio_list)):
        fin = raw_audio_list[x]
        fout = wav_audio_list[x]
        with open(fin, 'rb') as inp_f:
            data = inp_f.read()
            with wave.open(fout, 'wb') as out_f:
                out_f.setnchannels(1)  # mono
                out_f.setsampwidth(2)  # number of bytes
                out_f.setframerate(8000)  # Hz
                out_f.writeframesraw(data)
    return None


PATH = '/home/ferreiraa/Documents/Mestrado/agender_distribution/'
train_file_txt = 'trainSampleList_train.txt'
devel_file_txt = 'trainSampleList_devel.txt'

file_tr = pd.read_table(str(PATH + train_file_txt), delimiter=' ', header=None)
file_list_tr = file_tr[0]
file_dev = pd.read_table(str(PATH + devel_file_txt), delimiter=' ', header=None)
file_list_dev = file_dev[0]

raw2wav(PATH, file_list_tr)
raw2wav(PATH, file_list_dev)
