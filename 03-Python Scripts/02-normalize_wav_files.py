import librosa
import soundfile as sf
import os
import pandas as pd
import numpy as np


def Normal(x):
    tag = '-n.wav'
    [signal, sr] = librosa.load(os.path.join(DATASET_ROOT, x), sr= 8000, mono=True)
    # Remove silences from begining and end of each utterance
    clip = librosa.effects.trim(signal, top_db= 10)
    # Normalize signal amplitude
    signal_norm = clip[0]/np.max(np.abs(clip[0]))
    # Save normalized audio (.wav) file
    output_file = x+tag
    sf.write(os.path.join(DATASET_ROOT, output_file), signal_norm, sr)
    return None


DATASET_ROOT = os.path.join(os.path.expanduser("~"),'dataSet/audio/agender_distribution/')
NETWORK_ROOT = os.path.join(os.path.expanduser("~"),'Mestrado-PC/github/Conv1D/CNN/')

train_file_list_path = 'file_lists/train_database_full.csv'
test_file_list_path = 'file_lists/test_database_full.csv'

train_file_list = pd.read_csv(os.path.join(NETWORK_ROOT, train_file_list_path))
train_audio_files = train_file_list['file']

test_file_list = pd.read_csv(os.path.join(NETWORK_ROOT, test_file_list_path))
test_audio_files = test_file_list['file']

train_audio = train_audio_files.apply(lambda x: Normal(x))
test_audio = test_audio_files.apply(lambda x: Normal(x))