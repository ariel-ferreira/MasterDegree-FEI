import os
import pandas as pd
from scipy.io.wavfile import read


def getMetrics(x):
    fi = DATASET_ROOT + x
    samplerate, data = read(fi)
    tt = len(data)/samplerate
    ts = os.path.getsize(fi)
    return (tt, ts)

total_time = 0
total_size = 0

DATASET_ROOT = os.path.join(os.path.expanduser("~"),'dataSet/audio/agender_distribution/')
NETWORK_ROOT = os.path.join(os.path.expanduser("~"),'Mestrado-PC/github/Conv1D/CNN/')

# train_file_list_path = 'file_lists/train_database_full.csv'
# test_file_list_path = 'file_lists/test_database_full.csv'

train_file_list_path = 'file_lists/train_database_norm_full.csv'
test_file_list_path = 'file_lists/test_database_norm_full.csv'

train_file_list = pd.read_csv(os.path.join(NETWORK_ROOT, train_file_list_path))
train_audio_files = train_file_list['file']

test_file_list = pd.read_csv(os.path.join(NETWORK_ROOT, test_file_list_path))
test_audio_files = test_file_list['file']

for i in range(len(train_audio_files.values)):
    file = train_audio_files.iloc[i]
    tt, ts = getMetrics(file)
    total_time = total_time + tt
    total_size = total_size + ts

print("total time (min): {}".format(total_time/60))
print("total time (hrs): {}".format(total_time/3600))
print("total size (MB): {}".format(total_size*9.537e-7))
print("total size (GB): {}".format(total_size*9.31e-10))