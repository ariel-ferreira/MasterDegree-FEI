import os
import pandas as pd
import re
import numpy as np
import csv


def csv_list(DIR, files):
    raw_audio_list = [(DIR + files[j]) for j in range(len(files))]
    csv_list = [(DIR + (re.sub(r'.raw', '.mfc.csv', files[i]))) for i in range(len(files))]
    return csv_list


def ReadCSV(file):
    fin = file
    fout = pd.read_csv(fin, header=None)
    return fout


def NumberVectors(file_list):
    min_sample = 0
    c = 0
    list_vec_qtd = []
    for i in range(len(file_list)):
        file = file_list[i]
        df = ReadCSV(file)
        c += 1
        num_vectors = len(df.index)
        num_samples = num_vectors
        list_vec_qtd.append(num_vectors)
        if min_sample == 0:
            min_sample = num_samples
        elif num_samples < min_sample:
            min_sample = num_samples
            min_file = file_list[i]
        else:
            None
    return (min_sample, min_file, c, list_vec_qtd)


fields = ['file'] 

DIR = os.path.join(os.path.expanduser("~"),'dataSet/audio/agender_distribution/')

train_file_txt = 'trainSampleList_train.txt'
devel_file_txt = 'trainSampleList_devel.txt'

file_tr = pd.read_table(str(DIR + train_file_txt), delimiter=' ', header=None)
file_dev = pd.read_table(str(DIR + devel_file_txt), delimiter=' ', header=None)

raw_file_TRlist = file_tr[0]
raw_file_DEVlist = file_dev[0]

csv_list_train = csv_list(DIR, raw_file_TRlist)
csv_list_dev = csv_list(DIR, raw_file_DEVlist)

min_sample_train, min_file_train, counter_check_train, list_vec_qtd_train = NumberVectors(csv_list_train)
min_sample_dev, min_file_dev, counter_check_dev, list_vec_qtd_dev = NumberVectors(csv_list_dev)

# DataFrames que mostram a quantidade de vetores (39 elementos cada) em cada arquivo de áudio.
SampleListDataFrame_train = pd.DataFrame(list(zip(csv_list_train, list_vec_qtd_train)), columns=['Train Files', 'Qtd. of Vectors'])
SampleListDataFrame_dev = pd.DataFrame(list(zip(csv_list_dev, list_vec_qtd_dev)), columns=['Dev Files', 'Qtd. of Vectors'])

# Agrupamento de arquivos por quantidade de vetores.
counter_train = SampleListDataFrame_train['Qtd. of Vectors'].value_counts()
counter_dev = SampleListDataFrame_dev['Qtd. of Vectors'].value_counts()

# Criação das listas de arquivos que serão utilizados após a normalização

threshold_train = 169
files_to_use_train = 0
for k in range(len(counter_train)):
    row_index = counter_train.index[k]
    row_value = counter_train.iat[k]
    if row_index >= threshold_train:
        files_to_use_train = files_to_use_train + row_value

threshold_dev = 169
files_to_use_dev = 0
for k in range(len(counter_dev)):
    row_index = counter_dev.index[k]
    row_value = counter_dev.iat[k]
    if row_index >= threshold_dev:
        files_to_use_dev = files_to_use_dev + row_value

train_csv_use = []
train_not_used = []
for i in range(len(SampleListDataFrame_train.index)):
    train_vectors = SampleListDataFrame_train.iat[i,1]
    if train_vectors >= threshold_train:
        train_csv_use.append(SampleListDataFrame_train.iat[i,0])
    else:
        train_not_used.append(SampleListDataFrame_train.iat[i,0])

with open('/home/ferreiraa/Mestrado-PC/github/Conv1D/CNN/file_lists/train_database_normalized.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(fields)
    # write.writerows(train_csv_use)
    for row in train_csv_use:
        columns1 = [c.strip() for c in row.strip(', ').split(',')]
        write.writerow(columns1)

dev_csv_use = []
dev_not_used = []
for i in range(len(SampleListDataFrame_dev.index)):
    dev_vectors = SampleListDataFrame_dev.iat[i,1]
    if dev_vectors >= threshold_dev:
        dev_csv_use.append(SampleListDataFrame_dev.iat[i,0])
    else:
        dev_not_used.append(SampleListDataFrame_dev.iat[i,0])

with open('/home/ferreiraa/Mestrado-PC/github/Conv1D/CNN/file_lists/test_database_normalized.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(fields)
    # write.writerows(dev_csv_use)
    for row in dev_csv_use:
        columns2 = [c.strip() for c in row.strip(', ').split(',')]
        write.writerow(columns2)