import os
import pandas as pd
import re
import numpy as np
import csv


def Rename(file):
    x = re.sub('.wav', '.mfc.csv', file)
    return x
    

def ReadCSV(file):
    fin = file
    fout = pd.read_csv(fin, header=None)
    return fout


def NumberVectors(file_list):
    DIR = '/home/ferreiraa/dataSet/audio/agender_distribution/'
    min_sample = 0
    c = 0
    list_vec_qtd = []
    list_file_name = []
    for i in range(len(file_list.index)):
        file = file_list.iat[i , 0]
        df = ReadCSV(DIR + file)
        c += 1
        num_vectors = len(df.index)
        num_samples = num_vectors
        list_vec_qtd.append(num_vectors)
        list_file_name.append(file)
        if min_sample == 0:
            min_sample = num_samples
        elif num_samples < min_sample:
            min_sample = num_samples
            min_file = file_list.iat[i , 0]
        else:
            None
    return (min_sample, min_file, c, list_vec_qtd, list_file_name)


MIN_VEC = 169

train_file_list = pd.read_csv(
    '/home/ferreiraa/Mestrado-PC/github/Conv1D/CNN/file_lists/train_database_full.csv')
train_audio_files = train_file_list['file']
train_classes = train_file_list['class']

devel_file_list = pd.read_csv(
    '/home/ferreiraa/Mestrado-PC/github/Conv1D/CNN/file_lists/test_database_full.csv')
devel_audio_files = devel_file_list['file']
devel_classes = devel_file_list['class']

train_audio_df = pd.DataFrame(train_audio_files)
train_class_df = pd.DataFrame(train_classes)

devel_audio_df = pd.DataFrame(devel_audio_files)
devel_class_df = pd.DataFrame(devel_classes)

train_audio_renamed = train_audio_df.applymap(lambda x: Rename(x))
devel_audio_renamed = devel_audio_df.applymap(lambda x: Rename(x))

train_min_samplen, train_min_filename, train_cross_check, train_vec_qtd, train_list_filename = NumberVectors(train_audio_renamed)
devel_min_samplen, devel_min_filename, devel_cross_check, devel_vec_qtd, devel_list_filename = NumberVectors(devel_audio_renamed)

train_SampleList = list(zip(train_list_filename, train_vec_qtd, train_class_df.values.flatten().tolist()))
devel_SampleList = list(zip(devel_list_filename, devel_vec_qtd, devel_class_df.values.flatten().tolist()))

sel_filename = []
sel_class = []

for i in range(len(train_SampleList)):
    qtd_vector = train_SampleList[i][1]
    if qtd_vector >= MIN_VEC:
        sel_filename.append(train_SampleList[i][0])
        sel_class.append(train_SampleList[i][2])
    else:
        None

train_norm_zip_list = list(zip(sel_filename, sel_class))

sel_filename = []
sel_class = []

for i in range(len(devel_SampleList)):
    qtd_vector = devel_SampleList[i][1]
    if qtd_vector >= MIN_VEC:
        sel_filename.append(devel_SampleList[i][0])
        sel_class.append(devel_SampleList[i][2])
    else:
        None

devel_norm_zip_list = list(zip(sel_filename, sel_class))

fields = ['file', 'class']

with open('/home/ferreiraa/Mestrado-PC/github/Conv1D/CNN/file_lists/train_database_normalized.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(train_norm_zip_list)

with open('/home/ferreiraa/Mestrado-PC/github/Conv1D/CNN/file_lists/test_database_normalized.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(devel_norm_zip_list)