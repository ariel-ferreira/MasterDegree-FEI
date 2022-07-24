import numpy as np
import csv
import pandas as pd
import re

DIR = '/home/ferreiraa/dataSet/audio/agender_distribution/'

train_read_csv = pd.read_csv(
    '/home/ferreiraa/Mestrado-PC/github/Conv1D/CNN/file_lists/train_database_normalized.csv')
devel_read_csv = pd.read_csv(
    '/home/ferreiraa/Mestrado-PC/github/Conv1D/CNN/file_lists/test_database_normalized.csv')

train_csv_files = train_read_csv['file']
devel_csv_files = devel_read_csv['file']

train_csv_df = pd.DataFrame(train_csv_files)
devel_csv_df = pd.DataFrame(devel_csv_files)

train_array_ok = 0
train_array_nok = 0

for i in range(len(train_csv_df.index)):
    mfc_file = train_csv_df.iat[i,0]
    mfc_read = pd.read_csv(DIR+mfc_file, delimiter=' ', nrows=169, header=None)
    mfc_array = mfc_read.to_numpy()
    elements = mfc_array.size
    if elements == 6591:
        train_array_ok += 1
    else:
        train_array_nok += 1
    rename_file = re.sub(r'.mfc.csv', '.npy', mfc_file)
    with open(DIR+rename_file, 'wb') as f:
        np.save(f, mfc_array)

devel_array_ok = 0
devel_array_nok = 0

for j in range(len(devel_csv_df.index)):
    mfc_file = devel_csv_df.iat[j,0]
    mfc_read = pd.read_csv(DIR+mfc_file, delimiter=' ', nrows=169, header=None)
    mfc_array = mfc_read.to_numpy()
    elements = mfc_array.size
    if elements == 6591:
        devel_array_ok += 1
    else:
        devel_array_nok += 1
    rename_file = re.sub(r'.mfc.csv', '.npy', mfc_file)
    with open(DIR+rename_file, 'wb') as f:
        np.save(f, mfc_array)

print(train_array_ok)
print(train_array_nok)
print(devel_array_ok)
print(devel_array_nok)