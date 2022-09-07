import numpy as np
import pandas as pd
import re
import os

DATASET_ROOT = os.path.join(os.path.expanduser("~"),
                            'dataSet/audio/agender_distribution/')
NETWORK_ROOT = os.path.join(os.path.expanduser("~"),
                            'Mestrado-PC/github/Conv1D/CNN/')
QTD_VEC = 100
ARRAY_SIZE = QTD_VEC * 39

train_file_list_path = 'file_lists/train_database_normalized.csv'
devel_file_list_path = 'file_lists/test_database_normalized.csv'

train_read_csv = pd.read_csv(os.path.join(NETWORK_ROOT, train_file_list_path))
devel_read_csv = pd.read_csv(os.path.join(NETWORK_ROOT, devel_file_list_path))

train_csv_files = train_read_csv['file']
devel_csv_files = devel_read_csv['file']

train_csv_df = pd.DataFrame(train_csv_files)
devel_csv_df = pd.DataFrame(devel_csv_files)

for i in range(len(train_csv_df.index)):
    mfc_file = train_csv_df.iat[i,0]
    mfc_read = pd.read_csv(DATASET_ROOT+mfc_file, delimiter=' ', nrows=QTD_VEC, header=None)
    mfc_array = mfc_read.to_numpy()
    rename_file = re.sub(r'.mfc.csv', '.npy', mfc_file)
    with open(DATASET_ROOT+rename_file, 'wb') as f:
        np.save(f, mfc_array)

for j in range(len(devel_csv_df.index)):
    mfc_file = devel_csv_df.iat[j,0]
    mfc_read = pd.read_csv(DATASET_ROOT+mfc_file, delimiter=' ', nrows=QTD_VEC, header=None)
    mfc_array = mfc_read.to_numpy()
    rename_file = re.sub(r'.mfc.csv', '.npy', mfc_file)
    with open(DATASET_ROOT+rename_file, 'wb') as f:
        np.save(f, mfc_array)