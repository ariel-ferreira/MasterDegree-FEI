import numpy as np
import pandas as pd
import re
import os
import collections

DATASET_ROOT = os.path.join(os.path.expanduser("~"),
                            'dataSet/audio/agender_distribution/')
NETWORK_ROOT = os.path.join(os.path.expanduser("~"),
                            'Mestrado-PC/github/Conv1D/CNN/')

train_file_list_path = 'file_lists/train_database_normalized.csv'
devel_file_list_path = 'file_lists/test_database_normalized.csv'


def Rename(file):
    x = re.sub('.mfc.csv', '.wav', file)
    return x

def MenosUm(y):
    z = int(y)-1
    return z


train_dict = pd.read_csv(os.path.join(NETWORK_ROOT, train_file_list_path), header=None, index_col=0, squeeze=True).to_dict()
devel_dict = pd.read_csv(os.path.join(NETWORK_ROOT, devel_file_list_path), header=None, index_col=0, squeeze=True).to_dict()

del train_dict['file']
del devel_dict['file']

train_dict_renamed = {Rename(x): y for (x, y) in train_dict.items()}
devel_dict_renamed = {Rename(x): y for (x, y) in devel_dict.items()}

train_dict_renamed = {x: int(y) - 1 for (x, y) in train_dict_renamed.items()}
devel_dict_renamed = {x: int(y) - 1 for (x, y) in devel_dict_renamed.items()}

all_labels_temp = collections.Counter(train_dict_renamed)
all_labels_temp.update(devel_dict_renamed)
all_labels = dict(all_labels_temp)

train_dict_array = np.array(train_dict_renamed)
devel_dict_array = np.array(devel_dict_renamed)
all_labels_array = np.array(all_labels)

np.save(os.path.join(NETWORK_ROOT, 'file_lists/train_dict_array.npy'), train_dict_array)
np.save(os.path.join(NETWORK_ROOT, 'file_lists/devel_dict_array.npy'), devel_dict_array)
np.save(os.path.join(NETWORK_ROOT, 'file_lists/all_labels_array.npy'), all_labels_array)