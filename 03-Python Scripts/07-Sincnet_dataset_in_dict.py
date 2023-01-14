import numpy as np
import pandas as pd
import re
import os

SEED = 25
TEST_SPLIT = 0.75 # 75%

#DATASET_ROOT = os.path.join(os.path.expanduser("~"), 'dataSet/audio/agender_distribution/')
#NETWORK_ROOT = os.path.join(os.path.expanduser("~"), 'Mestrado-PC/github/Conv1D/')

#train_file_list_path = 'CNN/file_lists/7-classes/train_database_full.csv'
#devel_file_list_path = 'CNN/file_lists/7-classes/test_database_full.csv'
train_file_list_path = 'C:\Mestrado\/files\/3\/train_database_full.csv'
devel_file_list_path = 'C:\Mestrado\/files\/3\/test_database_full.csv'

def Rename(file):
    x = re.sub('.wav', '-n.wav', file)
    return x

def MenosUm(y):
    z = int(y)-1
    return z

def Strip(path):
    # print(path)
    return path.replace(" ", "")

#train_df = pd.read_csv(os.path.join(NETWORK_ROOT, train_file_list_path))
#test_df = pd.read_csv(os.path.join(NETWORK_ROOT, devel_file_list_path))

train_df = pd.read_csv(train_file_list_path)
test_df = pd.read_csv(devel_file_list_path)

# train_dict = pd.read_csv(os.path.join(NETWORK_ROOT, train_file_list_path),header=None, index_col=0, squeeze=True).to_dict()
# devel_dict = pd.read_csv(os.path.join(NETWORK_ROOT, devel_file_list_path), header=None, index_col=0, squeeze=True, skipfooter=14999).to_dict()

# To shuffle rows
# files_train = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
# files_test = test_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# To sort rows
files_train = train_df.sort_values(by=['class'], ascending=True, ignore_index=True, kind='mergesort')
files_test = test_df.sort_values(by=['class'], ascending=True, ignore_index=True, kind='mergesort')

'''
# For splitting train database into two groups: train and validation

# Split test_df into validation and test sub-datasets
mask = np.random.rand(len(files_train)) < 0.75
valid_df_split = files_train[~mask]
train_df_split = files_train[mask]

qtd_files_training = len(train_df_split.index)
qtd_files_validation = len(valid_df_split.index)
qtd_files_testing = len(files_test.index)

print("Using {} files for training.".format(qtd_files_training))
print("Using {} files for validation.".format(qtd_files_validation))
print("Using {} files for testing.".format(qtd_files_testing))

train_df_split = train_df_split.rename(columns={0: 'file', 1: 'class'})
valid_df_split = valid_df_split.rename(columns={0: 'file', 1: 'class'})
files_test = files_test.rename(columns={0: 'file', 1: 'class'})

train_dict = train_df_split.set_index('file').to_dict()['class']
val_dict = valid_df_split.set_index('file').to_dict()['class']

try:
    del train_dict['file']
except:
    None
try:
    del val_dict['file']
except:
    None

train_dict_renamed = {Rename(x): y for (x, y) in train_dict.items()}
valid_dict_renamed = {Rename(x): y for (x, y) in val_dict.items()}

train_dict_renamed = {x: int(y) - 1 for (x, y) in train_dict_renamed.items()}
valid_dict_renamed = {x: int(y) - 1 for (x, y) in valid_dict_renamed.items()}

all_labels = train_dict_renamed.copy()
all_labels.update(valid_dict_renamed)

print("Label array contains {} files.".format(len(all_labels)))

train_files = train_df_split['file']
valid_files = valid_df_split['file']

train_files = train_files.apply(lambda x: Rename(x))
train_files_string = train_files.to_string(index = False).replace(" ", "")
valid_files = valid_files.apply(lambda x: Rename(x))
valid_files_string = valid_files.to_string(index = False).replace(" ", "")

all_labels_array = np.array(all_labels)

np.save(os.path.join(NETWORK_ROOT, 'SincNet/file_lists/7_classes/norm_all_labels_array.npy'), all_labels_array)

with open(os.path.join(NETWORK_ROOT, 'SincNet/file_lists/7_classes/norm_train_files.scp'), 'w') as output:
    for row in train_files_string:
        output.write(str(row))

with open(os.path.join(NETWORK_ROOT, 'SincNet/file_lists/7_classes/norm_valid_files.scp'), 'w') as output:
    for row in valid_files_string:
        output.write(str(row))

# For having complete lists based on train and test databases
'''
qtd_files_training = len(files_train.index)
qtd_files_testing = len(files_test.index)

print("Using {} files for training.".format(qtd_files_training))
print("Using {} files for validation.".format(qtd_files_testing))

files_train = files_train.rename(columns={0: 'file', 1: 'class'})
files_test = files_test.rename(columns={0: 'file', 1: 'class'})

#files_train['file'] = files_train['file'].apply(lambda x: Rename(x))
#files_test['file'] = files_test['file'].apply(lambda x: Rename(x))

train_dict = files_train.set_index('file').to_dict()['class']
val_dict = files_test.set_index('file').to_dict()['class']

try:
    del train_dict['file']
except:
    None
try:
    del val_dict['file']
except:
    None

train_dict = {x: int(y) - 1 for (x, y) in train_dict.items()}
val_dict = {x: int(y) - 1 for (x, y) in val_dict.items()}

all_labels = train_dict.copy()
all_labels.update(val_dict)

print("Label array contains {} files.".format(len(all_labels)))

np.save('C:\Mestrado\/files\/3\/all_labels.npy', all_labels)
#np.save(os.path.join(NETWORK_ROOT, 'SincNet/file_lists/7_classes/norm/norm_all_labels_array.npy'), all_labels_array)

exit(0)

train_files = files_train['file']
valid_files = files_test['file']

train_files_string = train_files.to_string(index = False).replace(" ", "")
valid_files_string = valid_files.to_string(index = False).replace(" ", "")

all_labels_array = np.array(all_labels)

with open(os.path.join(NETWORK_ROOT, 'SincNet/file_lists/7_classes/norm/norm_train_files.scp'), 'w') as output:
    for row in train_files_string:
        output.write(str(row))

with open(os.path.join(NETWORK_ROOT, 'SincNet/file_lists/7_classes/norm/norm_valid_files.scp'), 'w') as output:
    for row in valid_files_string:
        output.write(str(row))
