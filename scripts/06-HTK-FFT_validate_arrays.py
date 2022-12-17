import numpy as np
import pandas as pd
import re
import os

DATASET_ROOT = os.path.join(os.path.expanduser("~"),
                            'dataSet/audio/agender_distribution/')
NETWORK_ROOT = os.path.join(os.path.expanduser("~"),
                            'Mestrado-PC/github/Conv1D/CNN/')
QTD_VEC = 30
ARRAY_SIZE = QTD_VEC * 39

train_file_list_path = 'file_lists/HTK-FFT/train_database_normalized.csv'
devel_file_list_path = 'file_lists/HTK-FFT/test_database_normalized.csv'

train_read_csv = pd.read_csv(os.path.join(NETWORK_ROOT, train_file_list_path))
devel_read_csv = pd.read_csv(os.path.join(NETWORK_ROOT, devel_file_list_path))

train_files = train_read_csv['file'].tolist()
devel_files = devel_read_csv['file'].tolist()

for i in range(len(train_files)):
    train_files[i] = re.sub('.mfc.csv', '.npy', DATASET_ROOT+train_files[i])

for i in range(len(devel_files)):
    devel_files[i] = re.sub('.mfc.csv', '.npy', DATASET_ROOT+devel_files[i])

train_element_OK = 0
train_element_NOK = 0
train_azero = 0

for i in range(len(train_files)):
    array_content_train = np.load(train_files[i])
    ashape = array_content_train.shape
    asize = array_content_train.size
    atype = array_content_train.dtype
    if ashape == (200,39) and asize == ARRAY_SIZE and atype == 'float64':
        train_element_OK += 1
    else:
        train_element_NOK += 1
    nonzero_size = array_content_train[np.nonzero(array_content_train)].size
    if nonzero_size != asize:
        train_azero += 1
    else:
        None
    
devel_element_OK = 0
devel_element_NOK = 0
devel_azero = 0

for i in range(len(devel_files)):
    array_content_devel = np.load(devel_files[i])
    ashape = array_content_devel.shape
    asize = array_content_devel.size
    atype = array_content_devel.dtype
    if ashape == (200,39) and asize == ARRAY_SIZE and atype == 'float64':
        devel_element_OK += 1
    else:
        devel_element_NOK += 1
    nonzero_size = array_content_devel[np.nonzero(array_content_devel)].size
    if nonzero_size != asize:
        devel_azero += 1
    else:
        None

zero_dataset_percentage_train = (train_azero / len(train_files)) * 100
zero_dataset_percentage_devel = (devel_azero / len(devel_files)) * 100

print("Qtd. matrizes OK (treino): " + str(train_element_OK) +'\n')
print("Qtd. matrizes NOK (treino): " + str(train_element_NOK) +'\n')
print("Qtd. matrizes com muitos zeros (treino): " + str(train_azero) +'\n')
print("Perc. de matrizes com muitos zeros (treino): " + str(zero_dataset_percentage_train) +'\n')
print("Qtd. matrizes OK (devel): " + str(devel_element_OK) +'\n')
print("Qtd. matrizes NOK (devel): " + str(devel_element_NOK) +'\n')
print("Qtd. matrizes com muitos zeros (devel): " + str(devel_azero) +'\n')
print("Perc. de matrizes com muitos zeros (treino): " + str(zero_dataset_percentage_devel) +'\n')