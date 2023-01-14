import csv
import os
import pandas as pd
import numpy as np


DATASET_ROOT = os.path.join(os.path.expanduser("~"),'dataSet/audio/agender_distribution/')
NETWORK_ROOT = os.path.join(os.path.expanduser("~"),'Mestrado-PC/github/Conv1D/CNN/')

SEED = 47

train_file_list_path = 'file_lists/3-classes/train_database_norm_sorted.csv'
test_file_list_path = 'file_lists/3-classes/test_database_norm_sorted.csv'

train_file_list = pd.read_csv(os.path.join(NETWORK_ROOT, train_file_list_path))
test_file_list = pd.read_csv(os.path.join(NETWORK_ROOT, test_file_list_path))

# Sort audio files by labels and save the new file list as csv
# ascending [1 to 7]
train_file_sorted = train_file_list.sort_values(by=['class'], ascending=True, ignore_index=True, kind='mergesort')
test_file_sorted = test_file_list.sort_values(by=['class'], ascending=True, ignore_index=True, kind='mergesort')

train_file_sorted.to_csv(os.path.join(NETWORK_ROOT, 'file_lists/3-classes/train_database_norm_full.csv'), index=False)
test_file_sorted.to_csv(os.path.join(NETWORK_ROOT, 'file_lists/3-classes/test_database_norm_full.csv'), index=False)

exit(0)

# Shuffle audio files and save the new file list as csv
train_df_shuffled = train_file_sorted.sample(frac=1, random_state=SEED).reset_index(drop=True)
test_df_shuffled = test_file_sorted.sample(frac=1, random_state=SEED).reset_index(drop=True)

train_df_shuffled.to_csv(os.path.join(NETWORK_ROOT, 'file_lists/HTK-FFT/train_database_shuffled2.csv'), index=False)
test_df_shuffled.to_csv(os.path.join(NETWORK_ROOT, 'file_lists/HTK-FFT/test_database_shuffled2.csv'), index=False)

'''# Split test_df into validation and test sub-datasets
mask = np.random.rand(len(test_df_shuffled)) < 0.75
print(mask)
test_sub_dataset_shuffled = test_df_shuffled[~mask]
valid_sub_dataset_shuffled = test_df_shuffled[mask]

valid_sub_dataset_shuffled.to_csv(os.path.join(NETWORK_ROOT, 'file_lists/normalizados/valid_sub_database_shuffled.csv'), index=False)
test_sub_dataset_shuffled.to_csv(os.path.join(NETWORK_ROOT, 'file_lists/normalizados/test_sub_database_shuffled.csv'), index=False)
'''