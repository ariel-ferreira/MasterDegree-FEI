import itertools
import numpy as np
import os
import pandas as pd
import re
import tensorflow as tf
# import tensorflow.compat.v1 as tf
from tensorflow import keras
import pathlib
from keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import precision_score

# tf.compat.v1.disable_eager_execution()

DATASET_ROOT = os.path.join(os.path.expanduser("~"),
                            'dataSet/audio/agender_distribution/')
VALID_SPLIT = 0.1
SHUFFLE_SEED = 43
BATCH_SIZE = 128
EPOCHS = 100
NORM = 169
RECORD_DEFAULTS = [float()]*39

def paths_and_labels_to_dataset(mfc_paths, labels):
    """Constructs a dataset of audios (mfc coeff.) and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(mfc_paths)
    mfc_ds = path_ds.map(lambda x: path_to_coeff(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((mfc_ds, label_ds))


def path_to_coeff(path):
    tensor = tf.io.read_file(path)
    tensor = tf.io.decode_csv(tensor, record_defaults=RECORD_DEFAULTS, field_delim=' ')
    return tensor


# Read train files and split class from file

train_file_list = pd.read_csv(
    '/home/ferreiraa/Mestrado/github/Conv1D/CNN/file_lists/train_database_full.csv')
train_audio_files = train_file_list['file']
train_classes = train_file_list['class']
train_audio_df = pd.DataFrame(train_audio_files)
train_class_df = pd.DataFrame(train_classes)

train_class_labels = list(train_classes.unique())
print("Age categories identified: {}".format(train_class_labels,))

audio_paths = []
labels = []

for label, category in enumerate(train_class_labels[:20]):
    print("Processing category {}".format(category,))
    speaker_sample_paths = [os.path.join(DATASET_ROOT, train_audio_files[i])
                            for i in range(len(train_audio_files))
                            if train_classes[i] == category]
    audio_paths += speaker_sample_paths
    labels += [category - 1] * len(speaker_sample_paths)

print(
    "Found {} files belonging to {} classes.".format(len(audio_paths),
                                                     len(train_class_labels)))

# Shuffle

rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(labels)

# Split into training and validation

num_val_samples = int(VALID_SPLIT * len(audio_paths))

print("Using {} files for training.".format(
    len(audio_paths) - num_val_samples))
print("Using {} files for validation.".format(num_val_samples))

train_audio_paths = audio_paths[:-num_val_samples]
train_labels = labels[:-num_val_samples]
valid_audio_paths = audio_paths[-num_val_samples:]
valid_labels = labels[-num_val_samples:]

train_mfc_paths = []
valid_mfc_paths = []

for i in range(len(train_audio_paths)):
    train_mfc_paths.append(
        re.sub(r'\d.wav', '.mfc.csv', train_audio_paths[i]))

for i in range(len(valid_audio_paths)):
    valid_mfc_paths.append(
        re.sub(r'\d.wav', '.mfc.csv', valid_audio_paths[i]))


train_ds = paths_and_labels_to_dataset(train_mfc_paths, train_labels)
valid_ds = paths_and_labels_to_dataset(valid_mfc_paths, valid_labels)

"""trecho = valid_mfc_paths[:5]
path_ds = tf.data.Dataset.from_tensor_slices(trecho)
    
mfc_ds = path_ds.map(lambda x: path_to_coeff(x, NORM))
label_ds = tf.data.Dataset.from_tensor_slices(labels)
return tf.data.Dataset.zip((mfc_ds, label_ds))
tensor = pd.read_csv(path, sep=' ', header=None, nrows=NORM)
tensor = tf.convert_to_tensor(tensor)
return tensor"""