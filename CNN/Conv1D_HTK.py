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


def path_to_coeff(path, NORM):
    tensor = pd.read_csv(path, sep=' ', header=None, nrows=NORM)
    tensor = tf.convert_to_tensor(tensor)
    return tensor


def paths_and_labels_to_dataset(mfc_paths, labels, NORM):
    """Constructs a dataset of audios (mfc coeff.) and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(mfc_paths)
    mfc_ds = path_ds.map(lambda x: path_to_coeff(x, NORM))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((mfc_ds, label_ds))


# Read train files and split class from file

train_file_list = pd.read_csv(
    '/home/ariel/github/Conv1D/CNN/file_lists/train_database_full.csv')
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


def readCSV(file):
    tensor = pd.read_csv(file, sep=' ', header=None, nrows=169)
    print(tensor)
    d = tensor.values
    print(d)
    print(type(d))
    x = tf.convert_to_tensor(d)
    print(x)
    print(type(x))
    return x


path_ds = pd.DataFrame(train_mfc_paths[:1])
mfc = path_ds.applymap(lambda x: readCSV(x))
print(mfc)
print(type(mfc))

exit(0)

# Create 2 datasets, one for training and the other for validation

train_ds = paths_and_labels_to_dataset(
    train_mfc_paths[:1], train_labels[:1], NORM)
valid_ds = paths_and_labels_to_dataset(
    valid_mfc_paths[:1], valid_labels[:1], NORM)

train_ds = train_ds.shuffle(
    buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE)
valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)

train_ds = train_ds.map(lambda x, y: (audio_to_fft(x), y),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
valid_ds = valid_ds.map(lambda x, y: (audio_to_fft(x), y),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)
