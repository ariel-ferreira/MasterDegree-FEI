import itertools
import numpy as np
import os
import pandas as pd
import re
import pathlib
import csv
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import precision_score


DATASET_ROOT = os.path.join(os.path.expanduser("~"),
                            'dataSet/audio/agender_distribution/')
VALID_SPLIT = 0.1
SHUFFLE_SEED = 43
BATCH_SIZE = 128
EPOCHS = 100


def paths_and_labels_to_dataset(mfc_paths, labels):
    """Constructs a dataset of audios (mfc coeff.) and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(mfc_paths)
    mfc_ds = path_ds.map(lambda x: path_to_coeff(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((mfc_ds, label_ds))


def path_to_coeff(path):
    tensor = tf.io.read_file(path)
    # tensor = tf.io.decode_csv(tensor, record_defaults=RECORD_DEFAULTS, field_delim=' ')
    return tensor


train_file_list = pd.read_csv(
    '/home/ferreiraa/Mestrado-PC/github/Conv1D/CNN/file_lists/train_database_normalized.csv')
train_audio_files = train_file_list['file']
train_classes = train_file_list['class']
train_audio_df = pd.DataFrame(train_audio_files)
train_class_df = pd.DataFrame(train_classes)

train_class_labels = list(train_classes.unique())

audio_paths = []
labels = []

for label, category in enumerate(train_class_labels[:20]):
    print("Processing category {}".format(category,))
    speaker_sample_paths = [os.path.join(DATASET_ROOT, train_audio_files[i])
                            for i in range(len(train_audio_files))
                            if train_classes[i] == category]
    audio_paths += speaker_sample_paths
    labels += [category - 1] * len(speaker_sample_paths)

rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(labels)

# train_ds = paths_and_labels_to_dataset(audio_paths, labels)

