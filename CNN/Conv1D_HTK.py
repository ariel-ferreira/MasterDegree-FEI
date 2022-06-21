import itertools
import numpy as np
import os
import pandas as pd
import re
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score, confusion_matrix)

DATASET_ROOT = os.path.join(os.path.expanduser("~"),
                            'dataSet/audio/agender_distribution/')
VALID_SPLIT = 0.1
SAMPLING_RATE = 8000
SHUFFLE_SEED = 43
BATCH_SIZE = 128
EPOCHS = 100


def FFT_paths_and_labels_to_dataset(audio_paths, labels):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def path_to_audio(path):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio


def path_to_mfc(path, dim):
    csv_file = tf.io.read_file(path, dim)
    tensor = tf.convert_to_tensor(csv_file)
    return tensor


def ReadCSV(file, dim):
    fin = file
    fout = pd.read_csv(fin, dtype=np.float32, header=None, nrows=dim)
    return fout


def HTK_paths_and_labels_to_dataset(audio_paths, labels, dim):
    for i in range(len(audio_paths)):
        audio_paths[i] = re.sub(r'.wav', '.mfc.csv', audio_paths[i])
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_mfc(x, dim))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


# Read train files and split class from file
train_file_list = pd.read_csv(
    '/home/arielferreira/github/Conv1D/CNN/file_lists/train_database_full.csv')
train_audio_files = train_file_list['file']
train_classes = train_file_list['class']
train_audio_df = pd.DataFrame(train_audio_files)
train_class_df = pd.DataFrame(train_classes)

train_class_labels = list(train_classes.unique())
print("Age categories identified: {}".format(train_class_labels,))

audio_paths = []
labels = []

for label, category in enumerate(train_class_labels):
    print("Processing category {}".format(category,))
    speaker_sample_paths = [os.path.join(DATASET_ROOT, train_audio_files[i])
                            for i in range(len(train_audio_files))
                            if train_classes[i] == category]
    audio_paths += speaker_sample_paths
    labels += [label] * len(speaker_sample_paths)

print("Found {} files belonging to {} classes.".format(len(audio_paths),
      len(train_class_labels)))

print(train_classes[:10])

print(audio_paths[:10])
print(labels[:-1])

exit(0)

# Shuffle
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(labels)




# Split into training and validation
num_val_samples = int(VALID_SPLIT * len(audio_paths))
print("Using {} files for training.".format(len(audio_paths) - num_val_samples))
print("Using {} files for validation.".format(num_val_samples))

train_audio_paths = audio_paths[:-num_val_samples]
train_labels = labels[:-num_val_samples]
valid_audio_paths = audio_paths[-num_val_samples:]
valid_labels = labels[-num_val_samples:]

train_ds = FFT_paths_and_labels_to_dataset(train_audio_paths, train_labels)
valid_ds = FFT_paths_and_labels_to_dataset(valid_audio_paths, valid_labels)

train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE)
valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)

train_ds = train_ds.map(lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
valid_ds = valid_ds.map(lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)


# Parâmetros de normalização ao se utilizar as características extraídas com HTK
NORM_TRAIN = 200
NORM_TEST = 239



train_audio_paths_mfc = []
for i in range(len(train_audio_paths)):
    train_audio_paths_mfc.append(re.sub(r'\.wav', '.mfc.csv', train_audio_paths[i]))

# Create 2 datasets, one for training and the other for validation
train_ds = HTK_paths_and_labels_to_dataset(train_audio_paths[:5], train_labels[:5], NORM_TRAIN)
#valid_ds = HTK_paths_and_labels_to_dataset(valid_audio_paths, valid_labels, NORM_TRAIN)

train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE)
valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)








# Read test files and split class from file
test_file_list = pd.read_csv(
    '/home/arielferreira/github/Conv1D/CNN/file_lists/test_database_full.csv')
test_audio_files = test_file_list['file']
test_classes = test_file_list['class']
test_audio_df = pd.DataFrame(test_audio_files)
test_class_df = pd.DataFrame(test_classes)

# print(test_audio_df[:10])
# print(test_class_df[:10])
