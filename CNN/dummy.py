import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras

DATASET_ROOT = os.path.join(
    os.path.expanduser("~"), 'dataSet/audio/agender_distribution/')
VALID_SPLIT = 0.1
SAMPLING_RATE = 8000
BATCH_SIZE = 128


def path_to_audio(path):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio


train_file_list = pd.read_csv(
    '/home/ariel/github/Conv1D/CNN/file_lists/train_database_full.csv')
train_audio_files = train_file_list['file']
train_classes = train_file_list['class']
train_audio_df = pd.DataFrame(train_audio_files)
train_class_df = pd.DataFrame(train_classes)

train_class_labels = list(train_classes.unique())

audio_paths = []
labels = []

for label, category in enumerate(train_class_labels):
    speaker_sample_paths = [os.path.join(DATASET_ROOT, train_audio_files[i])
                            for i in range(len(train_audio_files))
                            if train_classes[i] == category]
    audio_paths += speaker_sample_paths
    labels += [category - 1] * len(speaker_sample_paths)

num_val_samples = int(VALID_SPLIT * len(audio_paths))
train_audio_paths = audio_paths[:-num_val_samples]
train_labels = labels[:-num_val_samples]
valid_audio_paths = audio_paths[-num_val_samples:]
valid_labels = labels[-num_val_samples:]

print(train_audio_paths[0])
print(type(train_audio_paths[0]))

# def paths_and_labels_to_dataset
path_ds = tf.data.Dataset.from_tensor_slices(train_audio_paths[:1])
print(path_ds)
print(list(path_ds.as_numpy_iterator()))
audio_ds = path_ds.map(lambda x: tf.io.read_file(x))
print(audio_ds)
print(list(audio_ds.as_numpy_iterator()))
audio = audio_ds.map(lambda x: tf.audio.decode_wav(x, 1, 8000))
print(audio)
print(list(audio.as_numpy_iterator()))

exit(0)

# audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)

print(list(path_ds.as_numpy_iterator()))
print(audio_ds)
print(list(audio_ds.as_numpy_iterator()))
print(label_ds)
print(list(label_ds.as_numpy_iterator()))
