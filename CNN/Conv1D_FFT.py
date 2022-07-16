#!/usr/bin/env python
# coding: utf-8

import itertools
import numpy as np
import os
import pandas as pd
# import re
import tensorflow as tf
from tensorflow import keras
# from pathlib import Path
# from IPython.display import display, Audio
from keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import precision_score

DATASET_ROOT = os.path.join(
    os.path.expanduser("~"), 'dataSet/audio/agender_distribution/')
VALID_SPLIT = 0.1
SAMPLING_RATE = 8000
SHUFFLE_SEED = 43
BATCH_SIZE = 128
EPOCHS = 100


def paths_and_labels_to_dataset(audio_paths, labels):
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


def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64
                ))
    fft = tf.expand_dims(fft, axis=-1)
    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


# Read train files and split class from file

train_file_list = pd.read_csv(
    '/home/ferreiraa/Mestrado/github/Conv1D/CNN/file_lists/train_database_full.csv')
train_audio_files = train_file_list['file']
train_classes = train_file_list['class']
train_audio_df = pd.DataFrame(train_audio_files)
train_class_df = pd.DataFrame(train_classes)

# In the next section, get the list of audio file paths along with their
# corresponding labels

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
    labels += [category - 1] * len(speaker_sample_paths)

print("Found {} files belonging to {} classes.".format(
    len(audio_paths), len(train_class_labels)))

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

# Create 2 datasets, one for training and the other for validation

train_ds = paths_and_labels_to_dataset(train_audio_paths, train_labels)
valid_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)

train_ds = train_ds.shuffle(
    buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE)
valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)

# Transform audio wave to the frequency domain

train_ds = train_ds.map(lambda x, y: (audio_to_fft(x), y),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
valid_ds = valid_ds.map(lambda x, y: (audio_to_fft(x), y),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)

# MODEL DEFINITION


def residual_block(x, filters, conv_num=3, activation="relu"):
    # Shortcut
    s = keras.layers.Conv1D(filters, 1, padding="same")(x)
    for i in range(conv_num - 1):
        x = keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = keras.layers.Activation(activation)(x)
    x = keras.layers.Conv1D(filters, 3, padding="same")(x)
    x = keras.layers.Add()([x, s])
    x = keras.layers.Activation(activation)(x)
    return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)


def build_model(input_shape, num_classes):
    inputs = keras.layers.Input(shape=input_shape, name="input")

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)

    x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)

    outputs = keras.layers.Dense(
        num_classes, activation="softmax", name="output")(x)

    return keras.models.Model(inputs=inputs, outputs=outputs)


model = build_model((SAMPLING_RATE // 2, 1), len(train_class_labels))

model.summary()

# Compile the model using Adam's default learning rate

model.compile(optimizer="Adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Add callbacks:
# 'EarlyStopping' to stop training when the model is not enhancing anymore
# 'ModelCheckPoint' to always keep the model that has the best val_accuracy

model_save_filename = '/home/ariel/github/Conv1D/CNN/CNN_1D_model.h5'
earlystopping_cb = keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True)
mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
    model_save_filename, monitor="val_accuracy", save_best_only=True)

# # TRAINING

history = model.fit(train_ds, epochs=EPOCHS, validation_data=valid_ds,
                    callbacks=[earlystopping_cb, mdlcheckpoint_cb])

print(model.evaluate(valid_ds))


# # DEMONSTRATION

# Loading model trained previously

# model = load_model('CNN_model_7_class.h5')

# Read test files and split class from file

test_file_list = pd.read_csv(
    '/home/ariel/github/Conv1D/CNN/file_lists/test_database_full.csv')
test_audio_files = test_file_list['file']
test_classes = test_file_list['class']
test_audio_df = pd.DataFrame(test_audio_files)
test_class_df = pd.DataFrame(test_classes)

# Get the labels of test data

test_class_labels = list(test_classes.unique())

print("Age categories identified: {}".format(test_class_labels,))

# Get the list of test audio file paths along with their corresponding labels

test_audio_paths = []
test_labels = []

for label, category in enumerate(test_class_labels):
    print("Processing category {}".format(category,))
    speaker_sample_paths = [os.path.join(DATASET_ROOT, test_audio_files[i])
                            for i in range(len(test_audio_files))
                            if test_classes[i] == category]
    test_audio_paths += speaker_sample_paths
    test_labels += [category - 1] * len(speaker_sample_paths)

print("Found {} files belonging to {} classes.".format(
    len(test_audio_paths), len(test_class_labels)))


rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(test_audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(test_labels)

# Creating the test dataset

test_ds = paths_and_labels_to_dataset(test_audio_paths, test_labels)

test_ds = test_ds.shuffle(buffer_size=BATCH_SIZE * 8,
                          seed=SHUFFLE_SEED).batch(BATCH_SIZE)

test_ds = test_ds.map(lambda x, y: (audio_to_fft(x), y),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

# REVISAR!

audios_list = []
labels_list = []
y_pred_list = []

for j in range(len(test_ds)):
    for audios, labels in test_ds.take(j):
        # Get the signal FFT
        ffts = audio_to_fft(audios)
        # Predict
        y_pred = model.predict(ffts)
        audios = audios.numpy()
        labels = labels.numpy()
        y_pred = np.argmax(y_pred, axis=-1)
        audios_list.append(audios)
        labels_list.append(labels)
        y_pred_list.append(y_pred)

flatten_labels_list = list(itertools.chain(*labels_list))
flatten_y_pred_list = list(itertools.chain(*y_pred_list))

real_output = flatten_labels_list
predicted_output = flatten_y_pred_list

# Metrics

print(accuracy_score(real_output, predicted_output))

print(f1_score(real_output, predicted_output, average='macro'))
print(f1_score(real_output, predicted_output, average='micro'))

print(precision_score(real_output, predicted_output, average='macro'))
print(precision_score(real_output, predicted_output, average='micro'))

print(recall_score(real_output, predicted_output, average='macro'))
print(recall_score(real_output, predicted_output, average='micro'))

exit(0)

'''y_pred_list = []
for i in range(1, len(test_ds)):
    for audios, labels in test_ds.take(i):
        # Get the signal FFT
        ffts = audio_to_fft(audios)
        # Predict
        y_pred = model.predict(ffts)
    y_pred_list.append(y_pred)'''

'''audios_list = []
labels_list = []

for w in range(1, len(test_ds)):
    for audios, labels in test_ds.take(w):
        audios = audios.numpy()
        labels = labels.numpy()
    audios_list.append(audios)
    labels_list.append(labels)
flatten_labels_list = list(itertools.chain(*labels_list))'''

'''y_pred_transf = []
for i in range(len(y_pred_list)):
    for j in range(len(y_pred)):
        y_pred_label = np.argmax(y_pred_list[i][j], axis=-1)
        y_pred_transf.append(y_pred_label)'''
