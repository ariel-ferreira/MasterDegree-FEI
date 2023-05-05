import datetime
import time
import numpy as np
import os
import pandas as pd
import re
import tensorflow as tf
from tensorflow import keras
import random

DATASET_ROOT = '/home/'
NETWORK_ROOT = '/home/MasterDegree-FEI/00-Convolutional Network/'

VALID_SPLIT = 0.75
SAMPLING_RATE = 8000
SHUFFLE_SEED = 43
BATCH_SIZE = 128
EPOCHS = 100
qtd_class = 4
timestamp = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

train_file_list_path = 'file_lists/4-classes/train_database_full.csv'
test_file_list_path = 'file_lists/4-classes/test_database_full.csv'

def paths_and_labels_to_dataset(audio_paths, labels):
    # Constructs a dataset of audios and labels
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def path_to_audio(path):
    # Reads and decodes an audio file
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio


def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension, we need to squeeze the dimensions and then expand them again after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64))
    fft = tf.expand_dims(fft, axis=-1)
    # Return the absolute value of the first half of the FFT which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


# Read train files and split class from file

train_file_list = pd.read_csv(os.path.join(NETWORK_ROOT, train_file_list_path))
train_audio_files = train_file_list['file']
train_classes = train_file_list['class']

class_labels = list(train_classes.unique())
print("Age categories identified: {}".format(train_classes,))

audio_paths = list(train_audio_files)
labels = list(train_classes)

print("Found {} files belonging to {} classes.".format(len(audio_paths), len(class_labels)))

for i in range(len(audio_paths)):
    audio_paths[i] = os.path.join(DATASET_ROOT, audio_paths[i])

rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(labels)

train_audio_paths = audio_paths
train_labels = labels

# Read train files and split class from file

test_audio_files = test_file_list['file']
test_classes = test_file_list['class']

test_class_labels = list(test_classes.unique())
print("Age categories identified: {}".format(test_classes,))

test_audio_paths = list(test_audio_files)
test_labels = list(test_classes)

print("Found {} files belonging to {} classes.".format(len(test_audio_paths), len(tes_class_labels)))

for i in range(len(test_audio_paths)):
    test_audio_paths[i] = os.path.join(DATASET_ROOT, test_audio_paths[i])

rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(test_audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(test_labels)
    
qtd_files_testing = len(test_audio_paths)

print("Using {} files belonging to {} classes.".format(qtd_files_testing, len(test_class_labels)))

# Split into test and validation

num_val_samples = int(VALID_SPLIT * len(test_audio_paths))

qtd_files_training = len(test_audio_paths) - num_val_samples
qtd_files_validation = num_val_samples

print("Using {} files for training.".format(len(test_audio_paths) - num_val_samples))
print("Using {} files for validation.".format(num_val_samples))

test_audio_paths = test_audio_paths[:-num_val_samples]
test_labels = labels[:-num_val_samples]
valid_audio_paths = test_audio_paths[-num_val_samples:]
valid_labels = labels[-num_val_samples:]

# Create 3 datasets, one for training, one for validation, and one for testing

train_ds = paths_and_labels_to_dataset(train_audio_paths, train_labels)
valid_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
test_ds = paths_and_labels_to_dataset(test_audio_paths, valid_labels)

train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE)
valid_ds = valid_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE)
test_ds = test_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)

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
    # num_classes = num_classes + 1
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

model = build_model((SAMPLING_RATE // 2, 1), len(class_labels))

# model.summary()

# Compile the model using Adam's default learning rate

model.compile(optimizer="Adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Add callbacks:
# 'EarlyStopping' to stop training when the model is not enhancing anymore
# 'ModelCheckPoint' to always keep the model that has the best val_accuracy
# 'Tensorboard' to print logs/metrics from training phase

model_save_filename = os.path.join(NETWORK_ROOT, "simulations/model_FFT_"+str(qtd_class)+"_E"+str(EPOCHS)+"_B"+str(BATCH_SIZE)+"_"+timestamp+"/"+"model.h5")

earlystopping_cb = keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True)
mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
    model_save_filename, monitor="val_accuracy", save_best_only=True)
tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir=NETWORK_ROOT+"simulations/model_FFT_"+str(qtd_class)+"_E"+str(EPOCHS)+"_B"+str(BATCH_SIZE)+"_"+timestamp+"/"+"logs",
    histogram_freq=1,
    write_graph=True,
    write_images=False,
    write_steps_per_second=True,
    update_freq="epoch",
    profile_batch=0,
    embeddings_freq=0,
    embeddings_metadata=None
    )

# TRAINING

print("Início treinamento do modelo")

start_train = time.time()

history = model.fit(train_ds, epochs=EPOCHS, validation_data=valid_ds, callbacks=[earlystopping_cb, mdlcheckpoint_cb, tensorboard_cb])

end_train = (time.time() - start_train) / 60

print("Término treinamento do modelo")
print("Tempo transcorrido: {} min.".format(end_train))

acc_array = np.asarray(list(history.history['accuracy']), dtype=np.float64)
acc_mean = np.mean(acc_array)
acc_max = np.amax(acc_array)
acc_min = np.amin(acc_array)
acc = []
acc.append(acc_mean)
acc.append(acc_max)
acc.append(acc_min)

val_acc_array = np.asarray(list(history.history['val_accuracy']), dtype=np.float64)
val_acc_mean = np.mean(val_acc_array)
val_acc_max = np.amax(val_acc_array)
val_acc_min = np.amin(val_acc_array)
val_acc = []
val_acc.append(val_acc_mean)
val_acc.append(val_acc_max)
val_acc.append(val_acc_min)

loss_array = np.asarray(list(history.history['loss']), dtype=np.float64)
loss_mean = np.mean(loss_array)
loss_max = np.amax(loss_array)
loss_min = np.amin(loss_array)
loss = []
loss.append(loss_mean)
loss.append(loss_max)
loss.append(loss_min)

val_loss_array = np.asarray(list(history.history['val_loss']), dtype=np.float64)
val_loss_mean = np.mean(val_loss_array)
val_loss_max = np.amax(val_loss_array)
val_loss_min = np.amin(val_loss_array)
val_loss = []
val_loss.append(val_loss_mean)
val_loss.append(val_loss_max)
val_loss.append(val_loss_min)

model_evaluate = model.evaluate(valid_ds)

print(model_evaluate)

# TESTING

'''
model_folder = "simulations/model_FFT_E100_B128_20220911-045418/"
model_name = "model.h5"

model = keras.models.load_model(os.path.join(NETWORK_ROOT, model_folder + model_name))

p = 0.25  # 25% of the lines
# keep the header, then take only p*100% of lines from the source csv file
test_file_list = pd.read_csv(os.path.join(NETWORK_ROOT, test_file_list_path), header=0, skiprows=lambda i: i>0 and random.random() > p)
'''

y_true = []
y_predicted = []

for audios, labels in test_ds:
    ffts = audio_to_fft(audios)
    y_pred = model.predict(ffts)
    audios = audios.numpy()
    labels = labels.numpy()
    y_pred = np.argmax(y_pred, axis=-1)
    y_true.append(labels)
    y_predicted.append(y_pred)

real = np.concatenate(y_true)
predicted = np.concatenate(y_predicted)

r = len(real)
p = len(predicted)

if r == p:
    correct_predict = 0.0
    for i in range(len(real)):
        if int(predicted[i]) == int(real[i]):
            correct_predict += 1
else:
    print("Error - length of real and predicted vectors does not match!")

perc_corr_predict = (correct_predict*100)/r

metric_records = os.path.join(NETWORK_ROOT, "simulations/model_FFT_"+str(qtd_class)+"_E"+str(EPOCHS)+"_B"+str(BATCH_SIZE)+"_"+timestamp+"/"+"metrics.txt")

file = open(metric_records,'a+')
file.write("Epochs: "+str(EPOCHS)+"\n")
file.write("Batch: "+str(BATCH_SIZE)+"\n")
file.write("Tempo de Treino: "+str(end_train)+'\n')
file.write("Qtd. arquivos de treino: "+str(qtd_files_training)+'\n')
file.write("Qtd. arquivos de validação: "+str(qtd_files_validation)+'\n')
file.write("Validação do modelo: "+str(model_evaluate)+'\n')
file.write("Precisão: "+str(acc)+'\n')
file.write("Loss: "+str(loss)+'\n')
file.write("Precisão_val: "+str(val_acc)+'\n')
file.write("Loss_val: "+str(val_loss)+'\n')
file.write("Teste do modelo - resultado: "+"\n")
file.write("A porcentagem de acerto é de: "+str(perc_corr_predict)+"%")
