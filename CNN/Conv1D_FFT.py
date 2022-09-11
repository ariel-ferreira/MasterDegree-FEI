import datetime
import time
import numpy as np
import os
import pandas as pd
import re
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

DATASET_ROOT = os.path.join(os.path.expanduser("~"),'dataSet/audio/agender_distribution/')
NETWORK_ROOT = os.path.join(os.path.expanduser("~"),'Mestrado-PC/github/Conv1D/CNN/')

VALID_SPLIT = 0.20
SAMPLING_RATE = 8000
SHUFFLE_SEED = 43
BATCH_SIZE = 128
EPOCHS = 100

train_file_list_path = 'file_lists/normalizados/train_database_norm_sorted.csv'
test_file_list_path = 'file_lists/normalizados/test_database_norm_sorted.csv'

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

# In the next section, get the list of audio file paths along with their corresponding labels

class_labels = list(train_classes.unique())
print("Age categories identified: {}".format(train_classes,))

audio_paths = list(train_audio_files)
labels = list(train_classes)

for i in range(len(audio_paths)):
    audio_paths[i] = os.path.join(DATASET_ROOT, audio_paths[i])

print("Found {} files belonging to {} classes.".format(len(audio_paths), len(class_labels)))

# Shuffle

rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(labels)

# Split into training and validation

num_val_samples = int(VALID_SPLIT * len(audio_paths))

qtd_files_training = len(audio_paths) - num_val_samples
qtd_files_validation = num_val_samples

print("Using {} files for training.".format(len(audio_paths) - num_val_samples))
print("Using {} files for validation.".format(num_val_samples))

train_audio_paths = audio_paths[:-num_val_samples]
train_labels = labels[:-num_val_samples]
valid_audio_paths = audio_paths[-num_val_samples:]
valid_labels = labels[-num_val_samples:]

# Create 2 datasets, one for training and the other for validation

print("Início da criação dos datasets de treino e validação")

start_dataset = time.time()

train_ds = paths_and_labels_to_dataset(train_audio_paths, train_labels)
valid_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)

end_dataset = (time.time() - start_dataset) / 60

print("Termino da criação dos datasets de treino e validação")
print("Tempo transcorrido: {} min.".format(end_dataset))

train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE)
valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)

# Transform audio wave to the frequency domain

print("Início da aplicação de FFT nos sinais de áudio")

start_fft = time.time()

train_ds = train_ds.map(lambda x, y: (audio_to_fft(x), y),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
valid_ds = valid_ds.map(lambda x, y: (audio_to_fft(x), y),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

end_fft = (time.time() - start_fft) / 60

print("Termino da aplicação de FFT")
print("Tempo transcorrido: {} min.".format(end_fft))

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
    num_classes = num_classes + 1
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

timestamp = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

model_save_filename = os.path.join(NETWORK_ROOT, "simulations/model_FFT_"+"E"+str(EPOCHS)+"_"+"B"+str(BATCH_SIZE)+"_"+timestamp+"/"+"model.h5")

earlystopping_cb = keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True)
mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
    model_save_filename, monitor="val_accuracy", save_best_only=True)
tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir=NETWORK_ROOT+"simulations/model_FFT_"+"E"+str(EPOCHS)+"_"+"B"+str(BATCH_SIZE)+"_"+timestamp+"/"+"logs",
    histogram_freq=0,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq="epoch",
    profile_batch=0,
    embeddings_freq=0,
    embeddings_metadata=None
    )

# TRAINING

print("Início treinamento do modelo")

start_train = time.time()

history = model.fit(train_ds, epochs=EPOCHS, validation_data=valid_ds, callbacks=[mdlcheckpoint_cb, tensorboard_cb])

end_train = (time.time() - start_train) / 60

print("Termino treinamento do modelo")
print("Tempo transcorrido: {} min.".format(end_train))

print("Iniciando sessão de métricas")

start_metrics = time.time()

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

metric_records = os.path.join(NETWORK_ROOT, "simulations/model_FFT_"+"E"+str(EPOCHS)+"_"+"B"+str(BATCH_SIZE)+"_"+timestamp+"/"+"metrics.txt")

file = open(metric_records,'a+')
file.write("Epochs: "+str(EPOCHS)+"\n")
file.write("Batch: "+str(BATCH_SIZE)+"\n")
file.write("Tempo de Treino: "+str(end_train)+'\n')
file.write("Qtd. arquivos de treino: "+str(qtd_files_training)+'\n')
file.write("Qtd. arquivos de validação: "+str(qtd_files_validation)+'\n')
file.write("Validação do modelo: "+str(model_evaluate)+'\n')
file.write("Accuracy: "+str(acc)+'\n')
file.write("Loss: "+str(loss)+'\n')
file.write("Val_accuracy: "+str(val_acc)+'\n')
file.write("Val_loss: "+str(val_loss)+'\n')

end_metrics = (time.time() - start_metrics) / 60

print("Fim sessão de métricas")
print("Tempo transcorrido: {} min.".format(end_metrics))

print(model.evaluate(valid_ds))

exit(0)

# Test demo

model = keras.models.load_model(os.path.join(NETWORK_ROOT, "simulations/model_FFT_E100_B128_20220911-015734/model.h5"))

test_file_list = pd.read_csv(os.path.join(NETWORK_ROOT, test_file_list_path))
test_audio_files = test_file_list['file']
test_classes = test_file_list['class']

test_class_labels = list(test_classes.unique())
print("Age categories identified: {}".format(test_classes,))

test_audio_paths = list(test_audio_files)
test_labels = list(test_classes)

rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(test_audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(test_labels)

test_audio_paths = test_audio_paths[:1000]
test_labels = test_labels[:1000]

for i in range(len(test_audio_paths)):
    test_audio_paths[i] = os.path.join(DATASET_ROOT, test_audio_paths[i])

qtd_files_testing = len(test_audio_paths)

print("Found {} files belonging to {} classes.".format(qtd_files_testing, len(test_class_labels)))

test_ds = paths_and_labels_to_dataset(test_audio_paths, test_labels)

test_ds = test_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE)

SAMPLES_TO_DISPLAY = 50

y_true = []
y_predicted = []

for audios, labels in test_ds.take(1):
    # Get the signal FFT
    ffts = audio_to_fft(audios)
    # Predict
    y_pred = model.predict(ffts)
    # Take random samples
    rnd = np.random.randint(0, BATCH_SIZE, SAMPLES_TO_DISPLAY)
    audios = audios.numpy()[rnd, :, :]
    labels = labels.numpy()[rnd]
    y_pred = np.argmax(y_pred, axis=-1)[rnd]

    for index in range(SAMPLES_TO_DISPLAY):
        # For every sample, print the true and predicted label
        # as well as run the voice with the noise
        print(
            "True class: {} - Predicted class: {}".format(
                test_class_labels[labels[index]],
                test_class_labels[y_pred[index]],
            )
        )
        y_true.append(test_class_labels[labels[index]])
        y_predicted.append(test_class_labels[y_pred[index]])

correct_predict = 0.0
for i in range(len(y_true)):
     if int(y_predicted[i]) == int(y_true[i]):
         correct_predict += 1.

print("A porcentagem de acerto é de : "+str((correct_predict*100)/len(y_true))+"%")