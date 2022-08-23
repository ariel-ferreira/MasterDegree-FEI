import datetime
import numpy as np
import os
import pandas as pd
import re
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

DATASET_ROOT = os.path.join(os.path.expanduser("~"),'dataSet/audio/agender_distribution/')
NETWORK_ROOT = os.path.join(os.path.expanduser("~"),'Mestrado-PC/github/Conv1D/CNN/')
VALID_SPLIT = 0.1
SHUFFLE_SEED = 43
BATCH_SIZE = 128
EPOCHS = 100
QTD_VEC = 100

train_file_list_path = 'file_lists/train_database_normalized.csv'
devel_file_list_path = 'file_lists/test_database_normalized.csv'

def map_func(npy_path):
    npy_content = np.load(npy_path)
    return npy_content


def paths_and_labels_to_dataset(paths, labels):
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    array_ds = path_ds.map(lambda x: tf.numpy_function(map_func, [x], tf.float64),num_parallel_calls=tf.data.experimental.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((array_ds, label_ds))


# Read train files and split class from file

print("Inicio leitura lista de arquivos")

start_read = time.time()

train_file_list = pd.read_csv(os.path.join(NETWORK_ROOT, train_file_list_path))
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
    labels += [category] * len(speaker_sample_paths)

print(
    "Found {} files belonging to {} classes.".format(len(audio_paths),
                                                     len(train_class_labels)))

for i in range(len(audio_paths)):
    audio_paths[i] = re.sub('.mfc.csv', '.npy', audio_paths[i])

end_read = (time.time() - start_read) / 60

print("Termino leitura lista de arquivos")
print("Tempo transcorrido: {} min.".format(end_read))

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

train_ds = train_ds.shuffle(
    buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE)
valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)

train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)

print(list(train_ds.as_numpy_iterator()))

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
    # x = keras.layers.Dropout(0.2)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)

    outputs = keras.layers.Dense(
        num_classes, activation="softmax", name="output")(x)

    return keras.models.Model(inputs=inputs, outputs=outputs)


model = build_model((QTD_VEC, 39), len(train_class_labels))

#model.summary()

# Compile the model using Adam's default learning rate

model.compile(optimizer="Adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Add callbacks:
# 'EarlyStopping' to stop training when the model is not enhancing anymore
# 'ModelCheckPoint' to always keep the model that has the best val_accuracy

timestamp = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

model_save_filename = os.path.join(NETWORK_ROOT, 'simulations/model_HTK_'+'E'+str(EPOCHS)+'_'+'B'+str(BATCH_SIZE)+'_'+'V'+str(QTD_VEC)+'_'+timestamp+'.h5')

#earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(model_save_filename, monitor="val_accuracy", save_best_only=True)

# TRAINING

print("Início treinamento do modelo")

start_train = time.time()

# history = model.fit(train_ds, epochs=EPOCHS, validation_data=valid_ds, callbacks=[earlystopping_cb, mdlcheckpoint_cb])

history = model.fit(train_ds, epochs=EPOCHS, validation_data=valid_ds, callbacks=[mdlcheckpoint_cb])

end_train = (time.time() - start_train) / 60

print("Termino treinamento do modelo")
print("Tempo transcorrido: {} min.".format(end_train))

print("Iniciando sessão de métricas")

start_metrics = time.time()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# '''
epochs_range = range(EPOCHS)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig(os.path.join(NETWORK_ROOT, 'simulations/figure_HTK_'+'E'+str(EPOCHS)+'_'+'B'+str(BATCH_SIZE)+'_'+'V'+str(QTD_VEC)+'_'+timestamp+'.png'))
plt.show()
# '''

print(model.evaluate(valid_ds))

metrics_records = os.path.join(NETWORK_ROOT, 'simulations/metrics_HTK_'+'E'+str(EPOCHS)+'_'+'B'+str(BATCH_SIZE)+'_'+'V'+str(QTD_VEC)+'_'+timestamp+'.txt')

file = open(metrics_records,'a+')
file.write("Treino com "+str(EPOCHS)+" epochs e "+str(BATCH_SIZE)+" de batch: \n")
file.write("Tempo de Treino: "+str(end_train)+'\n')
file.write("Qtd. arquivos de treino: "+str(qtd_files_training)+'\n')
file.write("Qtd. arquivos de validação: "+str(qtd_files_validation)+'\n')
file.write("Accuracy: "+str(acc)+'\n')
file.write("Loss: "+str(loss)+'\n')
file.write("Val_accuracy: "+str(val_acc)+'\n')
file.write("Val_loss: "+str(val_loss)+'\n')

end_metrics = (time.time() - start_metrics) / 60

print("Fim sessão de métricas")
print("Tempo transcorrido: {} min.".format(end_metrics))