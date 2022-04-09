# Declaração das bibliotecas utilizadas
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.metrics import accuracy_score
from itertools import chain

# Definição do diretório onde estão armazenados os arquivos que compõem o banco
# de dados com os arquivos de áudio
DATASET_ROOT = '/home/arielferreira/Documents/Mestrado/trabalho/redeNeural/agender_distribution/'

# Definição da porção do banco de dados utilizado para treinamento do modelo de
#  rede neural, será utilizado para validação do modelo
VALID_SPLIT = 0.1

# Taxa de amostragem utilizada para os arquivos de áudio utilizados no modelo
SAMPLING_RATE = 8000

# "Seed" para geração de ações randômicas
SHUFFLE_SEED = 43

# Definição do tamanho do batch para uso no modelo de rede neural
BATCH_SIZE = 128

# Definição do número de epochs utilizo para o modelo de rede neural
EPOCHS = 100


# Função para construção de datasets composto por audios e labels
def paths_and_labels_to_dataset(audio_paths, labels):
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


# Função para leitura e decodificação dos arquivos de áudio .wav
def path_to_audio(path):
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio


# Função para aplicação do algoritmo de FFT nos sinais de áudio.
# Para um maior detalhamento da função, as próximas linhas alguns apresentam
# comentários em inglês pertencentes ao código original
def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(tf.cast(tf.complex(real=audio, imag=tf.zeros_like(
        audio)), tf.complex64))
    fft = tf.expand_dims(fft, axis=-1)
    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


# Criação de dois DataFrames iniciais, um contendo os nomes de cada arquivo de
# áudio que será utilizado para treinnamento do modelo, e outro contendo as
# respectivas classes a que tais aúdios pertencem
train_file_list = pd.read_csv('7_class_train.csv')
train_audio_files = train_file_list['file']
train_classes = train_file_list['class']
train_audio_df = pd.DataFrame(train_audio_files)
train_class_df = pd.DataFrame(train_classes)

# Criação de dois Dataframes, de forma similar como indicado no comentário do
# bloco anterior, porém nesse caso os arquivos serão utilizados para teste do
# modelo
test_file_list = pd.read_csv('7_class_test.csv')
test_audio_files = test_file_list['file']
test_classes = test_file_list['class']
test_audio_df = pd.DataFrame(test_audio_files)
test_class_df = pd.DataFrame(test_classes)

# Nas linhas do trecho de código a seguir, é feita a obtenção da lista com os
# nomes dos arquivos de áudio ("path" completo), bem como uma lista com as
# as labels correspondentes a classe a que cada arquivo pertence.
train_class_labels = list(train_classes.unique())
print("Age categories identified: {}".format(train_class_labels,))

# TODO_01: Fazer com que o código imprima uma tabela identificando as faixas
# etárias de cada categoria

audio_paths = []
labels = []
for label, category in enumerate(train_class_labels):
    print("Processing category {}".format(category,))
    speaker_sample_paths = [os.path.join(DATASET_ROOT, train_audio_files[i])
                            for i in range(len(train_audio_files))
                            if train_classes[i] == category]
    audio_paths += speaker_sample_paths
    labels += [label] * len(speaker_sample_paths)
print("Found {} files belonging to {} classes.".format(
    len(audio_paths), len(train_class_labels)))

# Bloco para executar o embaralhamento dos arquivos dentro das listas criadas
# anteriormente. Não há qualquer modificação de conteúdo
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(labels)
# Nas linhas do trecho de código a seguir, é feita a obtenção da lista com os
# nomes dos arquivos de áudio ("path" completo), bem como uma lista com as
# as labels correspondentes a classe a que cada arquivo pertence.
# Divisão da porção dos arquivos que serão utilizados para validação do modelo,
# logo após a etapa de treinamento
num_val_samples = int(VALID_SPLIT * len(audio_paths))
print("Using {} files for training.".format(len(audio_paths)
                                            - num_val_samples))
print("Using {} files for validation.".format(num_val_samples))
train_audio_paths = audio_paths[:-num_val_samples]
train_labels = labels[:-num_val_samples]
valid_audio_paths = audio_paths[-num_val_samples:]
valid_labels = labels[-num_val_samples:]

# Criação do dataset de treinamento e validação
train_ds = paths_and_labels_to_dataset(train_audio_paths, train_labels)
valid_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8,
                            seed=SHUFFLE_SEED).batch(BATCH_SIZE)
valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)

# No bloco a seguir, cada arquivo é submetido à função que aplica a FFT ao
# sinal de áudio
train_ds = train_ds.map(lambda x, y:
                        (audio_to_fft(x), y),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
valid_ds = valid_ds.map(lambda x, y:
                        (audio_to_fft(x), y),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)

# O trecho de código a seguir, engloba as funções e etapas de definição do
# modelo de rede neural convolucional 1D que está sendo utilizado no trabalho.
# Alguns comentários em inglês do código original foram mantidos para maior
# detalhamento do programa

# DEFINIÇÃO do Modelo
'''

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
    outputs = keras.layers.Dense(num_classes,
                                 activation="softmax",
                                 name="output")(x)
    return keras.models.Model(inputs=inputs, outputs=outputs)


model = build_model((SAMPLING_RATE // 2, 1), len(train_class_labels))
# model.summary()
# plot_model(model,
#            to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# Compile the model using Adam's default learning rate
model.compile(optimizer="Adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# Add callbacks:
# 'EarlyStopping' to stop training when the model is not enhancing anymore
# 'ModelCheckPoint' to always keep the model that has the best val_accuracy
model_save_filename = "CNN_1D_model.h5"
earlystopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                 restore_best_weights=True)
mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(model_save_filename,
                                                   monitor="val_accuracy",
                                                   save_best_only=True)

# TREINAMENTO do modelo

history = model.fit(train_ds, epochs=EPOCHS, validation_data=valid_ds,
                    callbacks=[earlystopping_cb, mdlcheckpoint_cb],)
print(model.evaluate(valid_ds))

# O trecho a seguir executa as linhas de código destinadas a realizar o teste
# do modelo. Para essa finalizada utiliza-se novos arquivos que não foram
# utilizados para a fase de treinamento
'''
# TESTE do modelo

# A linha de código abaixo serve apenas para quando se quer utilizar parâmetros
# da rede obtidos em treinamentos anteriores.

model = load_model('/home/arielferreira/github/Conv1D/CNN/CNN_1D_model.h5')

# Obtenção da lista com os nomes dos arquivos de áudio ("path" completo) para
# teste, bem como a lista com as as labels correspondentes as classes.

test_class_labels = list(test_classes.unique())
print("Age categories identified: {}".format(test_class_labels,))

test_audio_paths = []
test_labels = []

for label, category in enumerate(test_class_labels):
    print("Processing category {}".format(category,))
    speaker_sample_paths = [os.path.join(DATASET_ROOT, test_audio_files[i])
                            for i in range(len(test_audio_files))
                            if test_classes[i] == category]
    test_audio_paths += speaker_sample_paths
    test_labels += [label] * len(speaker_sample_paths)

print("Found {} files belonging to {} classes.".format(len(test_audio_paths),
                                                       len(test_class_labels)))
# Bloco para embaralhar as amostras de sinais de áudio e suas labels dentro
# das listas criadas~
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(test_audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(test_labels)

# Criação do dataset de teste e aplicação da FFT a cada sinal de áudio
test_ds = paths_and_labels_to_dataset(test_audio_paths, test_labels)
test_ds = test_ds.shuffle(buffer_size=BATCH_SIZE * 8,
                          seed=SHUFFLE_SEED).batch(BATCH_SIZE)
test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
test_ds = test_ds.map(lambda x, y:
                      (audio_to_fft(x), y),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

# No bloco a seguir, utilizando o modelo criado, é realizada a predição das
# classes a que cada arquivo deveria pertencer

audios_list = []
labels_list = []
y_pred_list = []

for audios, labels in test_ds:
    y_pred = model.predict(audios)
    audios = audios.numpy()
    labels = labels.numpy()
    y_pred = np.argmax(y_pred, axis=-1)
    audios_list.append(audios)
    labels_list.append(labels)
    y_pred_list.append(y_pred)

flatten_labels_list = list(chain(*labels_list))
flatten_y_pred_list = list(chain(*y_pred_list))
real_output = flatten_labels_list
predicted_output = flatten_y_pred_list

# Obtenção das métricas do modelo, a partir da execução da etapa de teste
print(accuracy_score(real_output, predicted_output))
# f1_score(real_output, predicted_output, average='macro'),
# f1_score(real_output, predicted_output, average='micro')
# precision_score(real_output, predicted_output, average='macro'),
# precision_score(real_output, predicted_output, average='micro')
# recall_score(real_output, predicted_output, average='macro'),
# recall_score(real_output, predicted_output, average='micro')
