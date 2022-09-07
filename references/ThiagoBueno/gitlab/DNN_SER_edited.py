import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import librosa
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping
import time


def Feat_extract(files):
    file_name = os.path.join(DATASET_ROOT+files)
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    # Generate Mel-frequency cepstral coefficients (MFCCs) and Delta-MFCCs from a time series
    MFCC_ = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
    mfccs = np.mean(MFCC_.T,axis=0)
    Delta_Mfcc = np.mean(librosa.feature.delta(MFCC_).T,axis=0)
    # Computes melspectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    # Compute 6 features based on Tones
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0) 
    # Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=np.abs(librosa.stft
    (X)), sr=sample_rate).T,axis=0) 
    # Zero-Crossing rate - 1 Value
    ZCR = np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0)
    # RMS - 1 Value
    RMS = np.mean(librosa.feature.rms(y=X).T,axis=0) 
    #return mfccs, mel, tonnetz, contrast, ZCR, RMS, Delta_Mfcc, P_feat[0:6]
    return mfccs, Delta_Mfcc, contrast, mel, tonnetz, ZCR, RMS

def Neural_TVT(Batch,Epoch,InpSp,X_train,Y_train,X_val,Y_val,X_test,Y_test):
    #Neural_TVT:
    #          - Batch (int)
    #          - epoch (int)
    #          - InpSp (int) Input Size
    #          - X_train (Vet)
    #          - Y_train (Vet)
    #          - X_val (Vet)
    #          - Y_val (Vet)
    #          - X_test (Vet)
    #          - Y_test(Vet)
    # Hiddent units (First Hidden Layer)
    HU = int(InpSp * 3)
    model = Sequential()
    model.add(Dense(HU, input_shape=(InpSp,), activation = 'relu'))
    model.add(Dense(InpSp*2, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(InpSp, activation = 'relu'))
    model.add(Dropout(0.15))
    model.add(Dense(6, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
    ini_train = time.time()
    history = model.fit(X_train, Y_train, batch_size=Batch, epochs=Epoch,validation_data=(X_val, Y_val), callbacks=[early_stop])
    model.save("model_"+str(Epoch)+'_'+str(Batch))

    fim_train = time.time() - ini_train
    file = open("FULL_DNNTestes.txt",'a+')
    file.write("Teste_ com "+str(Epoch)+" epocas e "+str(Batch)+" de batch: \n")
    file.write("Tempo de Retirada de caracteristicas: "+str((1.0*fim_feat)/len(Y_test))+'\n')
    file.write("Tempo de Treino: "+str(fim_train)+'\n')

    # Check out train accuracy and validation accuracy over epochs
    train_accuracy = history.history['accuracy']
    # Set figure size
    val_accuracy = history.history['val_accuracy']
    # Generate line plot of training, testing loss over epochs
    plt.figure(figsize=(12, 8))
    plt.plot(train_accuracy, label='Training Accuracy', color='#185fad')
    # Set title
    plt.plot(val_accuracy, label='Validation Accuracy', color='orange')
    plt.title('Training and Validation Accuracy by Epoch', fontsize = 25)
    plt.xlabel('Epoch', fontsize = 18)
    plt.ylabel('Categorical Crossentropy', fontsize = 18)
    plt.xticks(range(0,int(Epoch),int(Epoch/20)), range(0,int(Epoch),int(Epoch/20)))
    plt.legend(fontsize = 18);

    plt.show()
    
    ini_test = time.time()
    predictions = np.argmax(model.predict(X_test), axis=-1)
    predictions = lb.inverse_transform(predictions)
    test_predict = predictions
    fim_test = time.time() - ini_test

    file.write("Tempo de Teste: "+str((1.0*fim_test)/len(Y_test))+'\n')

    correct_predict = 0.0
    ConfMat = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
    for i in range(0,len(Y_test)):
        if str(test_predict[i]) != 'NaN' and int(test_predict[i]) == int(Y_test[i]):
            correct_predict += 1.

        ConfMat[Y_test[i]][int(test_predict[i])] += 1

    print("A porcentagem de acerto é de : "+str((correct_predict*100)/len(Y_test))+"%")
    file.write("Porcentagem de acerto: "+str((correct_predict*100)/len(Y_test))+"%\n")
    file.write('##########\n')
    file.close()

    file = open("FULL_DNN_Confusion_matrix.txt",'a+')
    file.write("Usando_: Batch = "+str(Batch)+" | Epochs = "+str(Epoch)+'\n')
    for i in ConfMat:
        file.write(str(i)+'\n')

    file.close()

    # segundo modo de validar precisão
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print('test_acc: ', test_acc)
    print('test_loss: ', test_loss)


DATASET_ROOT = os.path.join(os.path.expanduser("~"),'dataSet/audio/agender_distribution/')
NETWORK_ROOT = os.path.join(os.path.expanduser("~"),'Mestrado-PC/github/Conv1D/CNN/')

train_file_list_path = 'file_lists/SER-DNN/SER-DNN_model_TRAIN_database.csv'
test_file_list_path = 'file_lists/SER-DNN/SER-DNN_model_TEST_database.csv'

Batch = 73
Epoch = 100
VALID_SPLIT = 0.1
SHUFFLE_SEED = 43

#Read files, shuffle rows, and split validation dataset from train file list

train_file_list = pd.read_csv(os.path.join(NETWORK_ROOT, train_file_list_path))
train_file_list = train_file_list.sample(frac=1, random_state=SHUFFLE_SEED, axis=0).reset_index(drop=True)
test_file_list = pd.read_csv(os.path.join(NETWORK_ROOT, test_file_list_path))
test_audio_files = test_file_list.sample(frac=1, random_state=SHUFFLE_SEED, axis=0).reset_index(drop=True)

split_index = int((len(train_file_list)) * VALID_SPLIT)

train_audio_files = train_file_list.iloc[split_index:].reset_index(drop=True)
valid_audio_files = train_file_list.iloc[:split_index].reset_index(drop=True)

train_df = train_audio_files['file']
test_df = test_audio_files['file']
val_df = valid_audio_files['file']

Res_train = train_audio_files['class']
Res_test = test_audio_files['class']
Res_val = valid_audio_files['class']

train_features = train_df.apply(lambda x: Feat_extract(x))
ini_feat = time.time()
test_features = test_df.apply(lambda x: Feat_extract(x))
fim_feat = time.time() - ini_feat
val_features = val_df.apply(lambda x: Feat_extract(x))

features_train = []
for i in range(0, len(train_features)):
    features_train.append(np.concatenate((
        train_features[i][0],
        train_features[i][1],
        train_features[i][2],
        train_features[i][3],
        train_features[i][4],
        train_features[i][5],
        train_features[i][6]), axis=0))

features_test = []
for i in range(0, len(test_features)):
    features_test.append(np.concatenate((
        test_features[i][0],
        test_features[i][1],
        test_features[i][2],
        test_features[i][3],
        test_features[i][4],
        test_features[i][5],
        test_features[i][6]), axis=0))

features_val = []
for i in range(0, len(val_features)):
    features_val.append(np.concatenate((
        val_features[i][0],
        val_features[i][1],
        val_features[i][2],
        val_features[i][3],
        val_features[i][4],
        val_features[i][5],
        val_features[i][6]), axis=0))

X_test = np.array(features_test)
X_train = np.array(features_train)
Y_train = np.array(Res_train)
X_val = np.array(features_val)
Y_val = np.array(Res_val)

lb = LabelEncoder()
Y_train = to_categorical(lb.fit_transform(Y_train))
Y_val = to_categorical(lb.fit_transform(Y_val))

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_val = ss.transform(X_val)
X_test = ss.transform(X_test)

input = 223

Neural_TVT(Batch,Epoch,input,X_train,Y_train,X_val,Y_val,X_test,Res_test)