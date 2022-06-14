import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
import librosa
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping
import time
#path_app = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(path_app+'/Disvoice/DisVoice-master/prosody')

#from prosody import Prosody

#PATH = '../AUDIO_ONLY/'
PATH = '/home/ferreiraa/Documents/Mestrado/trabalho/redeNeural/agender_distribution/'

def Feat_extract(files):
    file_name = os.path.join(os.path.abspath(PATH+str(files.file)))
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

    # Generate Mel-frequency cepstral coefficients (MFCCs) and Delta-MFCCs from a time series
    MFCC_ = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
    mfccs = np.mean(MFCC_.T,axis=0)
    Delta_Mfcc = np.mean(librosa.feature.delta(MFCC_).T,axis=0)

    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)# Computes melspectrogram
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0) #Compute 6 features based on Tones
    contrast = np.mean(librosa.feature.spectral_contrast(S=np.abs(librosa.stft(X)), sr=sample_rate).T,axis=0) # Computes spectral contrast
    ZCR = np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0) # Zero-Crossing rate - 1 Value
    RMS = np.mean(librosa.feature.rms(y=X).T,axis=0) # RMS - 1 Value
    #########################
    #pros = Prosody()
    #P_feat = pros.extract_features_file(file_name, static=True, plots=False, fmt="npy") #103 Values

    ########################
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
    HU = int(InpSp * 3) # Hiddent units (First Hidden Layer)
    model = Sequential()
    model.add(Dense(HU, input_shape=(InpSp,), activation = 'relu'))
    model.add(Dense(InpSp*2, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(InpSp, activation = 'relu'))
    model.add(Dropout(0.15))
    model.add(Dense(6, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    #early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')

    ini_train = time.time()
    history = model.fit(X_train, Y_train, batch_size=Batch, epochs=Epoch,
                        validation_data=(X_val, Y_val))
                        #,callbacks=[early_stop])
    model.save("model_"+str(Epoch)+'_'+str(Batch))

    fim_train = time.time() - ini_train
    file = open("FULL_DNNTestes.txt",'a+')
    file.write("Teste_ com "+str(Epoch)+" epocas e "+str(Batch)+" de batch: \n")
    file.write("Tempo de Retirada de caracteristicas: "+str((1.0*fim_feat)/len(Y_test))+'\n')
    file.write("Tempo de Treino: "+str(fim_train)+'\n')

    # Check out our train accuracy and validation accuracy over epochs.
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']# Set figure size.
    plt.figure(figsize=(12, 8))# Generate line plot of training, testing loss over epochs.
    plt.plot(train_accuracy, label='Training Accuracy', color='#185fad')
    plt.plot(val_accuracy, label='Validation Accuracy', color='orange')# Set title
    plt.title('Training and Validation Accuracy by Epoch', fontsize = 25)
    plt.xlabel('Epoch', fontsize = 18)
    plt.ylabel('Categorical Crossentropy', fontsize = 18)
    plt.xticks(range(0,int(Epoch),int(Epoch/20)), range(0,int(Epoch),int(Epoch/20)))
    plt.legend(fontsize = 18);

    #plt.show()
    ini_test = time.time()
    predictions = np.argmax(model.predict(X_test), axis=-1)
    predictions = lb.inverse_transform(predictions)
    test_predict = predictions
    fim_test = time.time() - ini_test

    file.write("Tempo de Teste: "+str((1.0*fim_test)/len(Y_test))+'\n')

    correct_predict = 0.0
    #ConfMat = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
    for i in range(0,len(Y_test)):
        if str(test_predict[i]) != 'NaN' and int(test_predict[i]) == int(Y_test[i]):
            correct_predict += 1.

        #ConfMat[Y_test[i]][int(test_predict[i])] += 1

    print("A porcentagem de acerto é de : "+str((correct_predict*100)/len(Y_test))+"%")
    file.write("Porcentagem de acerto: "+str((correct_predict*100)/len(Y_test))+"%\n")
    file.write('##########\n')
    file.close()

    #file = open("FULL_DNN_Confusion_matrix.txt",'a+')
    #file.write("Usando_: Batch = "+str(Batch)+" | Epochs = "+str(Epoch)+'\n')
    #for i in ConfMat:
        #file.write(str(i)+'\n')

    #file.close()

    # segundo modo de validar precisão
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print('test_acc: ', test_acc)


#def main():
try:
    Batch = int(sys.argv[1])
    Epoch = int(sys.argv[2])
except ValueError:
    print("!!!!!!!!\n\n\n  Arguments must be Integers.\n Using default: 73 - 100\n\n\n")
    Batch = 73
    Epoch = 100
except IndexError:
    print("!!!!!!!!\n\n\n  You must pass Batch Size and Epochs for training.\n Using default: 73 - 100\n\n\n")
    Batch = 73
    Epoch = 100

#############
#Read files and split result from file
Train_file = pd.read_csv("Csv/Train_2.csv")
Test_file = pd.read_csv("Csv/Test_2.csv")
Val_file = pd.read_csv("Csv/Valid_2.csv")
#Arq_train = Train_file['Arquive']
Arq_train = Train_file['file']
#Arq_test = Test_file['Arquive']
Arq_test = Test_file['file']
#Arq_val = Val_file['Arquive']
Arq_val = Val_file['file']
#Res_train = Train_file['Results']
Res_train = Train_file['class']
#Res_test = Test_file['Results']
Res_test = Test_file['class']
#Res_val = Val_file['Results']
Res_val = Val_file['class']

V_train_x = []
for i in Arq_train:
    #if i !='Arquive':
    if i !='file':
        V_train_x.append(i)

V_test_x = []

for i in Arq_test:
    #if i !='Arquive':
    if i !='file':
        V_test_x.append(i)

V_val_x = []

for i in Arq_val:
    #if i !='Arquive':
    if i !='file':
        V_val_x.append(i)
#####################

train_df = pd.DataFrame(V_train_x)
test_df = pd.DataFrame(V_test_x)
val_df = pd.DataFrame(V_val_x)

train_df = train_df.rename(columns={0:'file'})
test_df = test_df.rename(columns={0:'file'})
val_df = val_df.rename(columns={0:'file'})

train_features = train_df.apply(Feat_extract, axis=1)
ini_feat = time.time()
test_features = test_df.apply(Feat_extract, axis=1)
fim_feat = time.time() - ini_feat
val_features = val_df.apply(Feat_extract, axis=1)


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


#end of creating df

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


#if __name__ == '__main__':
#    main()
