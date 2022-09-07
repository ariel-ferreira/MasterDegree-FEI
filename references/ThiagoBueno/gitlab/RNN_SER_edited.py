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
from keras.callbacks import EarlyStopping
import time
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
# from tensorflow.ragged import constant


#path_app = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(path_app+'/Disvoice/DisVoice-master/prosody')

#from prosody import Prosody


PATH = '../AUDIO_ONLY/'

def Feat_Conc(V1,V2):
    Vetor_Final = []
    if len(V1) == len(V2):
        for i in range(0,len(V1)):
            Vetor_Final.append([])
            for j in range(0,len(V1[i])):
                Vetor_Final[i].append(V1[i][j])
            for j in range(0,len(V2[i])):
                Vetor_Final[i].append(V2[i][j])

    return Vetor_Final

def Norm_size(V,n_carac=183,m_size=500):
    V2 = np.full((m_size,n_carac),fill_value=-500.0)
    #print(V.shape)
    if len(V) > m_size:
        Linhas = m_size
    else:
        Linhas = len(V)
    Colunas = n_carac
    for line in range(0,Linhas-1):
        for col in range(0,Colunas-1):
            if line < len(V) and col < len(V[0]):
                V2[line][col] = V[line][col]
    return V2

def Adapt_2_RNN(V,size=1000):
    Sup_Vec = np.array([V])
    Adapted_Vec = np.repeat(Sup_Vec,size,axis=0)
    return Adapted_Vec

def Feat_extract(files):
    file_name = os.path.join(os.path.abspath(PATH+str(files.file)))
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T # 40 Valores
    Audio_len = mfccs.shape[0]
    Delta_Mfcc = librosa.feature.delta(mfccs) # 40 Valores

    # Computes melspectrogram
    mel = librosa.feature.melspectrogram(X, sr=sample_rate).T # 128 Valores
    # Computes Tonal Centroid Features
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T # 6 Valores
    # Computes spectral contrast
    Ss = np.abs(librosa.stft(X))
    contrast = librosa.feature.spectral_contrast(S=Ss, sr=sample_rate).T # 7 Valores

    ZCR = librosa.feature.zero_crossing_rate(y=X).T # 1 Valor
    RMS = librosa.feature.rms(y=X).T # 1 Valor
    #########################
    #pros = Prosody()
    #P_feat = pros.extract_features_file(file_name, static=True, plots=False, fmt="npy") #103 Valores

    #prosodic_feat = Adapt_2_RNN(P_feat[0:6],size=Audio_len) # 6 valores

    ########################
    return mfccs, Delta_Mfcc, contrast, mel, tonnetz, ZCR, RMS

def RNN_TVT(Batch,Epoch,Mod_Shape,InpSp,X_train,Y_train,X_val,Y_val,X_test,Y_test):
    #RNN_TVT:
    #          - Batch (int)
    #          - epoch (int)
    #          - Mod_Shape (int) input_shape
    #          - InpSp (int) Input Size
    #          - X_train (Vet)
    #          - Y_train (Vet)
    #          - X_val (Vet)
    #          - Y_val (Vet)
    #          - X_test (Vet)
    #          - Y_test(Vet)
    HU = int(InpSp * 3) # Hiddent units (First Hidden Layer)
    model = Sequential()

    # Usar Masking para -500.0
    model.add(Masking(mask_value=-500.0,input_shape=(Mod_Shape,InpSp)))
    # Recurrent layer
    model.add(LSTM(Hu,return_sequences=False,
                   dropout=0.2, recurrent_dropout=0.2))

    model.add(Dense(InpSp*2, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(InpSp, activation='relu'))

    # Dropout for regularization
    model.add(Dropout(0.15))

    # Output layer
    model.add(Dense(6, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
    ini_train = time.time()
    history = model.fit(X_train,  Y_train,
                        batch_size=int(Batch), epochs=int(Epoch),
                        validation_data=(X_val, Y_val))
                        #, callbacks=[early_stop])
    model.save("RNN_model_"+str(Epoch)+'_'+str(Batch))

    fim_train = time.time() - ini_train
    N_epochs = int(epoch)
    file = open("../Dados_Finais/RNNTestes.txt",'a+')
    file.write("Teste com "+str(Epoch)+" epocas e "+str(Batch)+" de batch: \n")
    file.write("Tempo de Retirada de caracteristicas: "+str((1.0*fim_feat)/len(Y_test))+'\n')
    file.write("Tempo de Treino: "+str(fim_train)+'\n')


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
    #predictions = lb.inverse_transform(predictions)
    test_predict = predictions
    fim_test = time.time() - ini_test

    file.write("Tempo de Teste: "+str((1.0*fim_test)/len(Res_test))+'\n')

    correct_predict = 0.0
    ConfMat = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
    for i in range(0,len(Res_test)):
        if str(test_predict[i]) != 'NaN' and int(test_predict[i]) == int(Y_test[i]):
            correct_predict += 1.

        ConfMat[Y_test[i]][int(test_predict[i])] += 1


    print("A porcentagem de acerto é de : "+str((correct_predict*100)/len(Y_test))+"%")
    file.write("Porcentagem de acerto: "+str((correct_predict*100)/len(Y_test))+"%\n")
    file.write('##########\n')
    file.close()

    file = open("../Dados_Finais/LSTMConfusion_matrix.txt",'a+')
    file.write("Usando: Batch = "+str(Batch)+" | Epochs = "+str(Epoch)+'\n')
    for i in ConfMat:
        file.write(str(i)+'\n')

    file.close()

def main():
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

    Train_file = pd.read_csv("Csv/Train_2.csv")
    Test_file = pd.read_csv("Csv/Test_2.csv")
    Val_file = pd.read_csv("Csv/Valid_2.csv")
    Arq_train = Train_file['Arquive']
    Arq_test = Test_file['Arquive']
    Arq_val = Val_file['Arquive']
    Res_train = Train_file['Results']
    Res_test = Test_file['Results']
    Res_val = Val_file['Results']

    N_train = 0
    V_train_x = []
    for i in Arq_train:
        if i !='Arquive':
            V_train_x.append(i)
            N_train += 1

    V_test_x = []
    N_test = 0
    for i in Arq_test:
        if i !='Arquive':
            V_test_x.append(i)
            N_test += 1
    V_val_x = []
    N_val = 0
    for i in Arq_val:
        if i !='Arquive':
            V_val_x.append(i)
            N_val += 1

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
        features_train.append(Norm_size(np.concatenate((
            train_features[i][0],
            train_features[i][1],
            train_features[i][2],
            train_features[i][3]), axis=1),n_carac=215,m_size=200))



    features_test = []
    for i in range(0, len(test_features)):
        features_test.append(Norm_size(np.concatenate((
            test_features[i][0],
            test_features[i][1],
            test_features[i][2],
            test_features[i][3]), axis=1),n_carac=215,m_size=200))


    features_val = []
    for i in range(0, len(val_features)):
        features_val.append(Norm_size(np.concatenate((
            val_features[i][0],
            val_features[i][1],
            val_features[i][2],
            val_features[i][3]), axis=1),n_carac=215,m_size=200))



    Y_train = np.array(Res_train)
    Y_val = np.array(Res_val)

    X_train = np.asarray(features_train)

    X_test = np.array(features_test)
    X_val =  np.array(features_val)

    lb = LabelEncoder()
    Y_train = to_categorical(lb.fit_transform(Y_train))
    Y_val = to_categorical(lb.fit_transform(Y_val))

    X_train = X_train.reshape(len(X_train),200,215)

    X_test = X_test.reshape(len(X_test),200,215)

    X_val = X_val.reshape(len(X_val),200,215)

    input = 215
    input_shape = 200
    RNN_TVT(Batch,Epoch,input_shape,input,X_train,Y_train,X_val,Y_val,X_test,Res_test)

if __name__ == '__main__':
    main()
