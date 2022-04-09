import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
import librosa
import htk_featio as htk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping
import time
PATH = '/home/spilborghs/Documents/SER_DB/enterface_database/AUDIO_ONLY/'
def Feat_extract(files):
    file_name = os.path.join(os.path.abspath(PATH+str(files.file)))
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')# Generate Mel-frequency cepstral coefficients (MFCCs) from a time series
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)# Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)# Computes spectral contrast


    return mfccs, mel

nada,Bt, epoch = sys.argv

#FEAT_PATH = "/home/spilborghs/Documents/mel_filter_banks_features/Experiments/feat/"
Train_file = pd.read_csv("Csv/Train_wav.csv")
Test_file = pd.read_csv("Csv/Test_wav.csv")
Val_file = pd.read_csv("Csv/Valid_wav.csv")
Arq_train = Train_file['Arquive']
Arq_test = Test_file['Arquive']
Arq_val = Val_file['Arquive']
Res_train = Train_file['Results']
Res_test = Test_file['Results']
Res_val = Val_file['Results']
#print(Res_test)
V_train_x = []
for i in Arq_train:
    if i !='Arquive':
        V_train_x.append(i)

V_test_x = []

for i in Arq_test:
    if i !='Arquive':
        V_test_x.append(i)

V_val_x = []

for i in Arq_val:
    if i !='Arquive':
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
        train_features[i][1]), axis=0))


features_test = []
for i in range(0, len(test_features)):
    features_test.append(np.concatenate((
        test_features[i][0],
        test_features[i][1]), axis=0))


features_val = []
for i in range(0, len(val_features)):
    features_val.append(np.concatenate((
        val_features[i][0],
        val_features[i][1]), axis=0))


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

model = Sequential()
model.add(Dense(168, input_shape=(168,), activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(84, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(84, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation = 'softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
#early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')

N_epochs = int(epoch)
BSize = int(Bt)
ini_train = time.time()
history = model.fit(X_train, Y_train, batch_size=BSize, epochs=N_epochs,
                    validation_data=(X_val, Y_val)) #,
                    #callbacks=[early_stop])
model.save("model_"+str(N_epochs)+'_'+str(BSize))

fim_train = time.time() - ini_train
file = open("../../Dados_NN/librosaTestes.txt",'a+')
file.write("Teste com "+str(N_epochs)+" epocas e "+str(BSize)+" de batch: \n")
file.write("Tempo de Retirada de caracteristicas: "+str((1.0*fim_feat)/len(Res_test))+'\n')
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
plt.xticks(range(0,int(N_epochs),int(N_epochs/20)), range(0,int(N_epochs),int(N_epochs/20)))
plt.legend(fontsize = 18);

#plt.show()
ini_test = time.time()
# We get our predictions from the test data
predictions = np.argmax(model.predict(X_test), axis=-1)
#predictions = model.predict_classes(X_test)# We transform back our predictions to the speakers ids
predictions = lb.inverse_transform(predictions)# Finally, we can add those predictions to our original dataframe
test_predict = predictions
fim_test = time.time() - ini_test

file.write("Tempo de Teste: "+str((1.0*fim_test)/len(Res_test))+'\n')

correct_predict = 0.0
ConfMat = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
for i in range(0,len(Res_test)):
    if str(test_predict[i]) != 'NaN' and int(test_predict[i]) == int(Res_test[i]):
        correct_predict += 1.

    ConfMat[Res_test[i]][int(test_predict[i])] += 1


print("A porcentagem de acerto é de : "+str((correct_predict*100)/len(Res_test))+"%")
file.write("Porcentagem de acerto: "+str((correct_predict*100)/len(Res_test))+"%\n")
file.write('##########\n')
file.close()

file = open("/home/spilborghs/Documents/Dados_NN/Librosa_Confusion_matrix.txt",'a+')
file.write("Usando: Batch = "+str(Bt)+" | Epochs = "+str(epoch)+'\n')
for i in ConfMat:
    file.write(str(i)+'\n')

file.close()
