# Master degree work of Ariel Ferreira.
# Electrical Engineering Department of University Center of FEI - São Bernardo do Campo, SP - Brazil
# 2020 - 2022
# Age estimation based on human speech using a 1D convolutional neural network.

Para rodar as simulações: "targets"

estimação aleatória para 7 classes: (1/7)*100% = 14.3%

projeto Ivandro (parte 1):

precisão global para 7 classes: 56.8% (5138 arquivos utilizados no teste - retirados da lista de teste)
precisão global para 4* classes: 58.6% (5138 arquivos utilizados no teste - retirados da lista de teste)

* crianças, jovens, adultos, idosos. 

projeto Ivandro (parte 2):

precisão global para 4* classes: 49.2% (20492 arquivos utilizados no teste)
precisão global para 3** classes: 87.8% (20492 arquivos utilizados no teste)

** crianças, mulheres, homems.

---------------------------

Arquivos para treinamento, validação e teste:

test_database_full.csv - 
test_database_norm_full.csv - 
test_database_shuffled.csv - 
test_database_sorted.csv - 
test_sub_dataset_shuffled.csv - 
train_database_full.csv - 
train_database_norm_full.csv - 
train_database_shuffled.csv - 
train_database_sorted.csv - 
valid_sub_dataset_shuffled.csv - 

---------------------------

Agender Dataset:

dataset original:

20549 arquivos (utterances) para teste 
32527 arquivos (utterances) para treino

métricas:

total time (min): 1405.891666666668
total time (hrs): 23.431527777777802
total size (MB): 1288.5318511956002
total size (GB): 1.2578621720279999

---
dataset normalizado: 

20549 arquivos (utterances) para teste 
32527 arquivos (utterances) para treino

métricas:

total time (min): 654.0857333332484
total time (hrs): 10.901428888887475
total size (MB): 600.2144253204001
total size (GB): 0.585928101052


------ Organizar:

/home/ferreiraa/dataSet/audio/TIMIT/original/archive/timit/
/home/ferreiraa/dataSet/audio/TIMIT/normallized/

dataset preparation:
 
$ python TIMIT_preparation.py /home/ferreiraa/dataSet/audio/TIMIT/original/archive/timit /home/ferreiraa/dataSet/audio/TIMIT/normallized data_lists/TIMIT_all.scp

$ python TIMIT_preparation.py /home/ferreiraa/dataSet/audio/TIMIT/original/archive/timit /home/ferreiraa/dataSet/audio/TIMIT/torch-normalized data_lists/TIMIT_all.scp
 
 
# Open the wav file
wav_file=in_folder+'/'+list_sig[i]
[signal, fs] = sf.read(wav_file)
signal=signal.astype(np.float64)
 
# Signal normalization
signal=signal/np.abs(np.max(signal))

---

# Com 6 classes, fazendo a validação carregando o modelo treinado anteriormente:

Validação = 0.1 da base de treino

Total params: 1,712,470
Trainable params: 1,712,470
Non-trainable params: 0

[1.2998859882354736, 0.6088193655014038]
['loss', 'accuracy']
 
# Com 7 classes, fazendo a validação treinando o modelo:
 
Validação = 0.1 da base de treino
 
Total params: 1,712,728
Trainable params: 1,712,728
Non-trainable params: 0
 
[1.116835117340088, 0.5538130402565002]
['loss', 'accuracy']
Finished in 6683.54s] = 1h50min

# simulações:
teste==treino versus teste!=treino

considerando apenas teste!=treino:
Primeiramente montar a matriz de confusão com as 7 classes.
- 4 grandes classes (crianças, jovens, adultos e idosos): matriz, precision, recall, F1-score.
- 3 grandes classes (crianças, homens e mulheres): matriz, precision, recall, F1-score.

- Fazer os mesmos testes, porém remover a classe das crianças.

