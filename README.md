# Master degree work of Ariel Ferreira.
# Electrical Engineering Department of University Center of FEI - São Bernardo do Campo, SP - Brazil
# 2020 - 2022
# Age estimation based on human speech using a 1D convolutional neural network.

Folder scripts:

check_empty_files.py - script criado para verificar quantos arquivos haviam sido perdidos após a recuperação do bando de dados do HDD da Seagate.Não é mais utilizado, visto que todos os arquivos do banco foram recriados.

file_selection_norm.py - script utilizado para verificar a quantidade de vetores em cada arquivo de coeficientes MFCC (referente a cada arquivo de áudio do banco), e assim selecionar a lista de arquivos que serão utillizos na rede. Esse script apenas cria um .csv com o nome/classe de cada arquivo que possui número de vetores igual ou superior ao número mínimo definido para normalização das matrizes (169). 

create_arrays.py - lê os arquivos selecionados através do script "file_selection_norm.py", importando de cada csv apenas a quantidade de vetores definida como a quantidade padrão para normalização das matrizes (169, criando assim matrizes 39x169) e, em seguida, realiza a transformação para NumPy arrays e as salva em arquivos binários.


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

