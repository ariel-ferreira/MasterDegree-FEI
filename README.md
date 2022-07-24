# Master degree work of Ariel Ferreira.
# Electrical Engineering Department of University Center of FEI - São Bernardo do Campo, SP - Brazil
# 2020 - 2022
# Age estimation based on human speech using a 1D convolutional neural network.

Folder scripts:

check_empty_files.py - script criado para verificar quantos arquivos haviam sido perdidos após a recuperação do bando de dados do HDD da Seagate.Não é mais utilizado, visto que todos os arquivos do banco foram recriados.

file_selection_norm.py - script utilizado para verificar a quantidade de vetores em cada arquivo de coeficientes MFCC (referente a cada arquivo de áudio do banco), e assim selecionar a lista de arquivos que serão utillizos na rede. Esse script apenas cria um .csv com o nome/classe de cada arquivo que possui número de vetores igual ou superior ao número mínimo definido para normalização das matrizes (169). 

create_arrays.py - lê os arquivos selecionados através do script "file_selection_norm.py", importando de cada csv apenas a quantidade de vetores definida como a quantidade padrão para normalização das matrizes (169, criando assim matrizes 39x169) e, em seguida, realiza a transformação para NumPy arrays e as salva em arquivos binários.