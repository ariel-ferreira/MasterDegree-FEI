import pandas as pd

""" A funcao abaixo transforma as listas (.txt) com os nomes dos arquivos
de audio em um DataFrame, troca a extensao dos arquivos de .raw para .wav
(a fim de evitar erros com o librosa) e ainda faz a extração de informação
de a qual classe cada arquivo de audio pertence.
"""

in_file = "/home/ferreiraa/Documents/Mestrado/trabalho/redeNeural/\
agender_distribution/trainSampleList_train.txt"

out_file = "/home/ferreiraa/Documents/Mestrado/trabalho/redeNeural/\
agender_distribution/train_dataset.csv"

table_full = pd.read_table(in_file, delimiter=' ', header=None)
# lista arquivos
list_files = table_full[0]
list_files = list_files.str.replace(r'.raw', '.wav')
df_files = pd.DataFrame(list_files)
df_files = df_files.rename(columns={0: 'file'})
# lista generos
list_gender = table_full[4]
df_gender = pd.DataFrame(list_gender)
df_gender = df_gender.rename(columns={4: 'gender'})
# lista idades
list_age = table_full[3]
df_age = pd.DataFrame(list_age)
df_age = df_age.rename(columns={3: 'age'})
# criar coluna "class" no dataframe
table_full.insert(5, 'class', 0, True)
# loop para encontrar a qual classe cada arquivo pertence
for i in range(len(table_full)):
    g = table_full.loc[i, 4]
    if g == 'x':
        table_full.loc[i, 'class'] = 1
    elif g == 'm':
        a = table_full.loc[i, 3]
        if (a >= 15 and a <= 24):
            table_full.loc[i, 'class'] = 3
        elif (a >= 25 and a <= 54):
            table_full.loc[i, 'class'] = 5
        else:
            table_full.loc[i, 'class'] = 7
    else:
        a = table_full.loc[i, 3]
        if (a >= 15 and a <= 24):
            table_full.loc[i, 'class'] = 2
        elif (a >= 25 and a <= 54):
            table_full.loc[i, 'class'] = 4
        else:
            table_full.loc[i, 'class'] = 6
# lista classes
list_classes = table_full['class']
df_classes = pd.DataFrame(list_classes)
df_classes = df_classes.rename(columns={'class': 'class'})
#df_age = df_age.rename(columns={'class': 'class'})
#df_classes = df_classes.rename(columns={'class': 'class'})
tabela = pd.concat([df_files, df_age, df_gender, df_classes], axis=1)
tabela.to_csv(out_file, index=False)
