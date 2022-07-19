import csv
import pandas as pd
import os
import re


def csv_list(DIR, files):
    raw_audio_list = [(DIR + files[j]) for j in range(len(files))]
    csv_list = [(DIR + (re.sub(r'.raw', '.mfc.csv', files[i]))) for i in range(len(files))]
    return csv_list


DIR = os.path.join(os.path.expanduser("~"),'dataSet/audio/agender_distribution/')

train_file_txt = 'trainSampleList_train.txt'
devel_file_txt = 'trainSampleList_devel.txt'

file_tr = pd.read_table(str(DIR + train_file_txt), delimiter=' ', header=None)
file_dev = pd.read_table(str(DIR + devel_file_txt), delimiter=' ', header=None)

raw_file_TRlist = file_tr[0]
raw_file_DEVlist = file_dev[0]

csv_list_train = csv_list(DIR, raw_file_TRlist)
csv_list_dev = csv_list(DIR, raw_file_DEVlist)

invalid_count = 0

for i in range(len(csv_list_train)):
    fout = csv_list_train[i] + '.temp'
    run_command = 'wc ' + csv_list_train[i] + ' >> ' + fout
    with open('fout', 'w') as f:
        os.system(run_command)
        # Check  if it is empty or not
        regex = '\s0\s'
        k = re.search(regex, fout)
        if k == None:
            # Check if it is corrupted or not
            try:
                with open(csv_list_train[i], newline='') as f:
                    csvfile = csv.reader(f, delimiter=' ')
                    for row in csvfile:
                        print(', '.join(row))            
            except:
                invalid_count += 1
        else:
            invalid_count += 1
    os.system('rm ' + fout)

print(invalid_count)