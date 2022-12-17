import os
import pandas as pd
import numpy as np
import re
import csv


def list_raw2mfc(DIR, files):
    # raw_audio_list = [(DIR + files[j]) for j in range(len(files))]
    mfc_audio_list = [(DIR + (re.sub('.raw', '-n.mfc', files[i]))) for i in range(len(files))]
    return mfc_audio_list


def mfc2csv(files):
    mfc_audio_list = [(files[j]) for j in range(len(files))]
    csv_audio_list = [(re.sub('.mfc', '.mfc.csv', files[i])) for i in range(len(files))]
    for x in range(len(mfc_audio_list)):
        fin = mfc_audio_list[x]
        fout = csv_audio_list[x]
        fin_tmp = mfc_audio_list[x]+'.tmp'
        htk_command = 'HList -r '
        htk_output = htk_command + fin +' > '+ fin_tmp
        _ = os.system(htk_output)
        with open(fin_tmp, 'r') as f:
            text = f.readlines()
            regex1 = '\n'
            regex2 = '\s$'
            regex3 = '.+.\d+e\W\d+'
            for line in text:
                line = re.sub(regex1,'',line)
                line = re.sub(regex2,'',line)
            coef_list_pos = []
            for line in text:
                coef_list_tmp = re.findall(regex3, line)
                coef_list_pos.append(coef_list_tmp)
            coef_df = pd.DataFrame(coef_list_pos)
            coef_df.to_csv(fout, sep = ';', header = False, index=False)
    return None


DIR = os.path.join(os.path.expanduser("~"),'dataSet/audio/agender_distribution/')
train_file_txt = 'trainSampleList_train.txt'
devel_file_txt = 'trainSampleList_devel.txt'

file_tr = pd.read_table(str(DIR + train_file_txt), delimiter=' ', header=None)
raw_file_TRlist = file_tr[0]
file_dev = pd.read_table(str(DIR + devel_file_txt), delimiter=' ', header=None)
raw_file_DEVlist = file_dev[0]

mfc_file_TRlist = list_raw2mfc(DIR, raw_file_TRlist)
mfc_file_DEVlist = list_raw2mfc(DIR, raw_file_DEVlist)

mfc2csv(mfc_file_TRlist)
mfc2csv(mfc_file_DEVlist)