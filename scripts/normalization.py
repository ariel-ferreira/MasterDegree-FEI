#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def csv_list(DIR, files):
    raw_audio_list = [(DIR + files[j]) for j in range(len(files))]
    csv_list = [(DIR + (re.sub(r'.raw', '.mfc.csv', files[i]))) for i in range(len(files))]
    return csv_list


# In[3]:


def ReadCSV(file):
    fin = file
    fout = pd.read_csv(fin, header=None)
    return fout


# In[4]:


def NumberVectors(file_list):
    min_sample = 0
    c = 0
    list_vec_qtd = []
    for i in range(len(file_list)):
        file = file_list[i]
        df = ReadCSV(file)
        c += 1
        num_vectors = len(df.index)
        num_samples = num_vectors
        list_vec_qtd.append(num_vectors)
        if min_sample == 0:
            min_sample = num_samples
        elif num_samples < min_sample:
            min_sample = num_samples
            min_file = file_list[i]
        else:
            None
    return (min_sample, min_file, c, list_vec_qtd)


# In[5]:


DIR = os.path.join(os.path.expanduser("~"),'dataSet/audio/agender_distribution/')
train_file_txt = 'trainSampleList_train.txt'
devel_file_txt = 'trainSampleList_devel.txt'
file_tr = pd.read_table(str(DIR + train_file_txt), delimiter=' ', header=None)
raw_file_TRlist = file_tr[0]
file_dev = pd.read_table(str(DIR + devel_file_txt), delimiter=' ', header=None)
raw_file_DEVlist = file_dev[0]


# In[6]:


csv_list_train = csv_list(DIR, raw_file_TRlist)
csv_list_dev = csv_list(DIR, raw_file_DEVlist)


# In[7]:


min_sample_train, min_file_train, counter_check_train, list_vec_qtd_train = NumberVectors(csv_list_train)


# In[116]:


min_sample_train, min_file_train, counter_check_train


# In[9]:


min_sample_dev, min_file_dev, counter_check_dev, list_vec_qtd_dev = NumberVectors(csv_list_dev)


# In[115]:


min_sample_dev, min_file_dev, counter_check_dev


# In[11]:


SampleListDataFrame_train = pd.DataFrame(list(zip(csv_list_train, list_vec_qtd_train)), columns=['Train Files', 'Qtd. of Vectors'])


# In[99]:


SampleListDataFrame_dev = pd.DataFrame(list(zip(csv_list_dev, list_vec_qtd_dev)), columns=['Dev Files', 'Qtd. of Vectors'])


# In[117]:


SampleListDataFrame_train


# In[105]:


counter_train = SampleListDataFrame_train['Qtd. of Vectors'].value_counts()


# In[120]:


#counter_train = counter_train.to_frame('qtd')


# In[107]:


counter_dev = SampleListDataFrame_dev['Qtd. of Vectors'].value_counts()


# In[119]:


#counter_dev = counter_dev.to_frame('qtd')


# In[113]:


threshold_train = 169
files_to_use_train = 0
for k in range(len(counter_train)):
    row_index = counter_train.index[k]
    row_value = counter_train.iat[k, 0]
    if row_index >= threshold_train:
        files_to_use_train = files_to_use_train + row_value
files_to_use_train


# In[114]:


threshold_dev = 169
files_to_use_dev = 0
for k in range(len(counter_dev)):
    row_index = counter_dev.index[k]
    row_value = counter_dev.iat[k, 0]
    if row_index >= threshold_dev:
        files_to_use_dev = files_to_use_dev + row_value
files_to_use_dev


# In[121]:


SampleListDataFrame_train[:3]


# In[124]:


SampleListDataFrame_train.iat[0,1]


# In[131]:


SampleListDataFrame_train.iat[3,0]


# In[132]:


train_csv_use = []
train_not_used = []
for i in range(len(SampleListDataFrame_train.index)):
    train_vectors = SampleListDataFrame_train.iat[i,1]
    if train_vectors >= threshold_train:
        train_csv_use.append(SampleListDataFrame_train.iat[i,0])
    else:
        train_not_used.append(SampleListDataFrame_train.iat[i,0])


# In[134]:


len(train_csv_use), len(train_not_used)


# In[ ]:


train_csv_use = []
train_not_used = []
for i in range(len(SampleListDataFrame_train.index)):
    train_vectors = SampleListDataFrame_train.iat[i,1]
    if train_vectors >= threshold_train:
        train_csv_use.append(SampleListDataFrame_train.iat[i,0])
    else:
        train_not_used.append(SampleListDataFrame_train.iat[i,0])


# In[136]:


dev_csv_use = []
dev_not_used = []
for i in range(len(SampleListDataFrame_dev.index)):
    dev_vectors = SampleListDataFrame_dev.iat[i,1]
    if dev_vectors >= threshold_dev:
        dev_csv_use.append(SampleListDataFrame_dev.iat[i,0])
    else:
        dev_not_used.append(SampleListDataFrame_dev.iat[i,0])


# In[137]:


len(dev_csv_use), len(dev_not_used)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




