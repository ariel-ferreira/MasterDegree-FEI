#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Função do script: Ler arquivo .npy
# Autor: Ariel Ferreira
# Data: 23/04/2023


# In[21]:


import numpy as np
import re


# In[30]:


dicio = np.load('TIMIT_labels.npy', allow_pickle = True)


# In[33]:


timit_labels = {}
d = dict(enumerate(dicio.flatten(), 1))
timit_labels = d[1]


# In[38]:


dicio_at = {}


# In[39]:


for key in timit_labels:
    path = re.sub('(?:[\/]*\/[^\/]*)(\/)', "\g<0>-", key)
    path = re.sub('(.........)(/)', "\g<1>", path)
    dicio_at[path] = timit_labels[key]


# In[54]:


array = np.array(dicio_at)


# In[55]:


np.save('TIMIT_labels.npy', array)

