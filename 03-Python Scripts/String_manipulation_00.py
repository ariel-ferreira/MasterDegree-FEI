#!/usr/bin/env python
# coding: utf-8

# In[55]:


# Função do script: Ler arquivo txt com listas de arquivo (incluindo path) e trocar a segunda "/" por "-"
# Autor: Ariel Ferreira
# Data: 23/04/2023


# In[1]:


import re


# In[2]:


with open('TIMIT_all.scp', 'r') as f:
    file = f.readlines()


# In[50]:


files = []
for line in file:
    line = re.sub('(?:[\/]*\/[^\/]*)(\/)', "\g<0>-", line)
    line = re.sub('(.........)(/)', "\g<1>", line)
    files.append(line)


# In[54]:


with open('TIMIT_all.scp', 'w') as txt:
    for item in files:
        txt.write(item)


# In[ ]:





# In[ ]:




