#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import re
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os


# In[ ]:


get_ipython().system('ls bbc')


# In[2]:


#arquivos = [ x[0]+'/'+nome  for x in list(os.walk("bbc")) for nome in x[2]]
arquivos = [ x[0]+'/'+nome  for x in list(os.walk("movie_reviews")) for nome in x[2]]


# In[3]:


#arquivosTech = [x for x in arquivos if  x.split('/')[1] in ['tech']  ]
#arquivosNTech = [x for x in arquivos if  x.split('/')[1] in ['business', 'entertainment' , 'politics', 'sport']  ]
arquivosTech = [x for x in arquivos if  x.split('/')[1] in ['pos']  ]
arquivosNTech = [x for x in arquivos if  x.split('/')[1] in ['neg']  ]


# In[4]:


def readDoc(file):
    with open(file,'r',encoding='latin-1') as f:
        return f.read()


# In[5]:


#TODO: tirar números?
def tokenize(text):
    text=text.lower()
    text=re.sub('\n',' ',text)
    text=re.sub('[,.+=]',' ',text)
    text=re.sub('[0-9]','',text)
    text=re.sub('\s+',' ',text)
    text=text.strip()
    return text.split(' ')


# In[6]:


#retirar stop words?
def Train(docsTrue, docsFalse):
    total=len(docsTrue)+len(docsFalse)
    logPrioriT = np.log(len(docsTrue)/total)
    logPrioriF = np.log(len(docsFalse)/total)

    counterTotal = collections.Counter([y for x in [*docsTrue, *docsFalse]  for y in tokenize(x)])
    V = set([y for x in [*docsTrue, *docsFalse]  for y in tokenize(x)])
    counterT = collections.Counter([y for x in docsTrue for y in tokenize(x)])
    counterF = collections.Counter([y for x in docsFalse for y in tokenize(x)])

    denT = sum([x[1] for x in counterT.items()])+len(V)
    denF = sum([x[1] for x in counterF.items()])+len(V)

    likelyhoodT = { v: np.log((counterT[v]+1)/denT) for v  in V }
    likelyhoodF = { v: np.log((counterF[v]+1)/denF) for v  in V }
    
    return {'logPrioriT': logPrioriT, 'logPrioriF': logPrioriF, 'likelyT': likelyhoodT, 'likelyF': likelyhoodF, 'V': V}


# In[ ]:





# In[12]:


modelo = Train(docsTrue, docsFalse)


# In[7]:


def Classify(modelo, doc):
    logPrioriT = modelo['logPrioriT']
    logPrioriF = modelo['logPrioriF']
    likelyT = modelo['likelyT']
    likelyF = modelo['likelyF']
    V = modelo['V']
    
    classT = logPrioriT
    
    for w in tokenize(doc):
        if w in V:
            classT += likelyT[w]
    
    
    classF = logPrioriF
    for w in tokenize(doc):
        if w in V:
            classF += likelyF[w]
    
    return np.argmax([classF, classT])


# In[8]:


def CrossValidation(docsTrue, docsFalse, k=10):
    docs=[ (x,1) for x in docsTrue ]
    docs=docs+[ (x,0) for x in docsFalse ]
    np.random.shuffle(docs)
    
    sz = round(len(docs)/k)
    
    grupos = [ docs[idx:idx+sz]  for idx in range(0,len(docs), sz)]
    
    if(len(grupos) > k):
        grupos[-2] += grupos[-1]
    
    grupos=grupos[:k]
    
    for i in range(k):
        yield ( [doc for z in list(set(range(k)) -{i} ) for doc in grupos[z] ],  grupos[i] )


# In[ ]:





# In[ ]:


set(range(10))-{5}


# In[9]:


def Experimento(setup):
    train = setup[0]
    test = setup[1]
    
    docsTrueTrain=[x[0] for x in train if x[1]==1]
    docsFalseTrain=[x[0] for x in train if x[1]==0]
    
    
    modelo = Train(docsTrueTrain, docsFalseTrain)
        
    result = [ (Classify(modelo, x), y)  for x,y in test ]
    return result
    


# In[10]:


def Score(r):
    tp, tn, fp, fn = 0,0,0,0
    for x in r:
        tp += x[0]==1 and x[1]==1
        tn += x[0]==0 and x[1]==0
        fp += x[0]==1 and x[1]==0
        fn += x[0]==0 and x[1]==1
    
    prec =  tp/(tp+fp)
    rev  =  tp/(tp+fn)
    
    return ( prec, rev, 2*prec*rev/(prec+rev)  )


# In[13]:


setups = CrossValidation(docsTrue, docsFalse, 20)

for s in setups:
    train, test = s
    r = Experimento(s)
    print(Score(r))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


docsTrue=[readDoc(x) for x in arquivosTech]
docsFalse=[readDoc(x) for x in arquivosNTech]


# In[ ]:





# In[ ]:


Train(docsTrue, docsFalse)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




