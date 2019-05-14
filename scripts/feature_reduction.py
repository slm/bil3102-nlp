
# coding: utf-8

# In[42]:


'''
Öznitelikleri azaltma betiği

tf.csv dosyasını okur.
Karşıllıklı fayda bilgi kazanımı algoritmasıyla öznitelikleri 5000'e düşürür.
Yeni tabloyu reduced_data.csv dosyasına yazar.

'''


# In[38]:


import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif


# In[43]:


data = pd.read_csv("../tf.csv")


# In[40]:


clf  = SelectKBest(mutual_info_classif, k = 5000)
x = data[data.columns[~data.columns.isin(['class'])]]
y = data[data.columns[data.columns.isin(['class'])]]
clf.fit(x,np.ravel(y))
outcome = clf.get_support()
n_columns = []
for i in range(0,len(x.columns)):
    if outcome[i]:
        n_columns.append(x.columns[i])
#print("Selected features:%s" % n_columns)
feature_selected = datas[data.columns[data.columns.isin(n_columns)]]
feature_selected = feature_selected[feature_selected.columns[~feature_selected.columns.isin(['Unnamed: 0'])]]
feature_selected['class'] = y


# In[44]:


feature_selected.to_csv("../reduced_data.csv")

