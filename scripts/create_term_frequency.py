
# coding: utf-8

# In[ ]:


'''
Frekans tablosu oluşturan betil

Train dosyalarını okur. Zemberek ile normalleştirir ve ayıklar. 
Daha sonra ayıklanmış kelimeleri köklerine ayırır.
Bu kökler etkisiz kelimler ise (stop words) onları siler.
Bu kökler yardımıyla eşsiz kelimeleri bulur.
Bu eşsiz kelimelerin dökümanlarda geçme sıklıklarına göre frekans tablosu oluşturur.
Bu tabloyu tf.csv dosyasına yazar.

'''


# In[16]:


import grpc
import sys
import zemberek_grpc.language_id_pb2 as z_langid
import zemberek_grpc.language_id_pb2_grpc as z_langid_g
import zemberek_grpc.normalization_pb2 as z_normalization
import zemberek_grpc.normalization_pb2_grpc as z_normalization_g
import zemberek_grpc.preprocess_pb2 as z_preprocess
import zemberek_grpc.preprocess_pb2_grpc as z_preprocess_g
import zemberek_grpc.morphology_pb2 as z_morphology
import zemberek_grpc.morphology_pb2_grpc as z_morphology_g

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import os
import subprocess
from threading import Thread
import subprocess
import json
import pandas as pd


# In[17]:


#start zemberek


# In[18]:


#define base zemberek functions
channel = grpc.insecure_channel('localhost:1234')

langid_stub = z_langid_g.LanguageIdServiceStub(channel)
normalization_stub = z_normalization_g.NormalizationServiceStub(channel)
preprocess_stub = z_preprocess_g.PreprocessingServiceStub(channel)
morphology_stub = z_morphology_g.MorphologyServiceStub(channel)

def find_lang_id(i):
    response = langid_stub.Detect(z_langid.LanguageIdRequest(input=i))
    return response.langId

def tokenize(i):
    response = preprocess_stub.Tokenize(z_preprocess.TokenizationRequest(input=i))
    return response.tokens

def normalize(i):
    response = normalization_stub.Normalize(z_normalization.NormalizationRequest(input=i))
    return response

def analyze(i):
    response = morphology_stub.AnalyzeSentence(z_morphology.SentenceAnalysisRequest(input=i))
    return response;

def fix_decode(text):
    """Pass decode."""
    if sys.version_info < (3, 0):
        return text.decode('utf-8')
    else:
        return text


# In[19]:


stop_words = list(map(lambda x: x.replace("\n","").encode("utf-8"), open("stop-words.txt",encoding="iso-8859-9").readlines()))
def isStopWord(word):
    if word == "UNK":
        return True
    if len(word) <= 2:
        return True
    return word in stop_words

def preprocess(document):
    tokenized = fix_decode(tokenize(normalize(document).normalized_input))
    output = []
    for i in tokenized:
        if i.type == 'Word':
            lemma = analyze(i.token).results[0].best.lemmas[0]
            #print("lemma(%s)=%s"%(i.token,lemma))
            if not isStopWord(lemma):
                output.append(str(lemma))
    return output


# In[20]:


def preprocessWrite(path):
    f_in = open(path,encoding="iso-8859-9").read()
    output = preprocess(f_in)
    f_name = path.split("/")[-1]
    f_out_path = path.replace(f_name,"")+"../../../data-preprocessed"+path.replace("../data","")
    f_out = open(f_out_path,"wb")
    f_out.write(json.dumps(output,ensure_ascii=False).encode("utf-8"))
    f_out.close()
    return output


# In[21]:


root_path = "../data/train"
ignore = ['.DS_Store']
folders = set(os.listdir(root_path))-set(ignore)
txt_files = []
for folder in folders:
    folder_path = "%s/%s/" % (root_path,folder)
    for file in set(os.listdir(folder_path))-set(ignore):
        file_path = "%s/%s/%s" %(root_path,folder,file)
        txt_files.append(file_path)


# In[22]:


words = set()
for file in txt_files:
    outputs = preprocessWrite(file)
    for i in outputs:
        words.add(i)


# In[23]:


words = list(frozenset(words))
words.append('class')


# In[25]:


rows = []
for k in range(len(txt_files)):
    txt_file = txt_files[k].replace("data","data-preprocessed")
    doc_type = txt_file.split("/")[3]
    row = [0]*(len(words))
    document_words = json.loads(open(txt_file, encoding='utf-8').read())
    for index,word in enumerate(words):
        row[index] = document_words.count(word)
    row[-1] = doc_type
    row = tuple(row)
    rows.append(row)


# In[26]:


data = pd.DataFrame(rows,columns=words)


# In[28]:


data.to_csv("../tf.csv")

