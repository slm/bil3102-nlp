
# coding: utf-8

# In[290]:


'''
Model oluşturma ve oluşturulan modelin test edilmesi

reduced_data.csv dosyasını okur ve bu dosya ile 4 tane model oluşturur.
Test dosyalarını okur ve özniteliklerin sıklıklarıyla tablo oluşturur.

Kullanılan sınıflandırma algoritmalar:
1- Linear Support Vector 
2- K-Nearest Neighbors
3- Gaussian Naive Bayes
4- Stochastic Gradient Descent
5- Rocchio
6- Multinomial Naive Bayes

Bu algoritmalardan en iyi sonucu veren Multinominal Naive Bayes oldu.
Yüksek ihtimalle bu modelde ezberleme söz konusu olabilir bu kadar yüksek sonuçlar beklemiyordum.
Tüm algoritmaların raporlarını csv olarak yazdırdım.

'''


# In[567]:


import pandas as pd
import os
import numpy as np
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
from sklearn.metrics import classification_report

channel = grpc.insecure_channel('localhost:1234')
normalization_stub = z_normalization_g.NormalizationServiceStub(channel)
preprocess_stub = z_preprocess_g.PreprocessingServiceStub(channel)
morphology_stub = z_morphology_g.MorphologyServiceStub(channel)

def reportDf(report):
    report = [x.split(' ') for x in report.split('\n')]
    header = ['Class Name']+[x for x in report[0] if x!='']
    values = []
    for row in report[1:-5]:
        row = [value for value in row if value!='']
        if row!=[]:
            values.append(row)
    df = pd.DataFrame(data = values, columns = header)
    del df['support']
    avarages = ["ortalama"]
    for i in range(0,3):
        avarages.append(round(pd.to_numeric(df[['precision','recall','f1-score']].iloc[i]).mean(),2))  
    df.loc[df.index.max()+1] = avarages
    df = df.transpose()
    df.columns=list(df.iloc[0])
    df = df[1:]
    return df


# In[259]:


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

stop_words = list(map(lambda x: x.replace("\n","").encode("utf-8"), open("stop-words.txt",encoding="iso-8859-9").readlines()))
def isStopWord(word):
    if word == "UNK":
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


# In[260]:


data = pd.read_csv("../reduced_data.csv")
data = data[data.columns[~data.columns.isin(['Unnamed: 0'])]]
feature_names = data.columns


# In[261]:


def getFiles(root_path):
    ignore = ['.DS_Store']
    folders = set(os.listdir(root_path))-set(ignore)
    txt_files = []
    for folder in folders:
        folder_path = "%s/%s/" % (root_path,folder)
        for file in set(os.listdir(folder_path))-set(ignore):
            file_path = "%s/%s/%s" %(root_path,folder,file)
            txt_files.append(file_path)
    return txt_files

def getPreprocessedDocument(path):
    f_in = open(path,encoding="iso-8859-9").read()
    doc = preprocess(f_in)
    data = [0]*(len(feature_names))
    for index,word in enumerate(feature_names):
        data[index] = doc.count(word)
    data[-1] = doc_type = path.split("/")[3]
    return data


test_files = getFiles(root_path = "../data/test")


# In[262]:


test_data = []
for test_file in test_files:
    test_data.append(getPreprocessedDocument(test_file))


# In[591]:


test = pd.DataFrame(test_data,columns=list(feature_names))
test.to_csv("../test_data.csv")


# In[264]:


x = data[data.columns[~data.columns.isin(['class'])]]
y = data['class']
test_x = test[test.columns[~test.columns.isin(['class'])]]
test_y = test['class']
target_names = list(set(test_y))


# In[584]:


#LinearSVC
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support

svc_model = LinearSVC(random_state=12)
svc_model = svc_model.fit(x, y)
pred = svc_model.predict(test_x)

df = reportDf(classification_report(test_y, pred, target_names=target_names))
df.to_csv("../reports/linearsvc.csv")
df


# In[585]:


#Knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

neigh = KNeighborsClassifier(n_neighbors=5)

neigh.fit(x, y)

pred = neigh.predict(test_x)

df = reportDf(classification_report(test_y, pred, target_names=target_names))
df.to_csv("../reports/kneighborsclassifier.csv")
df


# In[586]:


#GaussianNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support

gnb = GaussianNB()
gnb = gnb.fit(x, y)
pred = gnb.predict(test_x)

df = reportDf(classification_report(test_y, pred, target_names=target_names))
df.to_csv("../reports/gaussiannb.csv")
df


# In[587]:


#SGDClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support

sgd = SGDClassifier(max_iter=1000, tol=1e-3)
sgd = clf.fit(x, y)
pred = sgd.predict(test_x)

df = reportDf(classification_report(test_y, pred, target_names=target_names))
df.to_csv("../reports/sgdclassifier.csv")
df


# In[594]:


#Rocchio
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.metrics import precision_recall_fscore_support
rocchio = NearestCentroid(metric='euclidean')
rocchio = rocchio.fit(x, y)
pred = rocchio.predict(test_x)
df = reportDf(classification_report(test_y, pred, target_names=target_names))
df.to_csv("../reports/rocchio.csv")
df


# In[590]:


#MultinomialNB
from sklearn.naive_bayes import MultinomialNB
multinomialNB = MultinomialNB()
multinomialNB.fit(x,y)
pred = multinomialNB.predict(test_x)
df = reportDf(classification_report(test_y, pred, target_names=target_names))
df.to_csv("../reports/multinomialnb.csv")
df


# In[275]:


def getData(str_data):
    doc =  preprocess(str_data)
    print(doc)
    data = [0]*(len(feature_names)-1)
    for index,word in enumerate(list(set(feature_names)-set(['class']))):
        data[index] = doc.count(word)
    return data    

