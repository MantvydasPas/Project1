# Project1
Personality type (python)







fig, ax = plt.subplots(len(data['type'].unique()), sharex=True, figsize=(15,10*len(data['type'].unique())))

k = 0
for i in data['type'].unique():
    data_4 = data[data['type'] == i]
    wordcloud = WordCloud().generate(data_4['posts'].to_string())
    ax[k].imshow(wordcloud)
    ax[k].set_title(i)
    ax[k].axis("off")
    k+=1
    
    
    # This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from nltk.corpus import stopwords 
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from scipy.misc import imread
from wordcloud import WordCloud, STOPWORDS
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD


data = pd.read_csv('../input/mbti_1.csv')
data.head()


plt.figure(figsize=(40,20))
plt.xticks(fontsize=24, rotation=0)
plt.yticks(fontsize=24, rotation=0)
sns.countplot(data=data, x='type')


data.groupby('type').agg({'type':'count'})

data.iloc[1,1]

data['postu_sk'] = data[['posts']].applymap(lambda x: str.count(x, '|||')+1)
data.head(20)

data['post_http'] = data[['posts']].applymap(lambda x: str.count(x, 'http'))
data.head()

data.groupby(['type']).agg({'post_http':[sum,np.mean]}).sort_values([('post_http', 'sum')])

data['post_jpg'] = data[['posts']].applymap(lambda x: str.count(x, 'jpg'))
data.head()

def var_row(row):
    l = []
    for i in row.split('|||'):
        l.append(len(i.split()))
    return np.var(l)

data['zodziu_sk'] = data['posts'].apply(lambda x: len(x.split()))
data['zodziu_per_koment'] = data['zodziu_sk'] / data['postu_sk']
data.head()

# Lemmatizer | Stemmatizer
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()
unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
lab_encoder = LabelEncoder().fit(unique_type_list) #Encode labels with value between 0 and n_classes-1.
list_personality = []  

for i, row in data.iterrows():
    
    # One post
    OnePost = row.posts

    # Cache the stop words for speed 
    cachedStopWords = stopwords.words("english")
    
    #Return the string obtained by replacing the leftmost non-overlapping occurrences of pattern in string by the replacement repl. 
    #If the pattern isn’t found, string is returned unchanged.
    
    # List all urls
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', OnePost)

    # Remove urls
    temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', OnePost)

    # Keep only words
    temp = re.sub("[^a-zA-Z]", " ", temp)

    # Remove spaces > 1
    temp = re.sub(' +', ' ', temp).lower()
     #if remove_stop_words:
    temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
       # else:
    #temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')]) #Join all items in a tuple into a string,

    data.loc[i,'pak_tekstas'] = temp
    
    data.loc[i, 'pak_tipas'] = lab_encoder.transform([row.type])[0]
    type_labelized = lab_encoder.transform([row.type])[0]
    list_personality.append(type_labelized)
    
    
    data['pak_tekst_zodziu_sk'] = data['pak_tekstas'].apply(lambda x: len(x.split()))

data['pak_zodziu_per_koment'] = data['pak_tekst_zodziu_sk'] / data['postu_sk']

data.head(15)



def get_types(row):
    t=row['type']

    I = 0; N = 0
    T = 0; J = 0
    
    if t[0] == 'I': I = 1
    elif t[0] == 'E': I = 0
    else: print('I-E incorrect')
        
    if t[1] == 'N': N = 1
    elif t[1] == 'S': N = 0
    else: print('N-S incorrect')
        
    if t[2] == 'T': T = 1
    elif t[2] == 'F': T = 0
    else: print('T-F incorrect')
        
    if t[3] == 'J': J = 1
    elif t[3] == 'P': J = 0
    else: print('J-P incorrect')
    return pd.Series( {'IE':I, 'NS':N , 'TF': T, 'JP': J }) 

data = data.join(data.apply (lambda row: get_types (row),axis=1))                                                                     
data.head(5)



print ("Intuition (N) – Sensing (S):\t\t", data['NS'].value_counts()[0], " / ", data['NS'].value_counts()[1])
print ("Thinking (T) – Feeling (F):\t\t", data['TF'].value_counts()[0], " / ", data['TF'].value_counts()[1])
print ("Judging (J) – Perceiving (P):\t\t", data['JP'].value_counts()[0], " / ", data['JP'].value_counts()[1])
N = 4
bottom = (data['IE'].value_counts()[0], data['NS'].value_counts()[0], data['TF'].value_counts()[0], data['JP'].value_counts()[0])
top = (data['IE'].value_counts()[1], data['NS'].value_counts()[1], data['TF'].value_counts()[1], data['JP'].value_counts()[1])

ind = np.arange(N)    # grupiu vieta
width = 0.85      # platumas

p1 = plt.bar(ind, bottom, width)
p2 = plt.bar(ind, top, width, bottom)

plt.ylabel('Kiekis')
plt.xlabel('Tipai')
plt.title('Pasiskirstymas')

plt.xticks(ind, ('I/E',  'N/S', 'T/F', 'J/P',))

plt.show()


data.groupby(['IE']).agg({'pak_tekst_zodziu_sk':[sum,np.mean]})


cntizer = CountVectorizer(analyzer="word", 
                             max_features=1500, 
                             tokenizer=None,    
                             preprocessor=None, 
                             stop_words=None,  
#                             ngram_range=(1,1),
                             max_df=0.5,
                             min_df=0.1) 
                                 
tfizer = TfidfTransformer()

print("CountVectorizer")
X_cnt = cntizer.fit_transform(data['pak_tekstas'])
print("Tf-idf")
X_tfidf =  tfizer.fit_transform(X_cnt).toarray()


y = data['pak_tipas']
X = X_tfidf
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier(max_depth = 10)
clf.fit(X_train, y_train)
clf.predict(X_test)

clf.score(X_test, y_test)


for i in range(100,1100,100):
    for j in range(1,11,1):
        rfc = RandomForestClassifier(n_estimators=i, max_depth=j)

        rfc.fit(X_train,y_train)
#y_pred=rfc.predict(X_test)

        print(i,j, rfc.score(X_test, y_test))
    
    
    
    rfc = RandomForestClassifier(n_estimators=1000)

rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)

print(rfc.score(X_test, y_test))


# Truncated SVD
svd = TruncatedSVD(n_components=12, n_iter=7, random_state=42)
svd_vec = svd.fit_transform(X_tfidf)

y = data['IE']
X = svd_vec
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


rfc = RandomForestClassifier(n_estimators=1000)

rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)

print(rfc.score(X_test, y_test))



y = data['IE']
X = X_tfidf
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

rfc = RandomForestClassifier(n_estimators=1000)

rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)

print(rfc.score(X_test, y_test))



y = data['NS']
X = svd_vec
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
rfc = RandomForestClassifier(n_estimators=1000)

rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)

print(rfc.score(X_test, y_test))



y = data['TF']
X = svd_vec
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
rfc = RandomForestClassifier(n_estimators=1000)

rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)

print(rfc.score(X_test, y_test))


y = data['JP']
X = svd_vec
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
rfc = RandomForestClassifier(n_estimators=1000)

rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)

print(rfc.score(X_test, y_test))



y = data['JP']
X = X_tfidf
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
rfc = RandomForestClassifier(n_estimators=1000)

rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)

print(rfc.score(X_test, y_test))




