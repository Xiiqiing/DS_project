#!/home/projects/ku_00039/people/zelili/programs/miniconda2/envs/phyluce-1.7.1/bin/python3.6
# -*- coding: utf-8 -*-

# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'

# %%
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

import numpy as np
import pandas as pd

# %%
# from google.colab import drive
# drive.mount('/content/drive')
# %%
news_path='/home/people/zelili/ds_p/final/data/all_news.csv'
sub_news_path='/home/people/zelili/ds_p/final/data/sub_news.csv'

#output the accuracy result to file
fout=open('/home/people/zelili/ds_p/final/data/out.txt','w')

# %%
#load data
news = pd.read_csv(news_path)
# marge labels
news['type'].replace(['political','reliable'], 1, inplace = True)
news['type'].replace(['conspiracy', 'junksci', 'hate', 'satire', 'fake','bias'], 0, inplace = True)
#delete 'unknown'
news = news.drop((news[news['type']=='unknown'].index)|(news[news['type']=='rumor'].index)|(news[news['type']=='unreliable'].index)|(news[news['type']=='clickbait'].index))
news['type']=pd.to_numeric(news['type'])
# remove NaN
news = news.dropna()


# %%
news.groupby('type').size().sort_values(ascending = False)


# %%
# for saving time and ram
sub_news=news.sample(frac=0.01)
sub_news.to_csv(sub_news_path,index=False)


# %%
sub_news.groupby('type').size().sort_values(ascending = False)


# %%
#sub_news = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Final/sub_news.csv')

# %% [markdown]
# # tf-idf

# %%
# Import Tfidf vectorizer from sklearn and apply the vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer()
content = sub_news['content']
content_tfidf = vect.fit_transform(content)

# %% [markdown]
# # Training a classifier
# %% [markdown]
# ### Split the data into training, validation and testing sets

# %%
from sklearn.model_selection import train_test_split

# Size of training is 60%
X_train, X_test, y_train, y_test = train_test_split(
    content_tfidf, sub_news['type'], test_size=0.40, random_state=42)

# Use the 40% test set to split further into test and validation set with 50/50 split
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50, random_state=42)

#print(len(y_train))
#print(len(y_val))
#print(len(y_test))


# %%
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define the classifier classes
svc = SVC(kernel='linear')

# Fit the model
svc.fit(X_train,y_train)

# Predict on the test set
svc_pred = svc.predict(X_val)

# Evaluate performance
#print >>fout, "svc accuracy:" + str(accuracy_score(y_val,svc_pred))
print("svc accuracy:"+str(accuracy_score(y_val,svc_pred)), file=fout)

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

names = ["Nearest Neighbors","Random Forest", "Neural Net","LogReg"]

classifiers = [
    KNeighborsClassifier(3),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    DecisionTreeClassifier(max_depth=5),
    LogisticRegression(penalty='l1',solver='saga', max_iter=10000)]

for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        #print(name,' score:',clf.score(X_val, y_val))
        #print >>fout, name + " accuracy:" + str(accuracy_score(y_val,svc_pred))
        print(name+" accuracy:"+str(clf.score(X_val, y_val)), file=fout)