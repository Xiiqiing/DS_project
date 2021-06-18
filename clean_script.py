#!/home/projects/ku_00039/people/zelili/programs/miniconda2/envs/phyluce-1.7.1/bin/python3.6
# -*- coding: utf-8 -*-

# NB: Not for directly run!

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from collections import Counter
#from nltk.stem import WordNetLemmatize
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# define regexp for date
monthsShort = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
monthsLong = "January|February|March|April|May|June|July|August|September|October|November|December"
months = "(" + monthsShort + "|" + monthsLong + ")"
separators = "[-/\s,.]"
days = "\d{2}"
years = "\d{4}"
regex1 = "(" + months + separators + days + "|" + years + ")"
regex2 = "(" + days + separators + months + "|" + years + ")"
regex3 = '^([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])(\.|-|\/)([1-9]|0[1-9]|1[0-2])(\.|-|\/)([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])$|^([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])(\.|-|\/)([1-9]|0[1-9]|1[0-2])(\.|-|\/)([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])$'

def my_clean_text(text):
    #replace multiple \s to single space
    text = re.sub(r'(\\n)+|(\\t)+|\s{2,}', ' ', text).lower()
    #delete email
    text = re.sub(r'^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$', ' ', text)
    #delete URLs
    text = re.sub(r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?', ' ', text)
    #delete date
    text = re.sub(regex1, ' ', text); text = re.sub(regex2, ' ', text); text = re.sub(regex3, ' ', text)
    #remove sign at start or end of the words
    text = re.sub(r'[^\w\s>]+\s|\s+[^\w\s<]', ' ', text)
    #remove in-words sign expect -,<,> and '
    text = re.sub(r'[^\w\s\-<>\']', '', text)
    #replace any numbers
    #text = re.sub(r'[\d+,?]\.?\d*', ' ', text)
    #replace multiple \s to single space again
    text = re.sub(r'\s{2,}', ' ', text)
    #text = re.sub(r'["”“@()*|\'#!≥+.,$€%&"]', ' ', text) #remove spical char
    #lower the text
    text = [w.lower() for w in word_tokenize(text)]
    return text

stop_words = stopwords.words('english')
stopwords_dict = Counter(stop_words)
ps=PorterStemmer()

def process_data(data,col):
  data[col] = data[col].apply(my_clean_text)
  data[col] = data.apply(lambda x: ([w for w in x[col] if not w in stopwords_dict]),axis=1)
  data[col] = data.apply(lambda x: ([ps.stem(w) for w in x[col]]),axis=1)
  data[col] = data[col].apply(lambda x: " ".join(x))
