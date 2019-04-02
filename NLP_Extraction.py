#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
import os


# In[39]:


import re

from nltk.data import load
from nltk.tokenize.casual import TweetTokenizer, casual_tokenize
from nltk.tokenize.mwe import MWETokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.regexp import (
    RegexpTokenizer,
    WhitespaceTokenizer,
    BlanklineTokenizer,
    WordPunctTokenizer,
    wordpunct_tokenize,
    regexp_tokenize,
    blankline_tokenize,
)
from nltk.tokenize.repp import ReppTokenizer
from nltk.tokenize.sexpr import SExprTokenizer, sexpr_tokenize
from nltk.tokenize.simple import (
    SpaceTokenizer,
    TabTokenizer,
    LineTokenizer,
    line_tokenize,
)
from nltk.tokenize.texttiling import TextTilingTokenizer
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize.util import string_span_tokenize, regexp_span_tokenize
from nltk.tokenize.stanford_segmenter import StanfordSegmenter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


# In[3]:


def openfile(path):
    df=open(path,'rb')
    text = df.read()
    return str(text)


# In[4]:


def sentencetokenizer(text):
    tokenizer = load('tokenizers/punkt/{0}.pickle'.format('english'))
    sentences= tokenizer.tokenize(text)
    return sentences


# In[5]:


def remove_stopwords(sentence):
    stop_words = set(stopwords.words('english')) 
      
    word_tokens = word_tokenize(text) 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    return filtered_sentence


# In[6]:



def remove_punctuations(tokens):
    tokens = [w.lower() for w in tokens if w.isalpha()]
    return tokens


# In[7]:


def extract(text):
    pattern = re.compile(r'technical[\s\S]*?education')
    x = re.findall(pattern, text)
    return x


# In[47]:


def lemmetize(text):
    
    lemmatizer = WordNetLemmatizer()
    
    li=[lemmatizer.lemmatize(w) for w in text]
    return li
    


# In[48]:


file_name = []
dir_names = []
for root, dirs, files in os.walk("/home/dheemanth/Downloads/Resume Classification-20190224T141633Z-001/Resume Classification/Training Data"):
    dir_names.append(dirs)
    file_name.append(files)    


# In[49]:


dirs=dir_names[0]


# In[97]:


data = {}
merge_list = []
i = 1
for key in dirs:
    file_name_dict = {}
    for ch in range(len(file_name[i])):
        path = "/home/dheemanth/Downloads/Resume Classification-20190224T141633Z-001/Resume Classification/Training Data/"+key+"/"+file_name[i][ch]
        text = openfile(path)
        text = str(text.lower())
        text = str(extract(text))
        sentences=sentencetokenizer(text)
        tokens=remove_stopwords(sentences)
        tokens = remove_punctuations(tokens)
        tokens = list(set(lemmetize(tokens)))
        merge_list = merge_list +tokens

    data[key] = merge_list
    i+=1


# In[98]:


data


# In[99]:


path = '/home/dheemanth/Downloads/Resume Classification-20190224T141633Z-001/Resume Classification/Training Data/BigData/txt1'


# In[100]:


text = openfile(path)
text = str(text.lower())
text = str(extract(text))
sentences=sentencetokenizer(text)
tokens=remove_stopwords(sentences)
tokens = remove_punctuations(tokens)
tokens1 = lemmetize(tokens)


# In[101]:


tokens1


# In[102]:


len(data['BigData'])


# In[103]:


list_count = []
for i in data['BigData']:
    for j in tokens1:
        if i == j:
            list_count.append(i)


# In[104]:


len(list(set(list_count)))


# In[ ]:




