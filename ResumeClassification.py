#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.getcwd()


# In[2]:


import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
import sklearn.datasets as skds
from pathlib import Path


# In[3]:


# For reproducibility
np.random.seed(1237)

# Source file directory
path_train = "/Users/pradeep/Documents/Project/resumes/Training Data"
files_train = skds.load_files(path_train,load_content=False)


# In[4]:


label_index = files_train.target
label_names = files_train.target_names
labelled_files = files_train.filenames

data_tags = ["filename","branch","content"]
data_list = []


# In[5]:


i=0
for f in labelled_files:
    data_list.append((f,label_names[label_index[i]],Path(f).read_text(encoding = 'utf-8',errors = 'ignore')))
    i += 1


# In[6]:


# We have training data available as dictionary filename, category, data
data = pd.DataFrame.from_records(data_list, columns=data_tags)
data


# In[7]:


# lets take 80% data as training and remaining 20% for test.
train_size = int(len(data) * .8)
 
train_posts = data['content'][:train_size]
train_tags = data['branch'][:train_size]
train_files_names = data['filename'][:train_size]
 
test_posts = data['content'][train_size:]
test_tags = data['branch'][train_size:]
test_files_names = data['filename'][train_size:]


# In[8]:


# 20 news groups
num_labels = 3
vocab_size = 13685
batch_size = 100
 
# define Tokenizer with Vocab Size
tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(train_posts)

x_train = tokenizer.texts_to_matrix(train_posts, mode='tfidf')
x_test = tokenizer.texts_to_matrix(test_posts, mode='tfidf')
 
encoder = LabelBinarizer()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)


# In[9]:


encoder = LabelBinarizer()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)


# In[10]:


model = Sequential()
model.add(Dense(512, input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
# model.add(Dense(512))
# model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(num_labels))
model.add(Activation('sigmoid'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=30,
                    verbose=1,
                    validation_split=0.1)


# In[176]:


y_pred = model.predict(x_test)
y_pred = y_pred > 0.5
# y_pred
len(x_test)


# In[11]:


score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
 
print('Test accuracy:', score[1])
 
text_labels = encoder.classes_
 
for i in range(10):
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = text_labels[np.argmax(prediction[0])]
    print(test_files_names.iloc[i])
    print('Actual label:' + test_tags.iloc[i])
    print("Predicted label: " + predicted_label)


# In[14]:


# These are the labels we stored from our training
# The order is very important here.
 
labels = np.array(['Big Data','Embedded Systems', 'VLSI'])
c = 1
for i in range(75):  
    test_files = ["/Users/pradeep/Documents/Project/resumes/Test Data/resume.bda/txt" + str(c)]
    x_data = []

    for t_f in test_files:
        t_f_data = Path(t_f).read_text(encoding ='utf-8',errors ='ignore')
        x_data.append(t_f_data)

    x_data_series = pd.Series(x_data)
    x_tokenized = tokenizer.texts_to_matrix(x_data_series, mode='tfidf')

    i=0
    for x_t in x_tokenized:
        prediction = model.predict(np.array([x_t]))
        predicted_label = labels[np.argmax(prediction[0])]
        print("File ->", test_files[i], "Predicted label: " + predicted_label)
        i += 1
    
    c += 1


# In[ ]:




