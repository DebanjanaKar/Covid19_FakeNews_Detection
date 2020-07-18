#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: debanjana
"""


import torch
import numpy as np
import pandas as pd
from simpletransformers.classification import ClassificationModel
import re
import sklearn

torch.manual_seed(1525)
np.random.seed(1525)


# In[ ]:


#loading english data
import pickle as pkl
with open('./resources/covid_en_tweet.pickle', 'rb') as pkl_in:
    tweets_en = pkl.load(pkl_in)
#loading bengali data
with open('./resources/covid_bn_tweet.pickle', 'rb') as pkl_in:
    tweets_bn = pkl.load(pkl_in)
#loading hindi data
with open('./resources/covid_hi_tweet.pickle', 'rb') as pkl_in:
    tweets_hi = pkl.load(pkl_in)


# In[ ]:


#simple text based classification
def fake_classify(train_set, test_set):

    # Create a TransformerModel


    model = ClassificationModel('bert', 'bert-base-multilingual-uncased', args={ 'num_train_epochs': 3, 'overwrite_output_dir': True, 'manual_seed' : 1525}, use_cuda = False)

    # Train the model

    model.train_model(train_set)

    # Evaluate the model

    result, model_outputs, wrong_predictions = model.eval_model(test_set, f1=sklearn.metrics.f1_score, acc=sklearn.metrics.accuracy_score)
    
    
    return model, result, model_outputs, wrong_predictions


# In[ ]:


def results(result):
    prec = result['tp']/(result['tp'] + result['fp'])
    rec = result['tp']/(result['tp'] + result['fn'])
    fscore = (2*prec*rec)/(prec + rec)
    print('Raw result = ', result)
    print('Precision = ', prec )
    print('Recall = ', rec)
    print('F-Score = ', fscore) 
    return fscore


# In[ ]:


path_multi = './models/model_debanjana/multi_model'


# In[ ]:


del tweets_en['text_info']
df_en = pd.DataFrame(tweets_en)
print(df_en.head())
#train_set_en, eval_set_en, test_set_en = split(df_en)


# In[ ]:


#classification on bengali tweets
del tweets_bn['text_info']
df_bn = pd.DataFrame(tweets_bn)
print(df_bn)


# In[ ]:


#classification on hindi tweets
del tweets_hi['text_info']
df_hi = pd.DataFrame(tweets_hi)
print(df_hi)


# In[ ]:


with open('./resources/covid_bn_tweet_test.pickle', 'rb') as pkl_in:
    tweets_bn_test = pkl.load(pkl_in)
del tweets_bn_test['text_info']
df_bn_test = pd.DataFrame(tweets_bn_test)
print(df_bn_test)


# In[ ]:


#multilingual model
frames = [df_en, df_bn, df_hi, df_bn_test]
df_merged = pd.concat(frames)
df_merged.index = range(len(df_merged))   #change indices
print(len(df_merged))


# In[ ]:


#multilingual results
from sklearn.model_selection import KFold
import torch

best_result = 0
kf = KFold(n_splits=5)
model_outputs_multi = {}
count = 1
for train, test in kf.split(df_merged):
    print('--------------------------', count, '------------------------------')
    #print("%s %s" % (train, test))
    df_train_multi = df_merged.copy()
    df_test_multi = df_merged.copy()
    df_train_multi = df_train_multi.drop(test)
    df_test_multi = df_test_multi.drop(train)
    print(len(df_train_multi), len(df_test_multi))
    
    model_multi, result_multi, model_outputs_multi, wrong_predictions_multi = fake_classify(df_train_multi, df_test_multi)
    
    fscr = results(result_multi)
    if fscr > best_result:
        best_result = fscr
        torch.save(model_multi, path_multi)
    
    model_outputs_multi[count] = {}
    model_outputs_multi[count]['indices'] = test
    model_outputs_multi[count]['outputs'] = model_outputs_multi 


# In[ ]:


#storing model outputs of mono and multilingual models
with open('./resources/multi_raw_outputs.pickle', 'wb') as pkl_out:
    pkl.dump(model_outputs_multi, pkl_out)

