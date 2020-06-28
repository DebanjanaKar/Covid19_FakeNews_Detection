#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv('./data/toxicity/train.csv')
df.head()


# In[ ]:


df['labels'] = list(zip(df.toxic.tolist(), df.severe_toxic.tolist(), df.obscene.tolist(), df.threat.tolist(),  df.insult.tolist(), df.identity_hate.tolist()))
df['text'] = df['comment_text'].apply(lambda x: x.replace('\n', ' '))


# In[ ]:


from sklearn.model_selection import train_test_split

train_df, eval_df = train_test_split(df, test_size=0.2)


# In[ ]:


from simpletransformers.classification import MultiLabelClassificationModel

model = MultiLabelClassificationModel('bert', 'bert-base-multilingual-uncased', num_labels=6, args={'train_batch_size':2, 'gradient_accumulation_steps':16, 'learning_rate': 3e-5, 'num_train_epochs': 3, 'max_seq_length': 512, 'manual_seed' : 1525, 'overwrite_output_dir' : True})

model.train_model(train_df)

import torch

torch.save(model, './models/models_debanjana/toxicity')


# In[ ]:


def results(result):
    prec = result['tp']/(result['tp'] + result['fp'])
    rec = result['tp']/(result['tp'] + result['fn'])
    fscore = (2*prec*rec)/(prec + rec)
    print('Raw result = ', result)
    print('Precision = ', prec )
    print('Recall = ', rec)
    print('F-Score = ', fscore) 


# In[ ]:


# Evaluate the model

result, model_outputs, wrong_predictions = model.eval_model(eval_df, f1=sklearn.metrics.f1_score, acc=sklearn.metrics.accuracy_score)
results(result)


# In[ ]:


use_cuda("=", "False")
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


#extract features / predict
del tweets_en['text_info']
df_en = pd.DataFrame(tweets_en)
print(df_en.head())
#train_set_en, eval_set_en, test_set_en = split(df_en)

#classification on bengali tweets
del tweets_bn['text_info']
df_bn = pd.DataFrame(tweets_bn)
print(df_bn)

#classification on hindi tweets
del tweets_hi['text_info']
df_hi = pd.DataFrame(tweets_hi)
print(df_hi)

with open('./resources/covid_bn_tweet_test.pickle', 'rb') as pkl_in:
    tweets_bn_test = pkl.load(pkl_in)
del tweets_bn_test['text_info']
df_bn_test = pd.DataFrame(tweets_bn_test)
df_bn_test

#multilingual model
frames = [df_en, df_bn, df_hi, df_bn_test]
df_merged = pd.concat(frames)
df_merged.index = range(len(df_merged))   #change indices
df_merged

del df_merged['Labels']

preds, outputs = model.predict(df_merged)

print('Predictions-----------------')

print(preds)

