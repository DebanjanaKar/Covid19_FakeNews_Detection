"""

@author : debanjana

"""

# coding: utf-8

import torch
import numpy as np
import pandas as pd
from simpletransformers.classification import ClassificationModel
import re
import sklearn

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


#loading english data
import pickle as pkl

df_user = pd.read_csv('./resources/en_dataset_tf.csv', index_col=0)
df_bias = pd.read_csv('./resources/en_dataset_bias.csv', index_col=0)
df_link = pd.read_csv('./resources/bn_dataset_link.csv', index_col=0)
df_linkbias = pd.read_csv('./resources/en_dataset_user+tf.csv', index_col=0)
df_all = pd.read_csv('./resources/en_dataset_userft+tf+link+bias.csv', index_col=0)

df_linkbias = df_linkbias.rename(columns={'label': 'labels'})

def split(df, seed):
    df_copy = df.copy()
    train_set = df_copy.sample(frac=0.80, random_state=seed)
    print(len(train_set), train_set.head())
    test_set_split = df_copy.drop(train_set.index)
    #print('-------', len(train_set.index), len(df_copy), len(df_copy) - len(train_set.index), len(test_set))
    eval_set = test_set_split.sample(frac=0.50, random_state=seed)
    print(len(eval_set), eval_set.head())
    test_set_split = test_set_split.drop(eval_set.index)
    print(len(test_set_split), test_set_split.head())
    
    return train_set, eval_set, test_set_split


def fake_classify(train_set, eval_set, test_set, seed):

    # Create a TransformerModel

    model = ClassificationModel('bert', 'bert-base-multilingual-uncased', args={ 'max_seq_length' : 512, 'num_train_epochs': 3, 'overwrite_output_dir': True, 'manual_seed' : seed}, use_cuda = True)
    print(model.args)
    
    # Train the model
    model.train_model(train_set)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(test_set, f1=sklearn.metrics.f1_score, acc=sklearn.metrics.accuracy_score)
    #print('Evaluation results = ', results(results))

    return result, model_outputs, wrong_predictions


def results(result):
    prec = result['tp']/(result['tp'] + result['fp'])
    rec = result['tp']/(result['tp'] + result['fn'])
    fscore = (2*prec*rec)/(prec + rec)
    print('Raw result = ', result)
    print('Precision = ', prec )
    print('Recall = ', rec)
    print('F-Score = ', fscore)



seed = 76
set_seed(seed)

# results
train_set, eval_set, test_set = split(df_link, seed)
result, model_outputs, wrong_predictions = fake_classify(train_set, eval_set, test_set, seed)
print('--------------------------------')
print('Classification Results with Text + User + text ft: ')
print(results(result))
                                                                                                         
