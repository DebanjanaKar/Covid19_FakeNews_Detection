
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
with open('./resources/covid_en_tweet.pickle', 'rb') as pkl_in:
    tweets = pkl.load(pkl_in)
#loading bengali data
with open('./resources/covid_bn_tweet.pickle', 'rb') as pkl_in:
    tweets_bn = pkl.load(pkl_in)
with open('./resources/covid_bn_tweet_test.pickle', 'rb') as pkl_in:
    tweets_bn_test = pkl.load(pkl_in)
#loading hindi data
with open('./resources/covid_hi_tweet.pickle', 'rb') as pkl_in:
    tweets_hi = pkl.load(pkl_in)
with open('./resources/covid_hi_tweet_test.pickle', 'rb') as pkl_in:
    tweets_hi_test = pkl.load(pkl_in)

#train - test split
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


del tweets['text_info']
df = pd.DataFrame(tweets)
print(df.head())

#simple text based classification
def fake_classify(train_set, eval_set, test_set, seed):

    # Create a TransformerModel

    model = ClassificationModel('bert', 'bert-base-multilingual-uncased', args={ 'num_train_epochs': 3, 'overwrite_output_dir': True, 'manual_seed' : seed}, use_cuda = True)
    print(model.args)
    # Train the model
    model.train_model(train_set)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(test_set, f1=sklearn.metrics.f1_score, acc=sklearn.metrics.accuracy_score)
    #print('Evaluation results = ', results(results))

    #save the model
    
    #import torch
    #torch.save(model, path) --> no need to do this, model gets saved in output dir

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

#classification on english tweets
train_set, eval_set, test_set = split(df, seed)
result, model_outputs, wrong_predictions = fake_classify(train_set, eval_set, test_set, path_en2, seed)
print('--------------------------------')
print('Classification Results English: ')
print(results(result))

#classification on bengali tweets
del tweets_bn['text_info']
del tweets_bn_test['text_info']
df_bn_fresh = pd.DataFrame(tweets_bn)
df_bn_test = pd.DataFrame(tweets_bn_test)
df_bn = pd.concat([df_bn_fresh, df_bn_test])
df_bn.index = range(len(df_bn))   #change indices
train_set_bn, eval_set_bn, test_set_bn = split(df_bn, seed)
result_bn, model_outputs, wrong_predictions = fake_classify(train_set_bn, eval_set_bn, test_set_bn, seed)
print('--------------------------------')
print('Classification Results Bengali : ')
results(result_bn)


#classification on hindi tweets
del tweets_hi['text_info']
del tweets_hi_test['text_info']
df_hi_fresh = pd.DataFrame(tweets_hi)
df_hi_test = pd.DataFrame(tweets_hi_test)
df_hi = pd.concat([df_hi_fresh, df_hi_test]) 
df_hi.index = range(len(df_hi))   #change indices
train_set_hi, eval_set_hi, test_set_hi = split(df_hi, seed)
result_hi, model_outputs, wrong_predictions = fake_classify(train_set_hi, eval_set_hi, test_set_hi, seed)
print('--------------------------------')
print('Classification Results Hindi : ')
results(result_hi)


#crosslingual model : train = en + hi, test = bn
frames = [df, df_hi]
df_merged_hi = pd.concat(frames)
df_merged_hi.index = range(len(df_merged_hi))   #change indices
#train_set_merge_hi, eval_set_merge_hi = split(df_merged_hi, seed)
train_set_merge_hi = df_merged_hi
test_set_merge_hi = df_bn
eval_set_merge_hi = []
#print(len(df_merged_hi), len(train_set_merge_hi), len(eval_set_merge_hi), len(test_set_merge_hi))
result_cross_hi, model_outputs_enhi, wrong_predictions = fake_classify(train_set_merge_hi, eval_set_merge_hi, test_set_merge_hi, seed)
print('--------------------------------')
print('Classification Results train = en + hi, test = bn : ')
results(result_cross_hi)

#crosslingual model : train = en + bn, test = hi
frames = [df, df_bn]
df_merged_bn = pd.concat(frames)
df_merged_bn.index = range(len(df_merged_bn))   #change indices
#train_set_merge_bn, eval_set_merge_bn = split(df_merged_bn, seed)
train_set_merge_bn = df_merged_bn
test_set_merge_bn = df_hi
eval_set_merge_bn = []
#print(len(df_merged_bn), len(train_set_merge_bn), len(eval_set_merge_bn), len(test_set_merge_bn))
result_cross_bn, model_outputs_bnen, wrong_predictions = fake_classify(train_set_merge_bn, eval_set_merge_bn, test_set_merge_bn, seed)
print('--------------------------------')
print('Classification Results train = en + bn, test = hi: ')
results(result_cross_bn)


#crosslingual model : train = bn + hi, test = en
frames = [df_bn, df_hi]
df_merged_en = pd.concat(frames)
df_merged_en.index = range(len(df_merged_en))   #change indices
#train_set_merge_en, eval_set_merge_en = split(df_merged_en, seed)
train_set_merge_en = df_merged_en
test_set_merge_en = df
eval_set_merge_en = []
#print(len(df_merged_en), len(train_set_merge_en), len(eval_set_merge_en), len(test_set_merge_en))

result_cross_en, model_outputs_bnhi, wrong_predictions = fake_classify(train_set_merge_en, eval_set_merge_en, test_set_merge_en, seed)
print('--------------------------------')
print('Classification Results train = bn + hi, test = en : ')
results(result_cross_en)

#multilingual model
frames = [df, df_bn, df_hi]
df_merged = pd.concat(frames)
df_merged.index = range(len(df_merged))   #change indices
train_set_merge, eval_set_merge, test_set_merge = split(df_merged, seed)
result_multi, model_outputs, wrong_predictions = fake_classify(train_set_merge, eval_set_merge, test_set_merge, seed)
print('--------------------------------')
print('Classification Results Multilingual: ')
results(result_multi)

#multilingual model with individual language results
train_set_en, eval_set_en, test_set_en = split(df, seed)
train_set_bn, eval_set_bn, test_set_bn = split(df_bn, seed)
train_set_hi, eval_set_hi, test_set_hi = split(df_hi, seed)

df_train = pd.concat([train_set_en, train_set_bn, train_set_hi])
df_train.index = range(len(df_train))

result, model_outputs, wrong_predictions = fake_classify(df_train, eval_set_en, test_set_en, seed)
print('--------------------------------')
print('Classification Results train = en + bn + hi, test = en : ')
results(result)

result, model_outputs, wrong_predictions = fake_classify(df_train, eval_set_en, test_set_bn, seed)
print('--------------------------------')
print('Classification Results train = en + bn + hi, test = bn : ')
results(result) 

result, model_outputs, wrong_predictions = fake_classify(df_train, eval_set_en, test_set_hi, seed)
print('--------------------------------')
print('Classification Results train = en + bn + hi, test = hi : ')
results(result) 

df_test = pd.concat([test_set_en, test_set_bn, test_set_hi])
df_test.index = range(len(df_test))
result, model_outputs, wrong_predictions = fake_classify(df_train, eval_set_en, df_test, seed)
print('--------------------------------')
print('Classification Results train = en + bn + hi, test = en + bn + hi : ')
results(result) 


#storing model outputs of mono and multilingual models
with open('./resources/multi_raw_outputs.pickle', 'wb') as pkl_out:
    pkl.dump(model_outputs, pkl_out)
#with open('./resources/en_raw_outputs.pickle', 'wb') as pkl_out:
    pkl.dump(en_model_outputs, pkl_out)
with open('./resources/bn_raw_outputs.pickle', 'wb') as pkl_out:
    pkl.dump(bn_model_outputs, pkl_out)
with open('./resources/hi_raw_outputs.pickle', 'wb') as pkl_out:
    pkl.dump(hi_model_outputs, pkl_out)
