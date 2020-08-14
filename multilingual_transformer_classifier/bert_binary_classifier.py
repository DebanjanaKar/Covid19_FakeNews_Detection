
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

#train_set, eval_set, test_set = split(df)

#simple text based classification
#very useful library : https://towardsdatascience.com/simple-transformers-introducing-the-easiest-bert-roberta-xlnet-and-xlm-library-58bf8c59b2a3
def fake_classify(train_set, eval_set, test_set, seed):

    # Create a TransformerModel

    model = ClassificationModel('bert', 'bert-base-multilingual-uncased', args={ 'num_train_epochs': 3, 'overwrite_output_dir': True, 'manual_seed' : seed}, use_cuda = True)
    print(model.args)
    # Train the model
    model.train_model(train_set)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(test_set, f1=sklearn.metrics.f1_score, acc=sklearn.metrics.accuracy_score)
    #r, m, p =  model.eval_model(eval_set, f1=sklearn.metrics.f1_score, acc=sklearn.metrics.accuracy_score)
    #print('Evaluation results = ', results(results))

    #save the model
    
    #import torch
    #torch.save(model, path)

    return result, model_outputs, wrong_predictions


def results(result):
    prec = result['tp']/(result['tp'] + result['fp'])
    rec = result['tp']/(result['tp'] + result['fn'])
    fscore = (2*prec*rec)/(prec + rec)
    print('Raw result = ', result)
    print('Precision = ', prec )
    print('Recall = ', rec)
    print('F-Score = ', fscore) 


path_en = './resources/en_model_seed0'
path_en1 = './resources/en_model_seed25'
path_en2 = './resources/en_model_seed76'
path_en3 = './resources/en_model_seed128'
path_en4 = './resources/en_model_seed512'
path_en5 = './resources/en_model_seed64'
path_en6 = './resources/en_model_seed32'
path_en7 = './resources/en_model_seed1000'
path_en8 = './resources/en_model_seed7634'
path_en9 = './resources/en_model_seed1'

'''
seed = 64
set_seed(seed)
#english results
train_set, eval_set, test_set = split(df, seed)
result, model_outputs, wrong_predictions = fake_classify(train_set, eval_set, test_set, path_en, seed)
print('--------------------------------')
print('Classification Results with seed ' + str(seed) + ': ')
print(results(result))


seed = 25
set_seed(seed)
#english results
train_set, eval_set, test_set = split(df, seed)
result, model_outputs, wrong_predictions = fake_classify(train_set, eval_set, test_set, path_en1, seed)
print('--------------------------------')
print('Classification Results with seed ' + str(seed) + ': ')
print(results(result))

seed = 76
set_seed(seed)
#english results
train_set, eval_set, test_set = split(df, seed)
result, model_outputs, wrong_predictions = fake_classify(train_set, eval_set, test_set, path_en2, seed)
print('--------------------------------')
print('Classification Results with seed ' + str(seed) + ': ')
print(results(result))

seed = 128
set_seed(seed)
#english results
train_set, eval_set, test_set = split(df, seed)
result, model_outputs, wrong_predictions = fake_classify(train_set, eval_set, test_set, path_en3, seed)
print('--------------------------------')
print('Classification Results with seed ' + str(seed) + ': ')
print(results(result))

seed = 512
set_seed(seed)
#english results
train_set, eval_set, test_set = split(df, seed)
result, model_outputs, wrong_predictions = fake_classify(train_set, eval_set, test_set, path_en4, seed)
print('--------------------------------')
print('Classification Results with seed ' + str(seed) + ': ')
print(results(result))

seed = 64
set_seed(seed)
#english results
train_set, eval_set, test_set = split(df, seed)
result, model_outputs, wrong_predictions = fake_classify(train_set, eval_set, test_set, path_en5, seed)
print('--------------------------------')
print('Classification Results with seed ' + str(seed) + ': ')
print(results(result))

seed = 32
set_seed(seed)
#english results
train_set, eval_set, test_set = split(df, seed)
result, model_outputs, wrong_predictions = fake_classify(train_set, eval_set, test_set, path_en6, seed)
print('--------------------------------')
print('Classification Results with seed ' + str(seed) + ': ')
print(results(result))

seed = 1000 #english results
set_seed(seed)
train_set, eval_set, test_set = split(df, seed)
result, model_outputs, wrong_predictions = fake_classify(train_set, eval_set, test_set, path_en7, seed)
print('--------------------------------')
print('Classification Results with seed ' + str(seed) + ': ')  
print(results(result))

seed = 7634
set_seed(seed)
train_set, eval_set, test_set = split(df, seed)
outputs, wrong_predictions = fake_classify(train_set, eval_set, test_set, path_en8, seed)
print('--------------------------------')
print('Classification Results with seed ' + str(seed) + ': ')  
print(results(result))

seed = 1
set_seed(seed)
train_set, eval_set, test_set = split(df, seed)
outputs, wrong_predictions = fake_classify(train_set, eval_set, test_set, path_en9, seed)                     
print('--------------------------------')
print('Classification Results with seed ' + str(seed) + ': ')
print(results(result))
# #### Results :
# -------------------------------------------------------------------------------------------------------Prec----Recall-----Fscore
# 1. With my preprocessing + 1 epoch on Covid19 dataset + no validation = 67.60,  87.27,     76.19
# 2. With almost infodemic preprocessing + 3 epochs + validation = 75.86, 78.57, 77.19

'''
seed = 76

set_seed(seed)

#classification on bengali tweets
del tweets_bn['text_info']
del tweets_bn_test['text_info']
df_bn_fresh = pd.DataFrame(tweets_bn)
df_bn_test = pd.DataFrame(tweets_bn_test)
df_bn = pd.concat([df_bn_fresh, df_bn_test])
df_bn.index = range(len(df_bn))   #change indices
'''
train_set_bn, eval_set_bn, test_set_bn = split(df_bn, seed)


result_bn, model_outputs, wrong_predictions = fake_classify(train_set_bn, eval_set_bn, test_set_bn, seed)
print('--------------------------------')
print('Classification Results Bengali : ')
results(result_bn)
'''

#classification on hindi tweets
del tweets_hi['text_info']
del tweets_hi_test['text_info']
df_hi_fresh = pd.DataFrame(tweets_hi)
df_hi_test = pd.DataFrame(tweets_hi_test)
df_hi = pd.concat([df_hi_fresh, df_hi_test]) 
df_hi.index = range(len(df_hi))   #change indices
'''
train_set_hi, eval_set_hi, test_set_hi = split(df_hi, seed)

result_hi, model_outputs, wrong_predictions = fake_classify(train_set_hi, eval_set_hi, test_set_hi, seed)
print('--------------------------------')
print('Classification Results Hindi : ')
results(result_hi)


#multilingual model
frames = [df, df_bn, df_hi]
df_merged = pd.concat(frames)
df_merged.index = range(len(df_merged))   #change indices
train_set_merge, eval_set_merge, test_set_merge = split(df_merged, seed)


result_multi, model_outputs, wrong_predictions = fake_classify(train_set_merge, eval_set_merge, test_set_merge, seed)
print('--------------------------------')
print('Classification Results Multilingual: ')
results(result_multi)




#crosslingual model
frames = [df, df_bn]
df_merged_bn = pd.concat(frames)
df_merged_bn.index = range(len(df_merged_bn))   #change indices
train_set_merge_bn, eval_set_merge_bn = split(df_merged_bn, seed)
test_set_merge_bn = df_hi
print(len(df_merged_bn), len(train_set_merge_bn), len(eval_set_merge_bn), len(test_set_merge_bn))
result_cross_bn, model_outputs, wrong_predictions = fake_classify(train_set_merge_bn, eval_set_merge_bn, test_set_merge_bn, seed)
print('--------------------------------')
print('Classification Results : ')
results(result_cross_bn)


#storing model outputs of mono and multilingual models
with open('./resources/multi_raw_outputs.pickle', 'wb') as pkl_out:
    pkl.dump(model_outputs, pkl_out)
#with open('./resources/en_raw_outputs.pickle', 'wb') as pkl_out:
    pkl.dump(en_model_outputs, pkl_out)
with open('./resources/bn_raw_outputs.pickle', 'wb') as pkl_out:
    pkl.dump(bn_model_outputs, pkl_out)
with open('./resources/hi_raw_outputs.pickle', 'wb') as pkl_out:
    pkl.dump(hi_model_outputs, pkl_out)
'''
'''
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

test_set_merge_hi = df_bn_test
#print(len(df_merged_hi), len(train_set_merge_hi), len(eval_set_merge_hi), len(test_set_merge_hi))
result_cross_hi, model_outputs_enhi, wrong_predictions = fake_classify(train_set_merge_hi, eval_set_merge_hi, test_set_merge_hi, seed)
print('--------------------------------')
print('Classification Results train = en + hi, test = bn fresh annotations : ')
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

test_set_merge_bn = df_hi_test
#print(len(df_merged_bn), len(train_set_merge_bn), len(eval_set_merge_bn), len(test_set_merge_bn))
result_cross_bn, model_outputs_bnen, wrong_predictions = fake_classify(train_set_merge_bn, eval_set_merge_bn, test_set_merge_bn, seed)
print('--------------------------------')
print('Classification Results train = en + bn, test = hi: fresh annotations ')
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
'''

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

'''
# In[15]:

def preprocess(tweet):
    tweet = tweet.lower()
    url = r'http\S+'
    tweet = re.sub(url, 'URL', tweet, flags=re.MULTILINE)
    emoji = re.compile("["         u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002500-\U00002BEF"  # chinese char
                                   u"\U00002702-\U000027B0"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"  # dingbats
                                   u"\u3030"
                                   "]+", flags=re.UNICODE)
    tweet =  emoji.sub(r'', tweet)
    tweet = ' '.join([word[1:] if word[0] == '#' else word for word in tweet.split()])
    return tweet


# In[16]:

#getting predictions on real tweets
def predict(path, sent):
    model = torch.load(path)
    sent = preprocess(sent)
    p, ro = model.predict([sent])
    c1 = np.exp(ro[0][0])/sum([np.exp(val) for val in ro[0]])
    c2 = np.exp(ro[0][1])/sum([np.exp(val) for val in ro[0]])
    result = 'This tweet has a verifiable claim.' if p[0] == 1 else 'This tweet does not have a verifiable claim.'
    cscore = c2*100 if p[0] == 1 else c1*100
    print(sent, ' : ', result)
    print('The model says this with a',round(cscore, 2), '% confidence score.')


# In[17]:

#predict english tweets
sent = input()
predict(path_en, sent)


# In[18]:

#predict english tweets
sent = input()
predict(path_en, sent)


# In[ ]:




# In[19]:

#predict bengali tweets
sent = input()
predict(path_bn, sent)


# In[ ]:

#predict hindi tweets #example from BBC News Hindi
sent = input()
predict(path_hi, sent)


# In[20]:

#predict multilingual tweets #example from DW Bangla account
sent = input()
predict(path_multi, sent)
'''
