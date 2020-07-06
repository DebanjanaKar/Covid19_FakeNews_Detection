#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

torch.manual_seed(1)
np.random.seed(1)

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

#predict    
sent = input()
predict(path_en, sent) #path_en is path to the model

