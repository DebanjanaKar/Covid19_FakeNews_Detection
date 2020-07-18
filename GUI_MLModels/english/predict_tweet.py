import private
import re
import tweepy
import difflib
import pandas as pd
import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt
from datetime import date
from sklearn.externals import joblib
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

auth = tweepy.OAuthHandler(private.CONSUMER_KEY, private.CONSUMER_SECRET)
auth.set_access_token(private.OAUTH_TOKEN, private.OAUTH_TOKEN_SECRET)
api = tweepy.API(auth)



def preprocess_tweet(tweet):

    ### Structure of the list returned should look like this
    ### [cleaned_tweet, length_before, length_after, ratio_of_length, number_of_upper_chars, has_question_marks, number_of_question_marks, has_exclamations, number_of_exclamation_marks]

    final_tweet = []

    length_before               = len(tweet)
    number_of_upper_chars       = 0
    number_of_question_marks    = 0
    number_of_exclamation_marks = 0

    for ch in tweet:
        number_of_upper_chars         += int(ch.isupper())
        number_of_question_marks      += int(ch == '?')
        number_of_exclamation_marks   += int(ch == '!')

    has_question_marks    = int(number_of_question_marks>0)
    has_exclamation_marks = int(number_of_exclamation_marks>0)


    tweet = tweet.lower()
    url = r'http\S+'
    tweet = re.sub(url, ' ', tweet, flags=re.MULTILINE)
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

    tweet = tweet.split()
    exclude_char_list = ['@','rt','|','%', '[', '(', '{', '}', ']', ')','url','URL']
    tweet = [word for word in tweet if all(ch not in word for ch in exclude_char_list)]
    tweet = ' '.join(tweet)
    


    length_after = len(tweet)
    length_ratio = length_after/length_before

    final_tweet = [tweet, length_before, length_after, length_ratio, number_of_upper_chars,has_question_marks, number_of_question_marks, has_exclamation_marks, number_of_exclamation_marks]

    # for item in final_tweet:
        # print(item)

    return final_tweet
    


def calculate_days_past(tweet_time):
    #Calculating Dates
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    a1 = int(tweet_time[-1])
    b1 = int(months.index(tweet_time[1])+1)
    c1 = int(tweet_time[2])

    #Todays date
    today = date.today()
    life  =  str(date(today.year,today.month,today.day) - date(a1,b1,c1))
    if(len(life)>7):
        life  = life.split(" ")
        life  = life[0]
        life  = int(life)
        return life
    else:
        return 0





def get_user_info(tweet):
    
    tweet_text   = tweet.full_text
    real_name    = tweet._json['user']['name']
    user_handle  = tweet._json['user']['screen_name']
    tweet_id     = tweet._json['id']
    user_id      = tweet._json['user']['id']
    
    user_info = api.get_user(user_id,user_mode='extended')
    desc = user_info._json['description']
    
    try:
        expanded_url = user_info._json['entities']['url']['urls'][0]['expanded_url']
        expanded_url = 1
    except:
        expanded_url = 0

    
    tweet_link             = "http://twitter.com/anyuser/status/" + str(tweet.id)
    chars_in_desc          = len(desc)
    chars_in_real_name     = len(real_name)
    chars_in_user_handle   = len(user_handle)
    num_matches            = len(difflib.SequenceMatcher(None, real_name,user_handle).get_matching_blocks())
    total_urls_in_desc     = len(user_info._json['entities']['description']['urls'])
    official_url_exists    = expanded_url
    followers_count        = user_info._json['followers_count']
    friends_count          = user_info._json['friends_count']
    listed_count           = user_info._json['listed_count']
    favourites_count       = user_info._json['favourites_count']
    geo_enabled            = int(user_info._json['geo_enabled'])
    acc_created_on         = user_info._json['created_at'].split()
    acc_life               = calculate_days_past(acc_created_on)
    verified               = int(user_info._json['verified'])
    num_tweets             = user_info._json['statuses_count']
    protected              = int(user_info._json['protected'])
    avg_likes_per_tweet    = favourites_count/num_tweets
    latest_tweet_time      = user_info._json['status']['created_at'].split()
    activity               = calculate_days_past(latest_tweet_time)

    if(acc_life==0):
        posting_frequency = num_tweets
    else:
        posting_frequency      = num_tweets/acc_life

    
    if(friends_count!=0):
        follower_friends_ratio = followers_count/friends_count
    else:
        follower_friends_ratio = 9999999
    
    list = [tweet_text, tweet_link, desc,
            real_name, user_handle, user_id,
            chars_in_desc, chars_in_real_name,
            chars_in_user_handle, num_matches,
            total_urls_in_desc, official_url_exists,
            followers_count, friends_count,
            listed_count, favourites_count,
            geo_enabled, acc_life, verified, num_tweets,
            protected, posting_frequency, activity,
            avg_likes_per_tweet, follower_friends_ratio]
    
   
    return list






#path = r"C:\Users\Mohit Bhardwaj\Downloads\Covid Project\Dataset\Processed Infodemic Dataset\bert_cls datasets\bert_cls_embeddings_text_user_df.pkl"
#f = open(path,'rb')
#df = pickle.load(f)
#f.close()
#
#
#x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,:-1].values, df.iloc[:,-1].values, test_size=0.2, random_state=42)
#
#print(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_test.shape)
#
#
##Applying Random Forest Classifier
#
#rfc = RandomForestClassifier(n_estimators=400).fit(x_train, y_train)
#
#path = r"C:\Users\Mohit Bhardwaj\Downloads\Covid Project\GUI\CFTD Demo\model\rfc_bert_cls_embeddings_text_user.pkl"
#
##Dumping the Random Forest Model
#joblib.dump(rfc,path)
#rfc = joblib.load(path)
#
#
#print("\n\nAccuracy report: ")
#predict_test = rfc.predict(x_test)
#print(classification_report(predict_test,y_test))
#print(np.unique(predict_test,return_counts = True))

#tweet_id = "1280170316447141888"
#tweet_id = "1280376260833787904"

#tweet_id = "1280183728753577990"

#tweet_id = "1280224654406430720"
#tweet_id = "1232532280884645888"
    
#Politics
#tweet_id = "1262434480528207873"
#tweet_id = "1232528385370251264"
#tweet_id = "1236588880389746688"

#Exams
#tweet_id = "1280198734656073728"
#tweet_id = "1280206047894880258"
#tweet_id = "1280213106753441792"
    
tweet_id = "1275848474987311107"
#tweet_id = "1280170462044012545"
#tweet_id = "1279806219767562244"

#tweet_id = "http://twitter.com/anyuser/status/1280224654406430720"

def predict(tweet_id):
    
    if( tweet_id.split("/")[0] == 'http:' or tweet_id.split("/")[0] == 'https:'):
        tweet_id = int(tweet_id.split("/")[-1])

    
    tweet = api.get_status(tweet_id,tweet_mode='extended')
    
    user_info = get_user_info(tweet)
    tweet_info = preprocess_tweet(tweet.full_text)
    
    
    bert_cls = SentenceTransformer('bert-base-nli-cls-token')
    tweet_embeddings = bert_cls.encode([tweet_info[0]])
    
    
    final_encoding = []
    for item in tweet_embeddings[0]:
        final_encoding.append(item)
    
    final_encoding.append(tweet.retweet_count)
    final_encoding.append(tweet.favorite_count)
    
    for item in tweet_info[1:]:
        final_encoding.append(item)
    
    for item in user_info[6:]:
        final_encoding.append(item)
    
    
    final_encoding = np.asarray(final_encoding)
    
    path = r"C:\Users\Mohit Bhardwaj\Downloads\Covid Project\GUI\CFTD Demo\model\rfc_bert_cls_embeddings_text_user.pkl"
    rfc = joblib.load(path)
    
    result = rfc.predict_proba(final_encoding.reshape(1,-1))
    
#    print(result)
    
    if(result[0][0]<=0.65):
        answer = "The tweet --> " + "https://twitter.com/anyuser/status/" + tweet.id_str + "\n\n" + tweet_info[0] + " is: Real"
    else:
        answer = "The tweet --> " + "https://twitter.com/anyuser/status/" + tweet.id_str + "\n\n" + tweet_info[0] + " is: Fake"
        

    return answer










