# Covid19_FakeNews_Detection



This package aims to detect fake tweets regarding Covid-19 from the real ones in real-time. We use an AI based technique to process the tweet text and use it, along with user features, to classify the tweets. We are handling tweets in three different languages: English, Hindi and Bengali. The details are as follows:

1. DATASET USED:
We use the Infodemic dataset (https://github.com/firojalam/COVID-19-tweets-for-check-worthiness) for training purpose. We consider the English tweets from the Infodemic dataset and scrap Bengali and Hindi tweets from the twitter which are related to Covid-19. Fresh annotations were done and incorporated to create a larger Indic dataset for this task. For this purpose, scraping and parsing  tools were created which might be helpful to further mine Indic data.

2. METHOD:
We experimented with two different models to handle the tweet classification. In one settings, we consider a mono-lingual model, for handling English tweets. We extend the concept, by replacing the classifier with the multi-lingual one, where we consider tweets from English, Hindi and Bengali languages, as of now. Due to less complexity, the monolingual classifier achieves better result and also beats the State of the Art (SOTA) result on Infodemic dataset. On the other hand, the multi-lingual classifier gives comparable performance, while supporting multiple languages. We discuss both of these classifiers in details in the following sub-sections.

2.1 MONO-LINGUAL CLASSIFIER:
    The overall architecture of the monolingual classifier is shown below.
    ![alt text](https://https://github.com/DebanjanaKar/Covid19_FakeNews_Detection/blob/master/mono_ar.png?raw=true)
    We use various textual and user related features for the classification task. The tweet is preprocessed where the mentions, hashtags, html links are removed for further processing. The features used for monolingual English tweet classifier are as follows: 
          - sentence encoding of the tweet using Sentence BERT (sBERT), finetuned on NLI dataset.
          - 10 tweet features
          - 9 User features
          ![alt text](https://https://github.com/DebanjanaKar/Covid19_FakeNews_Detection/blob/master/text_feature.png?raw=true)
          ![alt text](https://https://github.com/DebanjanaKar/Covid19_FakeNews_Detection/blob/master/user_feature.png?raw=true)
          ![alt text](https://https://github.com/DebanjanaKar/Covid19_FakeNews_Detection/blob/master/text_feature_corr.png?raw=true)
          ![alt text](https://https://github.com/DebanjanaKar/Covid19_FakeNews_Detection/blob/master/user_feature_corr.png?raw=true)
    It is evident from the correlation plots that subset of user features and tweet features are helpful. We leave it for the classifier to weigh the relevant features accordingly, for the final classification task. For this purpose, we have used random forest classifier. We also experimented with different classifiers, details of which is shown below.
       ![alt text](https://https://github.com/DebanjanaKar/Covid19_FakeNews_Detection/blob/master/mono_result.png?raw=true)
 
 2.2
    


