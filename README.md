# Covid19_FakeNews_Detection



This package aims to detect fake tweets regarding Covid-19 from the real ones in real-time. We use an AI based technique to process the tweet text and use it, along with user features, to classify the tweets. We are handling tweets in three different languages: English, Hindi and Bengali. The development phases of our project is shown below and the details of the steps are described in the following sub-sections.<br/>
<p align="center">
  <img width="200" alt="flowchart" src="https://user-images.githubusercontent.com/19144385/87823230-8a569f80-c890-11ea-9eef-4d1ea405ac17.png">
</p>

## DATASET USED:
We use the Infodemic dataset (https://github.com/firojalam/COVID-19-tweets-for-check-worthiness) for training purpose. We consider the English tweets from the Infodemic dataset and scrap Bengali and Hindi tweets from the twitter which are related to Covid-19. Fresh annotations were done and incorporated to create a larger Indic dataset for this task. For this purpose, scraping and parsing  tools were created which might be helpful to further mine Indic data.

## METHOD:
We experimented with two different models to handle the tweet classification. In one settings, we consider a mono-lingual model, for handling English tweets. We extend the concept, by replacing the classifier with the multi-lingual one, where we consider tweets from English, Hindi and Bengali languages, as of now. Due to less complexity, the monolingual classifier achieves better result and also beats the State of the Art (SOTA) result on Infodemic dataset. On the other hand, the multi-lingual classifier gives comparable performance, while supporting multiple languages. We discuss both of these classifiers in details in the following sub-sections.

#### MONO-LINGUAL CLASSIFIER:
The architecture of the monolingual classifier is shown below.  
<p align="center">
  <img width="400" alt="mono_ar" src="https://user-images.githubusercontent.com/19144385/87823196-7dd24700-c890-11ea-81eb-a44042e68002.png">
</p>
We use various textual and user related features for the classification task. The tweet is preprocessed where the mentions, hashtags, html links are removed for further processing. The features used for monolingual English tweet classifier are as follows: <br />
        <ul>
        <li>sentence encoding of the tweet using Sentence BERT (sBERT), finetuned on NLI dataset.</li>
        <li>10 tweet features</li>
        <li>9 User features</li>
        </ul>
        <p align="center">
          <img width="682" alt="mono_features" src="https://user-images.githubusercontent.com/19144385/87824733-706a8c00-c893-11ea-9b9e-d936fd581675.png">
        </p>
    It is evident from the correlation plots that subset of user features and tweet features are helpful. We leave it for the classifier to weigh the relevant features accordingly, for the final classification task. For this purpose, we have used random forest classifier. We also experimented with different classifiers, details of which is shown below.
    <p align="center">
       <img width="500" alt="mono_result" src="https://user-images.githubusercontent.com/19144385/87823203-7f9c0a80-c890-11ea-86dd-486f59324418.png">
    </p>
 
#### MULTI-LINGUAL CLASSIFIER:
The architecture of the multi-lingual classifier is shown below:
<p align="center">
  <img width="600" alt="ml_ar" src="https://user-images.githubusercontent.com/19144385/87823189-7b6fed00-c890-11ea-9d72-ba52e130739c.png">
</p>
The usual tweet preprocessing is done (as mentioned above) and we use four different types of features for this task, which are as follows:
  <ul>
    <li> sentence encoding of the tweet using pretrained multilingual BERT (mBERT) model </li>
    <li> link score - Ratio of similarity calculated between a given tweet and titles of mined URLs obtained on querying the tweet on Google Search Engine (algorithm given below) <p align="center"> <img width="400" alt="link_score" src="https://user-images.githubusercontent.com/19144385/87823179-77dc6600-c890-11ea-8295-e847f5b48d07.png"> </p> </li>
    <li> bias score - The probability of a tweet containing offensive language. </li>
    <li> class weights - Weightage given to each of the labels by BERT finetuned on the train set</li>
   </ul>
<!--- The correlation of the last three features, i.e. link score, bias score and class weights, with the class label is shown below. It is clear from the correlation map that these features play an important role in determining the correct class (fake or real) of the tweet.
<img width="400" alt="ml_feature_corr" src="https://user-images.githubusercontent.com/19144385/87823194-7ca11a00-c890-11ea-9083-673fb51f8d23.png"><br/> --->
We have tested the performance of the classifier in different settings, details of which is shown in the below table.
<p align="center">
<img width="500" alt="multi_result" src="https://user-images.githubusercontent.com/19144385/87823208-81fe6480-c890-11ea-9c9b-c81bb24080f4.png">
</p>


## Graphical User Interface (GUI):
We design a simple static HTML page to obtain the tweet id/URL, as user input, and detect if the tweet is real or fake. Though our monolingual English classifier gave the best performance, even by beating the SOTA, we choose the multi-lingual classifier for its wider application. Some of the snapshots of our demo is shown below:
<p align="center">
<img width="400" alt="gui_hindi" src="https://user-images.githubusercontent.com/19144385/87827270-30f26e80-c898-11ea-8e03-402b12c21403.png"><br/>
<img width="400" alt="gui_bengali" src="https://user-images.githubusercontent.com/19144385/87827159-e8d34c00-c897-11ea-9352-b3db68af97cb.png"><br/>
<img width="400" alt="gui_english" src="https://user-images.githubusercontent.com/19144385/87827205-086a7480-c898-11ea-8033-89a045036481.png"><br/>
</p>

## FLASK API:
process.py is a working code to host the GUI in the localhost. It can be easily modified to host the demo in any server as well.


