# TathyaCov : Detecting Fake Tweets in the time of COVID 19
<p align="center">
<img width="1000" height="200" alt="title" src="https://user-images.githubusercontent.com/19144385/87828882-9300a300-c89b-11ea-8e18-6325dfe88dfb.png">
</p>

<p align="center">
  <b><ins> DEMO VIDEO: https://youtu.be/pdWoBxBu9-k </ins></b>
</p>

This repository contains the implementation of the paper : **"No Rumours Please! A Multi-Indic-Lingual Approach for Covid Fake-Tweet Detection"**. The system aims to classify whether a tweet contains a verifiable claim or not in real-time and has been specifically trained to detect COVID19 related fake news. We use AI based techniques to process the tweet text and use it, along with user features, to classify the tweets as either **REAL** or **FAKE**. We are handling tweets in three different languages: **English, Hindi and Bengali**. 
<p align="center">
  <img width="300" alt="flowchart" src="https://user-images.githubusercontent.com/19144385/87848329-cf182000-c8fc-11ea-9e1f-11d9466c5e6e.png">
</p>

## Dataset:
We create the **Indic-covidemic tweet dataset** and use it for training and testing purpose. We consider the English tweets from the Infodemic dataset (https://github.com/firojalam/COVID-19-tweets-for-check-worthiness) and scrape Bengali and Hindi tweets from Twitter which are related to COVID-19. Fresh annotations were done and incorporated to create the larger Indic dataset for this task. For this purpose, scraping and parsing  tools were created which might be helpful to further mine Indic data.

## Method:
We experimented with two different models to handle the tweet classification. In one setting, we consider a mono-lingual model, for handling English tweets. We extend the concept, by replacing the classifier with the multi-lingual one, where we consider tweets from English, Hindi and Bengali languages, as of now. The main essence of our proposed approach lies in the features we have used for the classification task, the different classifiers and  their  corresponding  adaptation  done  for  identifying  the fake tweets.

The architecture of the classifier is as shown below.  
<p align="center">
  <img width="300" alt="mono_ar" src="https://github.com/DebanjanaKar/Covid19_FakeNews_Detection/blob/master/images/architecture.png">
</p>
We have used various textual and user related features for the classification task as follows: <br />
        <ul>
        <li>bert based sentence encoding of the tweets (TxtEmbd)</li> 
        <li>tweet features (twttxt)</li>
        <li>user features (twtusr) <p align="center"> <img width="400" alt="link_score" src="https://user-images.githubusercontent.com/19144385/87823225-875baf00-c890-11ea-8c7b-6c57a4198eb6.png"> </p> </li>
        <li> link score (FVer) - Ratio of similarity calculated between a given tweet and titles of verified URL list obtained on querying the tweet on Google Search Engine (algorithm given below). We have a list of 50 URLs listed as verified sources. <p align="center"> <img width="300" alt="link_score" src="https://user-images.githubusercontent.com/19144385/87823179-77dc6600-c890-11ea-8295-e847f5b48d07.png"> </p> </li>
        <li> bias score (Bias) - The probability of a tweet containing offensive language. </li>
        </ul>
        <p align="center">
          <img width="450" alt="mono_features" src="https://github.com/DebanjanaKar/Covid19_FakeNews_Detection/blob/master/images/correlation.png">
        </p>
    It is evident from the correlation plot that a subset of user features and tweet features can be helpful. We have experimented with different classifiers, the results of which are as given below.
    <p align="center">
       <img width="350" alt="mono_result" src="https://github.com/DebanjanaKar/Covid19_FakeNews_Detection/blob/master/images/mono_results.png">
        <img width="350" alt="multi_result" src="https://github.com/DebanjanaKar/Covid19_FakeNews_Detection/blob/master/images/multi_results.png">
    </p>

## Usage :

Each of the folders are equipped with detailed READMEs on how to run the scripts.

- For dataset, refer to the `data` folder
- To scrape and annotate more dataset, refer to `scraping_tools` folder (We encourage extending the dataset to accomadate more annotations in languages explored and unexplored in this work)
- For the transformer based classifiers, refer to the `transformer_classifiers` folder
- For ML based models and GUI implementation, refer to the `GUI_MLModels` folder

## Graphical User Interface (GUI):
We design a simple static HTML page to obtain the tweet id/URL, as user input, and detect if the tweet is real or fake. Though our monolingual English classifier gave the best performance, even by beating the SOTA, we choose the multi-lingual classifier for its wider application. Some of the snapshots of our demo is shown below:
<p align="center">
<img width="400" alt="gui_hindi" src="https://user-images.githubusercontent.com/19144385/87827270-30f26e80-c898-11ea-8e03-402b12c21403.png"><br/>
<img width="400" alt="gui_bengali" src="https://user-images.githubusercontent.com/19144385/87827159-e8d34c00-c897-11ea-9352-b3db68af97cb.png"><br/>
<img width="400" alt="gui_english" src="https://user-images.githubusercontent.com/19144385/87827205-086a7480-c898-11ea-8033-89a045036481.png"><br/>
</p>

## FLASK API:
The GUI has been hosted in a IBM server (http://pca.sl.cloud9.ibm.com:1999/) which is accessible within IBM domain. <br/>
process.py is a working code to host the GUI in the localhost. It can be easily modified to host the demo in any other server as well.

## Citation :
If you find our work useful, please cite our work.

