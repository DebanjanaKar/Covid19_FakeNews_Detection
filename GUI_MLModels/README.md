## Tweet Class Detection (Fake or Real)

This section provides a detailed description on how to use the monolingual and multilingual classifier for analysing a tweet. 
The GUI integrates these classifiers and can be deployed using Python Flask API easily. 

#### Pre-requisites :
1. `Linux` system (WSL works for Windows Systems)
2. `Python 3.x`

#### Set up the virtual environment by :
-  Create an environment in your local server using the given `requirements.txt` file with the following command :  
`virtualenv -p python3 venv`  
Install the required packeges:
`pip3 install -r requirements.txt`
-  Activate the environment using :
`source venv/bin/activate`
One can use a conda environment as well with the required packages, as mentioned in the requirements.txt file.

#### For predicting the tweets : 
Set up a Twitter Developer account for accessing the tweets. Details can be found in https://developer.twitter.com/en/docs/basics/getting-started.
Provide the required details in private.py
1. for monolingual english classifier run the command: `python predict_tweet.py <tweet id/URL>`.
2. for multilingual classifier run the command: python `predict_multi_lingual.py <tweet id/URL>`.
The deployed multilingual classifier uses the Random Forest Classifier with the 10 tweet features and 9 user features along with the BERT embeddings (multilingual BERT base uncased).  

Some of the examples of tweet IDs are given below:
- 1275848474987311107 
- 1277229173568765957
- 1280700280640864261
- 1280740783449010177

#### For deploying Flask API :
run `process.py`
This will deploy the API in the local server. The code can be easily modified (by providing server address and desired port number) to deploy the API in any suitable server.
