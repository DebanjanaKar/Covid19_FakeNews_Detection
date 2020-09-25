## Multilingual Transformer Based Classifiers

This section provides a detailed description on how to use the best performing multilingual models. 

#### Pre-requisites :
1. `Linux` system (WSL works for Windows Systems)
2. `Anaconda 4.8.2, Python 3.x`

#### Set up the conda environment by :
-  Create an environment in your local server using the given `environment.yml` file with the following command :  
`conda env create -f environment.yml`  
The first line of the yml file sets the new environment's name.
-  Activate the environment using :
`conda activate <env_name>`

While we provide the steps below on how to train and test from scratch, one can access the pretrained model we provide and use it directly for prediction.

#### For using the pretrained model : 

1. Download the pretrained model from : https://ibm.box.com/v/pretrained-indic-model and put it in a folder 'models'
2. Set the path to the pretrained model in the `predict(path, sent)` call in the script `prediction.ipynb`

#### For training from scratch :
For each of the files below, set the correct path of your local server

1. Use `preprocess.ipynb` to preprocess the dataset
2. For feature extraction :
- Run `extract_toxic.ipynb` to calculate the bias  score.
- Run `fact_check.py` to get link scores.
- Run `feature_embedder.ipynb` to embed all the extracted features to be pushed into the final classifier.
- Run `feature_embedder_user.ipynb` to embed user features to be pushed into the final classifier.
3. For training transformer based classifier :
- Run `bert_binary_classifier.py` for running monolingual, crosslingual and multilingual text-based experiments. Examples of all the language configurations are mentioned in the script. You would want to comment out all the configurations except the one of your choice and then train and evaluate.
- Run `bert_binary_userft.py` for running monolingual, crosslingual and multilingual text+features-based experiments. The same language configurations can be adapted to this script as well.
4. For inference, run `prediction.ipynb` loading your trained model.



