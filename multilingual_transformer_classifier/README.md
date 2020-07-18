This section provides a detailed description on how to use the best performing multilingual models. 
These models have currently not been integrated with the GUI and is a work in progress.

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
2. Set the path to the pretrained model in the 'predict(path, sent)` call in the script `prediction.ipynb`

#### For training from scratch :
For each of the files below, set the correct path of your local server

1. Use `preprocess.ipynb` to preprocess the dataset
2. For finetuning transformer based classifier :
- Run `bert_multilingual_kfold_classifier.py`
- Run `prediction.ipynb` for predicting.
3. For training bilstm based classifier :
- Run `extract_toxic.ipynb` to calculate the bias  score.
- Run `fact_check.py` to get link scores.
- Run `feature_embedder.ipynb` to embed all features to be pushed into the final classifier. In this script, you can either use the pretrained weights of the training set already given in the resources folder as `multi_raw_outputs.pickle` or you can save the weights you hav obtained while finetuning the transformer based classifier.
- Run `bilstm_based_classifier.ipynb` to train and test the model.



