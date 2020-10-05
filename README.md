# Model Uncertainty in Deep Question Classification

This repository contains all the code to run the experiments from 'Model Uncertainty in Deep Question Classification'. 
This report is a result of the DL4NLP course at the UvA. 

## Files in this repo

* code

  * CNNText.py - Contains the CNN architecture that we trained on our data and used to perform experiments with.
  * dataset.py - Contains the PyTorch dataset class that is used to process our data to a format usable by the CNN.
  * load_data.py - Pre-processes the raw data to pickle format containing BERT-embeddings. Should be run for training, test and possibly experimental data.
  
* models

  * Contains all the weights for our trained CNN models. We trained on different seeds, training set sizes and saved at multiple epochs.

* data

  * train_{n}.label - File with n labeled training questions
  * TREC_10.label - File with 500 labeled test questions
  * masked_questions.label - Should be manually created with all words of a question removed one-by-one. Used for qualitative experiments in Bayesian Model Averaging notebook.

