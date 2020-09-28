from transformers import BertTokenizer, BertModel
import torch
import pandas as pd 
from pandas import DataFrame
import gc

import os
from collections import defaultdict

import re
import numpy as np 

def obtain_articles(path):
	"""
		Function: Iterates over all the files in the folder and creates a dict in the form of {doc_id: "article"}
		Input: path to location of files
		Returns: defaultdict in form of {doc_id: "article"}
	"""
	articles = defaultdict()
	print("Reading data")
	for filename in os.listdir(path):
		f = open(path + filename, "r")
		article = f.read()
		articles[int(filename)] = article
	return articles

def create_one_hot_labels(path):
	"""
		Function: Creates a one-hot encoded representation for the labels 
		Input: Path to location of labels
		Output: defaultdict in form of {doc_id: [one_hot_encoded label]}
	"""
	print("Pre-processing labels")

	f = open(path, "r")
	labels_dict = defaultdict()
	for line in f:
		tokenized_line = line.split(" ")
		document_id = tokenized_line[0].split("/")[1] #remove train/trade
		labels = tokenized_line[1:]
		labels[-1] = labels[-1].rstrip() #removes \n at the end
		labels_dict[document_id] = labels


	all_labels = set([x for (_, value) in labels_dict.items() for x in value])
	print(len(all_labels)) # this should be 90

	word_to_id = {token: idx for idx, token in enumerate(set(all_labels))}

	print("Converting labels")
	all_labels_one_hot = defaultdict()
	for key, value in labels_dict.items():
		one_hot_vector = np.zeros((90))
		for label in value:
			one_hot_vector[word_to_id[label]] = 1
		all_labels_one_hot[int(key)] = one_hot_vector

	return all_labels_one_hot

def data_to_dataframe(articles, name, model, tokenizer, all_labels_one_hot):
	"""
		Function: Converts the data to .csv files in batches of 50 docs
		Input: dict of articles and name (train/test)
		Output: None, saves data to .csv file
	"""
	data = []
	counter = 0

	for id_, sentence in articles.items():
		counter += 1
		print(f"{counter}/{len(articles.items())}")
		input_ids = torch.tensor(tokenizer.encode(sentence))[:512].unsqueeze(0)
		#model.eval()
		outputs = model(input_ids.long())
		last_hidden_states = outputs[0] #take the output of the last hidden layer
		# append zeroes for 512 - last_hidden_states.shape[0]
		final_input = last_hidden_states
		if 512-last_hidden_states.shape[1] != 0:
			pad = torch.zeros(1, 512-last_hidden_states.shape[1], 768)
			final_input = torch.cat((last_hidden_states, pad), dim =1)
		label = all_labels_one_hot[id_]
		#pickle.dump((id_, final_input, label), f)
		#if counter % 200 == 0:

		#print((id_, final_input, label))
		final_input = final_input.detach().numpy()
		data.append((id_, final_input, label))

		if counter % 50 == 0:
			df = DataFrame(data, columns =["id", "input_tensor", "label"])
			df.to_pickle(f"../data/{name}_data_pre_processed/{counter}_reuters_data_{name}.pkl")
			data = []
			gc.collect()

def pre_process_data(training_data_path, test_data_path,labels_path):
	"""
		Function: This function works as a main function, it calls the other functions and combines the output to a .csv file per batch of 50
		Input: corresponding paths to the location of the files
		Output: None, it creates .csv files
	"""
	all_labels = create_one_hot_labels(labels_path)
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	model = BertModel.from_pretrained('bert-base-uncased')

	articles = obtain_articles(training_data_path)
	data_to_dataframe(articles, "train", model, tokenizer, all_labels)

	articles = obtain_articles(test_data_path)
	data_to_dataframe(articles, "test", model, tokenizer, all_labels)

training_data_path = "../data/reuters/training/"
test_data_path = "../data/reuters/test/"
labels_path = "../data/reuters/cats.txt"

pre_process_data(training_data_path, test_data_path,labels_path)