import numpy as np 
import pickle
import torch

# to do: make it functional for batches (?) for now this should/could suffice (per 1 datapoint)
class Dataset(torch.utils.data.Dataset):
	def __init__(self, file_path):
		with open(file_path, 'rb') as train_data_file:
			self.data = pickle.load(train_data_file)

	def __len__(self):
		return len(self.train_data)