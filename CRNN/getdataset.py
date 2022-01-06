import os
from PIL import Image
import numpy as np
import torch
import pickle

device = 'cuda' 

def load_data(filename):
	with open(filename, 'rb') as f:
		trainset_bacth = pickle.load(f, encoding="latin1")
	return trainset_bacth


def read_data(filename):
	setname = "data_batch_"
	trainset_data = []
	trainset_label = []
	for i in range(1, 6):
		trainset_batch = load_data(filename+setname+str(i))

		data_data = trainset_batch['data']
		data_label = trainset_batch["labels"]

		trainset_data.extend(data_data)
		trainset_label.extend(data_label)

	for i in range(len(trainset_data)):
		trainset_data[i].resize(3, 32, 32)

	return trainset_data, trainset_label


def Imagetoarray(myimage):
	#图片转成array
	myimage_array = np.array(myimage, dtype = np.float32)
	#print(myimage_array)
	return myimage_array


def get_trainset(dirname):
	#得到训练集的x和y
	trainset = []
	train_x, train_y = read_data(dirname)

	for i in range(len(train_x)):
		trainset.append((torch.tensor(Imagetoarray(train_x[i])).to(device), train_y[i]))
	
	return trainset


if __name__ == "__main__":
	trainset = get_trainset('./data/')
	print(len(trainset))
