# -*- coding:utf-8 -*-
import torch
from model import CRNN_Net
import numpy as np
from getdataset import get_trainset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

device = 'cuda' 
torch.backends.cudnn.enabled = False

def train_network(net, train_x, train_y):

	net.train()
	predict = net(train_x)#.log_softmax(2)
	#print("@@",predict.shape)
	#print("@@",train_y)
	net.optimizer.zero_grad() 
	net.loss(predict, train_y).backward() 
	net.optimizer.step() 

def test_network(net, train_x, train_y):

	net.eval()
	with torch.no_grad():
		predict_label = net(train_x).argmax(1) 
		acc = torch.sum(train_y == predict_label).float()/len(train_y) 
		return acc

if __name__ == "__main__":
	batch_size = 5
	lr = 0.001

	net = CRNN_Net(64, 10).to(device)
	#交叉熵损失
	net.loss = torch.nn.CrossEntropyLoss().to(device)
	#亚当优化器
	net.optimizer = torch.optim.Adam(net.parameters(), lr = lr) 

	#获取Dataset
	train_dataset = get_trainset('./data/')
	#放入DataLoader
	trainset = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

	best_acc_score = 0.0
	for i in range(100):
		acc_score = 0.0
		n = 0

		for data_x, label in trainset:
			data_x = data_x.to(device)
			label = label.to(device)

			train_network(net, data_x, label)
			
			acc_score += test_network(net, data_x, label)
			n += batch_size

		acc_score = (acc_score/n)
		best_acc_score = acc_score if acc_score > best_acc_score else best_acc_score
		print('Epoch: %03d  accuracy=%.3f'%(i+1, acc_score))

	print('best_acc_score: ', best_acc_score)

	#torch.save(net, 'model.pkl') #保存网络的模型
	#mdl = torch.load('model.pkl')