# -*- coding:utf-8 -*-
import torch
from model import Net
import numpy as np
from getdataset import get_trainset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

device = 'cuda' 
#device = 'cpu' 
torch.backends.cudnn.enabled = False
'''
#https://www.jb51.net/article/189417.htm
#base_lr = 1e-4
def adjust_lr(epoch, optimizer, base_lr):
	lr = base_lr * (0.1 ** np.sum(epoch >= np.array(step)))
	for params_group in sgd_opt.param_groups:
		params_group['lr'] = lr
	return lr
'''
def train_network(net, train_x, train_y):

	net.train() #把网络设置成 .train()的训练模式
	predict = net(train_x) #输出10类别

	net.optimizer.zero_grad() #把优化器想要优化的参数，都设置成0
	net.loss(predict, train_y).backward() #求预测结果和标签的损失，接着后项推理，即把所有可求梯度的部分求梯度，然后累加。
	net.optimizer.step() #由学习率决定，沿着逆梯度

def test_network(net, train_x, train_y):

	net.eval() #设置成测试模式
	with torch.no_grad():
		predict_label = net(train_x).argmax(1) #argmax
		acc = torch.sum(train_y == predict_label).float()/len(train_y) 
		return acc #.to('cpu').numpy() #显卡里面调到cpu

if __name__ == "__main__":

	net = Net(64, 10).to(device) #创建网络
	net.loss = torch.nn.CrossEntropyLoss().to(device) #定义交叉熵损失函数，用来逆向求梯度，被优化器使用，然后调整参数，损失下降
	base_lr = 0.0004 #学习率初始，用于学习率动态变化
	#net.optimizer = torch.optim.Adam(net.parameters(), lr=0.0004)
	#mlr_scheduler = lr_scheduler.StepLR(net.optimizer, step_size=6, gamma=0.1)
	net.optimizer = torch.optim.Adam(net.parameters(), lr = base_lr) #定义亚当优化器

	#获取训练集
	train_x, train_y = get_trainset('../../data')
	train_x = torch.tensor(train_x).float().to(device)
	train_x = torch.squeeze(train_x, dim=1)
	train_x = train_x.permute(0, 3, 1, 2)
	train_y = torch.tensor(train_y).long().to(device)
	print(train_x.shape, train_y.shape)

	#trainset = list(zip(train_x, train_y))
	trainset = TensorDataset(train_x, train_y)
	trainset = DataLoader(dataset=trainset, batch_size=5, shuffle=True)
	
	step = [10,20,30,40]
	
	for i in range(250):
		lr = base_lr * (0.1 ** np.sum(i >= np.array(step))) #学习率动态变化

		for params_group in net.optimizer.param_groups: #学习率动态变化
			params_group['lr'] = lr

		for step, traindata in enumerate(trainset):
			train_xx, train_yy = traindata
			train_network(net, train_xx, train_yy)
		train_acc = test_network(net, train_x, train_y)
		print('Epoch: %03d  train_acc=%.3f'%(i+1, train_acc))

	torch.save(net, 'model.pkl') #保存网络的模型
	mdl = torch.load('model.pkl')