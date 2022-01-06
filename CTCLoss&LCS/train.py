# -*- coding:utf-8 -*-
import torch
from model import Net
import numpy as np
from getdataset import get_trainset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import itertools

device = 'cuda' 
#device = 'cpu' 
torch.backends.cudnn.enabled = False

def LCS(s1, s2):
	size1 = len(s1) + 1
	size2 = len(s2) + 1
	chess = [[["", 0] for j in list(range(size2))] for i in list(range(size1))]
	for i in list(range(1, size1)):
		chess[i][0][0] = s1[i - 1]
	for j in list(range(1, size2)):
		chess[0][j][0] = s2[j - 1]

	for i in list(range(1, size1)):
		for j in list(range(1, size2)):
			if s1[i - 1] == s2[j - 1]:
				chess[i][j] = ['↖', chess[i - 1][j - 1][1] + 1]
			elif chess[i][j - 1][1] > chess[i - 1][j][1]:
				chess[i][j] = ['←', chess[i][j - 1][1]]
			else:
				chess[i][j] = ['↑', chess[i - 1][j][1]]
	i = size1 - 1
	j = size2 - 1
	s3 = []
	while i > 0 and j > 0:
		if chess[i][j][0] == '↖':
			s3.append(chess[i][0][0])
			i -= 1
			j -= 1
		if chess[i][j][0] == '←':
			j -= 1
		if chess[i][j][0] == '↑':
			i -= 1
	s3.reverse()
	#print('最大公共子序列：',s3)
	return len(s3)

def train_network(net, train_x, train_y):

	net.train() #把网络设置成 .train()的训练模式
	predict = net(train_x).log_softmax(2) #输出11类别，[16, 5, 11]，log_probs一般需要经过torch.nn.functional.log_softmax处理后再送入到CTCLoss中；
	#将该batch_size 内每一张图片的字符的index拼成一个【一维数组】. 会按照target_lengths 中的值自动对该一维数组中的index进行划分到对应图片
	'''
	train_y1 = []
	for y in train_y:
		for i in range(len(y)):
			train_y1.append(y[i])
	'''
	train_y = train_y.view(-1) # 将该batch_size 内每一张图片的字符的index拼成一个【一维数组】
	input_lengths = [predict.shape[0] for i in range(5)]
	input_lengths = torch.tensor(input_lengths).int().to(device)
	target_lengths = [5, 5, 5, 5, 5] #shape 为(N=5)，每一张图片中包含的字符个数=5，target_lengths = (5,5,5,5,5)
	target_lengths = torch.tensor(target_lengths).int().to(device)
	net.optimizer.zero_grad() #把优化器想要优化的参数，都设置成0
	net.loss(predict, train_y, input_lengths, target_lengths).backward() #求预测结果和标签的损失，接着后项推理，即把所有可求梯度的部分求梯度，然后累加。
	'''
	log_probs: 网络输出的tensor, shape为 (T, N, C）, T 为时间步, N 为batch_size, C 为字符总数（包括blank）。网络输出需要进行log_softmax。
	targets: 目标tensor, targets有两种输入形式。其一: shape为 （N，S），N为batch_size，S 为识别序列的最长长度，值为每一个字符的index，不能包含blank的index。由于可能每个序列的长度不一样，而数组必须维度一样，就需要将短的序列padded 为最长序列长度。 
												其二: 将该batch_size 内每一张图片的字符的index拼成一个【一维数组】. 会按照target_lengths 中的值自动对该一维数组中的index进行划分到对应图片
	target_lengths: shape 为(N) 的Tensor, 每一个位置记录了对应图片所含有的字符数. 假如 N=4，即共有4张图片，每一张图片中包含的字符个数分别为: 8, 10, 12, 20, 那么 target_lengths = (8, 10, 12, 20), 同时targets 中共有 （8 + 10 + 12 + 20）个值，按照target_lengths中的值依次在targets 中取值即可
	input_lengths: shape 为 (N) 的Tensor, 值为输出序列长度T, 因为图片宽度都固定了，所以都为T
	'''
	net.optimizer.step() #由学习率决定，沿着逆梯度

def test_network(net, train_x, train_y):
	'''
	最长公共子序列
	'''
	#print("@@@@@@@@@@@",train_y)
	net.eval() #设置成测试模式
	with torch.no_grad():
		predict_label = net(train_x).permute(1,0,2)
		predict_label = predict_label.argmax(2).tolist()
		for i in range(len(train_y)):
			while 10 in predict_label[i]:
				predict_label[i].remove(10)
		predict_label = [[key for key, mydata in itertools.groupby(t)] for t in predict_label]
		#print('每个位置的可能数字：\n')
		#for i in range(len(predict_label)):
		#	print("原来：",''.join(map(str,train_y.tolist()[i])))
		#	print("预测：",''.join(map(str,predict_label[i]))) #[训练集大小, 时间步(16)]，-> [训练集大小, 5]
		rightnum = 0.0
		for i in range(len(train_y)): #训练集大小
			rightlen = LCS(''.join(map(str,train_y.tolist()[i])), ''.join(map(str,predict_label[i])))
			rightnum += rightlen
			#if len(predict_label[i]) > 5:
			#	for j in range(len(train_y[i])):
			#		if train_y[i][j] == predict_label[i]:
			#			rightnum += 1
		acc = rightnum/float(5*len(train_y))
		return acc

if __name__ == "__main__":

	net = Net(64, 11).to(device) #创建网络
	#blank：空白标签所在的label值，0~9+空，blank = 11-1
	net.loss = torch.nn.CTCLoss(blank=10, reduction='mean').to(device) #ctcloss
	#base_lr = 0.0004 #学习率初始，用于学习率动态变化
	net.optimizer = torch.optim.Adam(net.parameters(), lr=0.0009)  #定义亚当优化器

	#获取训练集
	train_x, train_y = get_trainset('../../data')
	train_x = torch.tensor(train_x).float().to(device) #[5, 64, 320,]
	train_x = train_x.squeeze(dim=1)
	train_y = torch.tensor(train_y).long().to(device) #
	print(train_x.shape, train_y.shape)

	trainset = TensorDataset(train_x, train_y)
	trainset = DataLoader(dataset=trainset, batch_size=5, shuffle=True)
	
	#step = [10,20,30,40]
	
	for i in range(2000):
		'''
		lr = base_lr * (0.1 ** np.sum(i >= np.array(step))) #学习率动态变化

		for params_group in net.optimizer.param_groups: #学习率动态变化
			params_group['lr'] = lr
		'''
		for step, traindata in enumerate(trainset):
			train_xx, train_yy = traindata
			train_network(net, train_xx, train_yy)
		train_acc = test_network(net, train_x, train_y)
		print('Epoch: %03d   train=%.3f'%(i+1, train_acc))

	torch.save(net, 'model.pkl') #保存网络的模型
	mdl = torch.load('model.pkl')