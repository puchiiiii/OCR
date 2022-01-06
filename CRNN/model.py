# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class BiLSTM(nn.Module):
	
	def __init__(self, idim, odim, hdim):
		super(BiLSTM, self).__init__()

		self.softmax = nn.Softmax()
		self.lstmlayer1 = nn.LSTM(
			input_size=idim, 
			hidden_size=hdim, 
			bidirectional=True,
			batch_first=True
		)
		self.linear1 = nn.Linear(hdim*2, odim)

	def forward(self, x): #[5, 1, 256]

		o1, (h,c) = self.lstmlayer1(x)
		o = self.linear1(o1).softmax(dim=0) 

		return o


class CRNN_Net(nn.Module):
	def __init__(self, idim, odim, hdim=256):
		super(CRNN_Net, self).__init__()
		self.conv1 = nn.Sequential( 
			nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			nn.MaxPool2d(2, 2)
		)
		self.conv2 = nn.Sequential( 
			nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.MaxPool2d(2, 2)
		)
		self.conv3 = nn.Sequential( 
			nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			nn.MaxPool2d(2, 2)
		)
		self.conv4 = nn.Sequential( 
			nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm2d(256),
			nn.ReLU(True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
			nn.ReLU(True),
			nn.MaxPool2d(4, 4)
		)

		self.bilstm = nn.Sequential(
			BiLSTM(256, hdim, hdim),
			BiLSTM(hdim, odim, hdim)
		)

		self.linear = torch.nn.Linear(hdim, odim)
		self.softmax = nn.Softmax()


	def forward(self, x): #[10, 3, 64, 320] [5,3,32,32]
		#print(x.shape)
		o1 = self.conv1(x) #[10, 64, 32, 160] [5,32,16,16]
		#print(o1.shape)
		o2 = self.conv2(o1) #[5,64,8,8]
		#print(o2.shape)
		o3 = self.conv3(o2) #[10, 128, 16, 80] [5,128,4,4]
		#print(o3.shape)
		o4 = self.conv4(o3) #[10, 256, 4, 20] [5,256,1,1]
		#print(o4.shape)
		o5 = o4.squeeze(2) #[10, 512, 19] [5,256,1]
		#print(o5.shape)

		o_crnn = o5.permute(0, 2, 1) #[10, 19, 512] [5,1,256]
		#print(o_crnn.shape)
		o_crnn_lstm = self.bilstm(o_crnn) #[10, 19, 11] [5,1,10]
		#print(o_crnn_lstm.shape)
		#o = o_crnn_lstm.permute(1, 0, 2) #[19, 10, 11] [1,5,10]
		o = o_crnn_lstm.mean(dim=1)
		#print(o.shape) #[5,10]
		return o

		#(5,32,4096) -> (5,32,10)
		#o6 = self.linear(o5).softmax(dim=0) 

if __name__ == "__main__":

	idim = 32
	odim = 10
	net = CRNN_Net(idim, odim).to(device)

	x = torch.randn(10, idim, 32)
	y = net(x)
	print(x.shape, y.shape)