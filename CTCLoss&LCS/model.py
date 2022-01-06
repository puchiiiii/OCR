# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

class Net(nn.Module):
	'''
	空洞卷积/膨胀卷积/扩张卷积：空洞系数dilated大，增大感受野；dilated = 1 每一个点选一个（原来的卷积核）
	'''
	def __init__(self, idim, odim, hdim=128):
		super(Net, self).__init__()
		self.conv1 = nn.Sequential( 
			nn.Conv1d(in_channels=idim, out_channels=32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool1d(2, stride=2),
			nn.BatchNorm1d(32)
		)
		self.conv2 = nn.Sequential(
			nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.MaxPool1d(4, stride=4),
			nn.BatchNorm1d(16)
		)
		self.softmax = nn.Softmax()
		self.relu = nn.ReLU()
		self.lstmlayer1 = nn.LSTM(
			input_size=40, 
			hidden_size=hdim, 
			batch_first=True
		)
		self.lstmlayer2 = nn.LSTM(
			input_size=hdim, 
			hidden_size=192, 
			batch_first=True
		)
		self.linear1 = nn.Linear(192, odim)

	def forward(self, x): #[5, 64, 320]
		#输出（T,N,C），分类个数C = 11，T是句子长度，N是batch_size，一个batch中N张拼接图 = 5
		#print(x.shape)
		o1 = self.conv1(x) #[5, 32, 160]
		#print(o1.shape)
		o2 = self.conv2(o1) #[5, 16, 40]
		#print(o2.shape)
		#o3 = o2.softmax(dim=0) #[5, 16, 40]
		#print(o3.shape)
		o4 = o2.permute(1, 0, 2) #[16, 5, 40]
		#print(o4.shape)
		o5, (h,c) = self.lstmlayer1(o4) #[16, 5, 128]
		o6 = o5.relu()
		#print(o6.shape)
		o7, (h,c) = self.lstmlayer2(o6) #[16, 5, 11]
		#print(o7.shape)
		o = self.linear1(o7).softmax(dim=0)  #加个全连接层
		#o = o[-1] #5, 11
		#print(o.shape) #[16, 5, 11]

		return o


if __name__ == "__main__":

	idim = 64
	odim = 11 #0~9+空
	net = Net(idim, odim) 
	#print(net)

	x = torch.randn(5, idim, 320)
	y = net(x)
	print(x.shape, y.shape)
	#torch.Size([5, 64, 320]) torch.Size([16, 5, 11])