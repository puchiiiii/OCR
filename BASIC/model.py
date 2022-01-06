# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

class Net(nn.Module):
	
	def __init__(self, idim, odim, hdim=4096):
		super(Net, self).__init__()
		#输入3通道，代表RGB，输出8个方向，padding=(kernel_size-1)/2, 将会有8个5x5x3堆叠
		self.conv1 = nn.Sequential( 
			nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(3, stride=1, padding=1),
			nn.BatchNorm2d(8)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(3, stride=1, padding=1),
			nn.BatchNorm2d(8)
		)
		self.softmax = nn.Softmax()
		self.linear1 = torch.nn.Linear(idim, hdim)
		self.tanh = torch.nn.Tanh()
		self.linear2 = torch.nn.Linear(hdim, odim)
		

	def forward(self, x): #(5,3,64,64)
		o1 = self.conv1(x) #(5,8,64,64)
		o2 = self.conv2(o1) #(5,8,64,64)
		#o2 = torch.squeeze(o2, dim=1)
		#o3 = o2.permute(0, 2, 1)
		o3 = o2.view(o2.size(0), o2.size(1), -1) #展平 (5,8,4096)
		#o4 = self.linear1(o3).tanh()
		o5 = self.linear2(o3).softmax(dim=0) #(5,8,10)
		o = o5.permute(0, 2, 1)
		o = o.mean(dim=2) #(5,10)

		return o


if __name__ == "__main__":

	idim = 64
	odim = 10 
	net = Net(idim, odim) 
	print(net)

	x = torch.randn(10, idim, 64)
	y = net(x)
	print(x.shape, y.shape)