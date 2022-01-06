# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class SELayer(nn.Module):
	def __init__(self, channel, reduction=16):
		super(SELayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1) #Squeeze操作，64中取最大的1
		self.fc = nn.Sequential(				#Excitation操作，得到Squeeze的1 × 1 × C 1\times 1\times C1×1×C的feature map后，使用FC全连接层，对每个通道的重要性进行预测，得到不同channel的重要性大小。有两个全连接，一个降维，一个恢复维度。
			nn.Conv2d(channel, channel // reduction, 1, 1),
			nn.ReLU(),
			nn.Conv2d(channel // reduction, channel, 1, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.avg_pool(x).view(b, c) #5,8,64,64 -> 5,8,1,1 -> 5,8
		y = self.fc(y.view(b, c, 1, 1)) #5,8,1,1
		return x * y.expand_as(x)

class Net(nn.Module):
	
	def __init__(self, idim, odim, hdim=4096, reduction=16):
		super(Net, self).__init__()
		#输入3通道，代表RGB，输出8个方向，padding=(kernel_size-1)/2, 将会有8个5x5x3堆叠
		self.conv1 = nn.Sequential( 
			nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(3, stride=1, padding=1),
			nn.BatchNorm2d(8)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(3, stride=1, padding=1),
			nn.BatchNorm2d(8)
		)
			
		self.conv3 = nn.Sequential(
			nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(3, stride=1, padding=1),
			nn.BatchNorm2d(32)
		)
		
		self.se = SELayer(channel=32, reduction=reduction)
		self.relu = nn.ReLU(inplace=True)

		self.softmax = nn.Softmax()
		self.linear1 = torch.nn.Linear(idim, hdim)
		self.tanh = torch.nn.Tanh()
		self.linear2 = torch.nn.Linear(hdim, odim)
		

	def forward(self, x): #(5,3,64,64) #batch,channel,64x64
		residual = x

		o1 = self.conv1(x) #(5,8,64,64)
		o2 = self.conv2(o1) #(5,8,64,64)
		o3 = self.conv3(o2) #(5,32,64,64)
		o4 = self.se(o3) #(5,32,64,64)
		o5 = o4.view(o4.size(0), o4.size(1), -1) #展平 (5,32,4096)
		o6 = self.linear2(o5).softmax(dim=0) #(5,32,10)
		o = o6.permute(0, 2, 1)
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