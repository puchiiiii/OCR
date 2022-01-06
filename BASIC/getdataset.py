import os
from PIL import Image
import numpy as np
import torch

def process_image_channels(image):
	#处理通道数

	if image.mode == 'RGBA':
		r, g, b, a = image.split()
		image = Image.merge("RGB", (r, g, b))
	elif image.mode != 'RGB':
		image = image.convert("RGB")
	return image


def Imagetoarray(fname):
	#图片转成array

	myimage = Image.open(fname)
	myimage = myimage.resize((64, 64))
	myimage = process_image_channels(myimage) #处理通道数
	myimage_array = np.array(myimage, dtype = np.float32)
	#myimage_tensor = torch.tensor(myimage_array).permute(2, 0, 1)  #torch.Size([3, 64, 64])
	#print(myimage_array.shape)
	return myimage_array


def get_list(dirname):
	#通过过滤得到正确图片的路径名

	files = os.listdir(dirname) #利用os.listdir(目录名)  返回目录里所有的文件
	files = filter(lambda x: x.endswith('.jpg'), files) #过滤出以.jpg结尾的图片
	files = ['%s/%s'%(dirname,name) for name in files] #把上面的文件名和目录拼接在一起，成全路径
	
	return files


def get_trainset(dirname):
	#得到训练集的x和y

	trainpic = get_list(dirname)
	train_x = [Imagetoarray(imgname) for imgname in trainpic] #每个x为每张图片的array
	train_y = [int(imgname.split('-')[1][0]) for imgname in trainpic] #每个y为每张图片的标记

	return train_x, train_y


if __name__ == "__main__":
	train_x, train_y = get_trainset('../data')
