import os
from PIL import Image
import numpy as np
import torch
from os import listdir
from PIL import Image
import random
import cv2

globalname = 0
def combinepics(files):
	global globalname
	im_list = [Image.open(f) for f in files[:5]]
	im_y = [int(f.split('-')[1][0]) for f in files[:5]]
	#im_y = "".join(im_y)
	#print(im_y)
	
	ims = []
	for i in im_list:
		new_img = i.resize((64, 64))
		new_img = process_image_channels(new_img)
		ims.append(new_img)
	width, height = ims[0].size

	result = Image.new(ims[4].mode, (width * len(ims), height))

	for i, im in enumerate(ims):
		result.paste(im, box=(i * width, 0))
	
	globalname += 1
	#result.save('combine'+str(globalname)+'.jpg')

	return result, im_y

def process_image_channels(image):
	'''
	if image.mode == 'RGBA':
		r, g, b, a = image.split()
		image = Image.merge("RGB", (r, g, b))
	elif image.mode != 'RGB':
		image = image.convert("RGB")
	'''
	image = image.convert("L")
	return image


def Imagetoarray(myimage):

	myimage_array = np.array(myimage, dtype = np.float32)
	return myimage_array


def get_list(dirname):

	files = listdir(dirname) 
	files = filter(lambda x: x.endswith('.jpg'), files) 
	files = ['%s/%s' % (dirname,name) for name in files] 

	return files


def get_trainset(dirname):
	#得到训练集的x和y

	train_x = []
	train_y = []
	trainpic = get_list(dirname)
	for idx in range(200): #训练集大小
		random.shuffle(trainpic)  
		myimage, myimage_y = combinepics(trainpic) #结果为五张合并成一张
		train_x.append(Imagetoarray(myimage))
		train_y.append(myimage_y) #每张合成图片的标记，str

	return train_x, train_y


if __name__ == "__main__":
	train_x, train_y = get_trainset('../../data')
	print(train_x[0].shape) #(5, 64, 320)
	print(train_y)
	print(len(train_x)) #5
	print(len(train_y)) #5
