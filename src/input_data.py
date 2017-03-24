# coding: utf-8
# input_data.py
# to include some data related file.

from __future__ import division
import sys
sys.path.append('../data/hcl/')
sys.path.append('../data/address/')
import hcl
import at
import small_char_set
from PIL import Image
import numpy as np

gb2312_path = '../data/hcl/GB2312_3755.txt'
keyword_list = ['省','市','县','区','乡','镇','村','巷','弄','路', '街', '社', '组', '队', '州', 'X']
full_list = [i.rstrip('\n') for i in open(gb2312_path, 'r').readlines()]
small_list = list(set(small_char_set.samll_char_set).intersection(set(full_list)))
full_list = small_list

arguments = {
	'kwd': [
		hcl.input_data(keyword_list, raw_data = False, direct_info = False), 
		(1, 32, 32),
		16,
		'../data/result/',
		'kwd_weights.hdf5',
		'kwd_history.txt',
		100,
		10,
		1000000,
		False
		],
	'full': [
		hcl.input_data(full_list, raw_data = False, direct_info = False),
		(1, 32, 32),
		len(full_list),
		'../data/result/',
		'full_weights.hdf5',
		'full_history.txt',
		1200,
		10,
		12000000,
		False
		]
}


def proba_to_char(probas, word_list = keyword_list):
	kps = []
	probas = list(probas)
	for i in probas:
		i = list(i)
		proba = sorted(i, reverse=True)
		kp = [(word_list[i.index(j)],j) for j in proba]
		kps.append(kp)
	return kps

def index_to_char(predict_classes_result, word_list = keyword_list):
	return [word_list[i] for i in predict_classes_result]

def generate_data(dataset, batch_size, type_of_want='train'):
	while True:
		x,y = dataset.next_batch(batch_size, type_of_want)
		yield (np.array(x), np.array(y))

def preprocess(img_file_path, wanna_size = (32, 32)):
	img = Image.open(img_file_path)
	charlist = img_to_matrix(img)
	charlist = resize_all(charlist, wanna_size)
	charlist = np.array(charlist)
	return charlist

def img_to_matrix(img):
	if img.mode != '1':
		img = img.convert('1')
	d = list(map(lambda x: 0 if x == 255 else 1, img.getdata()))
	size = img.size
	m = [d[i*size[0]:i*size[0]+size[0]] for i in range(size[1])]
	m = [[[m[i][j*size[1]:j*size[1]+size[1]] for i in range(size[1])]] for j in range(int(size[0]/size[1]))]
	return m

def resize_all(charlist, size):
	def resize(character, size):
		s_size = (len(character), len(character[0]))
		a,b = size[0]/s_size[0], size[1]/s_size[1]
		if a==b==1:
			return character
		character = [[character[int(i/a)][int(j/b)] for j in range(size[1])]for i in range(size[0])]
		return character
	return [[resize(i[0], size)] for i in charlist]