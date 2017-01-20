# hcl dataset.
# coding: utf-8
# hcll.py
# this hcl dataset is HCL2000 and is realigned base on character categories.

from __future__ import division
import os
import random
import math
import numpy as np

CURRENT_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]+'/'
SUB_DIRECTORY     = 'hcl'
CHARACTER_DICT    = 'GB2312_3755.txt'

class input_data:
	train_num   = 700
	test_num    = 300
	default_shape= (32, 32)
	batch_max   = 50

	def __init__(self, chars = [], num_train = 700, num_test = 300, raw_data = True, one_hot = True, shape = (32, 32), direct_info = False):
		''' to get the hcl data. '''
		self.indexes   = input_data.char_to_index(chars)
		self.num_train = num_train
		self.num_test  = num_test
		self.raw_data  = raw_data
		self.one_hot   = one_hot
		self.shape      = shape
		self.direct_info = direct_info
		self.batch_index = 0
		self.cache     = {}
		
	def next_batch(self, batch_size, type_of_want = 'train'):
		''' to get next batch data. '''
		return input_data.read_data(self.indexes, self.num_train if (type_of_want == 'train') else self.num_test, self.raw_data, self.one_hot, self.shape, self.direct_info, type_of_want, batch_size, self.cache)
	
	@staticmethod
	def read_data(indexes, char_range, raw_data, one_hot, shape, direct_info, type_of_want, num_of_want, cache):
		''' read data from default directory with settings. '''
		file_path = os.path.join(CURRENT_DIRECTORY, SUB_DIRECTORY, type_of_want)
		original_index = indexes
		indexes = [indexes[random.randrange(len(indexes))] for i in range(num_of_want)]
		char_list = []
		label_list = []
		for i in indexes:
			key = type_of_want + str(i)
			if key in cache:
				r = cache[key]
			else:
				k = (i if i != -1 else random.randrange(3755))
				filename = os.path.join(file_path, str(k+1)+'.chars')
				r = np.fromfile(filename, dtype='uint8')
				r = r.reshape(input_data.train_num, input_data.default_shape[0], input_data.default_shape[1])
				cache[key] = r
			s = random.randrange(char_range)
			character = r[s]
			character = input_data.reshape(character, shape) if (shape != (32, 32)) else character
			character = input_data.direct_info(character) if direct_info else character
			character = character.reshape(character.size) if raw_data else character
			char_list.append(character)
			if one_hot:
				label = [0 for x in range(len(original_index))]
				label[original_index.index(i)] = 1
				label_list.append(label)
			else:
				label_list.append(i)
		#return (np.array(char_list).astype('float64'), np.array(label_list).astype('float64'))
		return (char_list, label_list)

	@staticmethod
	def char_to_index(chars):
		''' to get the index of the characters in list `chars`. '''
		if type(chars[0]) == int:
			return chars
		elif type(chars[0]) == str:
			f = open(CURRENT_DIRECTORY + CHARACTER_DICT, 'r')
			char_dict = f.readlines()
			f.close()
			indexes = [(char_dict.index(i+'\n') if (i+'\n' in char_dict) else -1) for i in chars]
			return indexes
		else:
			return [-1]

	@staticmethod
	def show(character):
		''' to show the character matrix. '''
		character = character.astype('uint8')
		if character.ndim == 2:
			for i in range(character.shape[0]):
				print ''.join([str(j) for j in list(character[i])])
		else:
			print 'not printalbe: dim %d' %character.ndim

	@staticmethod
	def reshape(character, shape):
		''' to reshape the character with `shape`'''
		s_shape = character.shape
		a,b = shape[0]/s_shape[0],shape[1]/s_shape[1]
		c = np.zeros(shape, dtype='uint8')
		for i in range(shape[0]):
			for j in range(shape[1]):
				c[i][j] = character[int(i/a)][int(j/b)]
		return c

	def get_index(self):
		if (self.batch_index + 1) >= self.batch_max:
			self.batch_index = 0
		else:
			self.batch_index += 1
		return self.batch_index

	def read_direct_info_dataset_of_3755_classes_1000_batches(self, ty = 'train'):
		path = os.path.join(CURRENT_DIRECTORY, SUB_DIRECTORY, ty, 'bin', str(self.get_index()))
		x = np.fromfile(path+'.binx', dtype=np.int64)
		x = x.reshape(1000,8,32,32)
		y = np.fromfile(path+'.biny', dtype=np.int64)
		y = y.reshape(1000,3755)
		return (x, y)

	@staticmethod
	def direct_info(character):
		''' to generate a 8xshape div tensor to get character's direction information with sobel operator, gradient.'''
		shape = character.shape
		f = lambda x, y: character[x][y]
		gx = lambda x, y: (f(x+1,y-1)+2*f(x+1,y)+f(x+1,y+1))-(f(x-1,y-1)+2*f(x-1,y)+f(x-1,y+1))
		gy = lambda x, y: (f(x-1,y-1)+2*f(x,y-1)+f(x+1,y-1))-(f(x-1,y+1)+2*f(x,y+1)+f(x+1,y+1))
		dd = np.zeros((8, shape[0], shape[1]), dtype = 'uint8')
		def direction(x, y, arctand):
			if (arctand > -22.5 )&(arctand <= 22.5):
				return (0 if x > 0 else 4)
			elif (arctand > 22.5)&(arctand <= 67.5):
				return (1 if x > 0 else 5)
			elif (arctand > 67.5)&(arctand <= 90.0):
				return (2 if x > 0 else 6)
			elif (arctand > -67.5)&(arctand <= -22.5):
				return (7 if x > 0 else 3)
			elif (arctand >= -90.0)&(arctand <= -67.5):
				return (6 if x > 0 else 2)
		for i in range(1, shape[0]-1):
			for j in range(1, shape[1]-1):
				x = gx(i,j)
				y = gy(i,j)
				dd[direction(x,y,math.degrees(math.atan((y/(x if x!=0 else 0.00001)))))][i][j] = 1
		return dd

def main():
	hcl = input_data([i for i in range(3755)], raw_data =  False)
	a = hcl.next_batch(50, 'train')
	input_data.show(a[0][0])

if __name__ == '__main__':
	main()