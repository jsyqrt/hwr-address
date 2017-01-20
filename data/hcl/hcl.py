# hcl dataset.
# coding: utf-8
# this hcl dataset is HCL2000 and is realigned base on character categories.

from __future__ import division
import os
import random
import math
import numpy as np
# TODO: use numpy to optimize the save and restore of hcl dataset.

CURRENT_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]+'/'
SUB_DIRECTORY     = 'hcl'
CHARACTER_DICT    = 'GB2312_3755.txt'

class input_data:
	
	def __init__(self, chars = [], num_train = 700, num_test = 300, raw_data = True, one_hot = True, size = (32, 32), direct_info = False, fisheye = False):
		''' to get the hcl data. '''
		self.indexes   = input_data.char_to_index(chars)
		self.num_train = num_train
		self.num_test  = num_test
		self.raw_data  = raw_data
		self.one_hot   = one_hot
		self.size      = size
		self.fisheye   = fisheye
		self.direct_info = direct_info
		self.batch_index = 0
		self.batch_max   = 50

	def next_batch(self, batch_size, type_of_want = 'train'):
		''' to get next batch data. '''
		return input_data.read_data(self.indexes, self.num_train if (type_of_want == 'train') else self.num_test, self.raw_data, self.one_hot, self.size, self.direct_info, self.fisheye, type_of_want, batch_size)
	
	@staticmethod
	def read_data(indexes, num_of_every_sample, raw_data, one_hot, size, direct_info, fisheye, type_of_want, num_of_want ):
		''' read data from default directory with settings. '''
		file_path = os.path.join(CURRENT_DIRECTORY, SUB_DIRECTORY, type_of_want)
		original_index = indexes
		indexes = [indexes[random.randrange(len(indexes))] for i in range(num_of_want)]
		char_list = []
		label_list = []
		for i in indexes:
			k = (i if i != -1 else random.randrange(3755))
			filename = os.path.join(file_path, str(k+1)+'.chars')
			with open(filename, 'rb') as f:
				s = random.randrange(num_of_every_sample)
				r = list(map(lambda x: ord(x), f.read()[s*32*32:(s+1)*32*32]))
				if (direct_info | fisheye | (size != (32, 32))):
					character = [[r[m * 32 + n] for n in range(32)] for m in range(32)]
					character = input_data.resize(character, size) if (size != (32, 32)) else character
					character = input_data.direct_info(character) if direct_info else character
					character = input_data.fisheye(character) if fisheye else character
					character = character if not raw_data else (reduce(lambda x, y: x+y, character) if not direct_info else reduce(lambda x, y: x+y, reduce(lambda x, y: x+y, character)))
				else:
					character = r if raw_data else [[[r[m * 32 + n] for n in range(32)] for m in range(32)]]
				char_list.append(character)
				if one_hot:
					label = [0 for x in range(len(original_index))]
					label[original_index.index(i)] = 1
					label_list.append(label)
				else:
					label_list.append(i)
		return [char_list, label_list]

	@staticmethod
	def char_to_index(chars):
		''' to get the index of the characters in list `chars`. '''
		if type(chars[0]) == int:
			return chars
		elif type(chars[0]) == str:
			with open(CURRENT_DIRECTORY + CHARACTER_DICT, 'r') as f:
				char_dict = f.readlines()
			indexes = [(char_dict.index(i+'\n') if (i+'\n' in char_dict) else -1) for i in chars]
			return indexes
		else:
			return [-1]

	@staticmethod
	def fisheye(character):
		'''to get the fisheye change of the character.'''
		# TODO: FIX THIS FUNCTION.
		size = [len(character), len(character[0])]
		r = size[0]//1.5
		new = [[0 for j in range(size[1])] for i in range(size[0])]
		f = lambda x,y,r:complex(x,y)*r/(r+abs(complex(x,y)))
		m = size[0]//2
		n = size[1]//2
		for i in range(size[0]):
			for j in range(size[1]):
				ff = f(abs(i-m),abs(j-n),r)
				a, b = int(ff.real), int(ff.imag)
				x = (m - a) if i< m else (m + a)
				y = (n - b) if j< n else (n + b)
				new[x][y] = character[i][j]
		ll = lambda l, r:math.sin(l/r)*r
		l = int(ll(m, r))
		h = int(ll(n, r))
		new = [i[n-h:n+h] for i in new[m-l:m+l]]
		new = input_data.resize(new, size)
		return new

	@staticmethod
	def show(char_matrix, raw = False):
		''' to show the character matrix. '''
		if not raw:
			for i in char_matrix:
				s=''.join(list(map(str,i)))
				print s
		else:
			size = int(pow(len(char_matrix), 0.5))
			s = [[char_matrix[size * m + n] for n in range(size)] for m in range(size)]
			input_data.show(s)

	@staticmethod
	def resize(character, size):
		''' to resize the character with `size`'''
		s_size = (len(character), len(character[0]))
		a,b = size[0]/s_size[0],size[1]/s_size[1]
		if a==b==1:
			return character
		character = [[character[int(i/a)][int(j/b)] for j in range(size[1])]for i in range(size[0])]
		return character
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
		''' to generate a 8xsize div tensor to get character's direction information with sobel operator, gradient.'''
		f = lambda x, y: character[x][y]
		gx = lambda x, y: (f(x+1,y-1)+2*f(x+1,y)+f(x+1,y+1))-(f(x-1,y-1)+2*f(x-1,y)+f(x-1,y+1))
		gy = lambda x, y: (f(x-1,y-1)+2*f(x,y-1)+f(x+1,y-1))-(f(x-1,y+1)+2*f(x,y+1)+f(x+1,y+1))
		dd = [[[0 for i in range(len(character[0]))] for j in range(len(character))] for k in range(8)]
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
		for i in range(1, len(character)-1):
			for j in range(1, len(character[0])-1):
				x = gx(i,j)
				y = gy(i,j)
				dd[direction(x,y,math.degrees(math.atan((y/(x if x!=0 else 0.00000001)))))][i][j] = 1
		return dd

def main():
	hcl = input_data([i for i in range(3755)], raw_data =  False)
	a = hcl.next_batch(50, 'train')
	input_data.show(a[0][0])
	print len(a),len(a[0]),len(a[1]),len(a[0][0])

if __name__ == '__main__':
	main()