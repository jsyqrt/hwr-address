# coding: utf-8
# at.py
# address tree.

import os
CURRENT_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]+'/'
from hccrn import recognizer
f = recognizer(*input_data.arguments['full'])
k = recognizer(*input_data.arguments['kwd'])


class node:
	def __init__(self, data):
		self._data = data
		self._children = []

	def getdata(self):
		return self._data

	def getchildren(self):
		return self._children
	
	def getlen(self):
		min = 100
		max = 0
		for i in self._children:
			if len(i) < min:
				min = len(i)
			if len(i) > max:
				max = len(i)
		return [min, max]
	
	def add(self, node):
		self._children.append(node)

	def go(self, data):
		for child in self._children:
			if child.getdata() == data:
				return child
		return None

class tree:
	def __init__(self):
		self._head = node('header')

	def linktohead(self, node):
		self._head.add(node)

	def insert(self, path, data):
		cur = self._head
		for step in path:
			if cur.go(step) == None:
				return False
			else:
				cur = cur.go(step)
		cur.add(node(data))
		return True

	def search(self, path):
		cur = self._head
		for step in path:
			if cur.go(step) == None:
				return None
			else:
				cur = cur.go(step)
		return cur

def build_address_tree(dir_string_list=''):
	lt = tree()
	root = node(u'中国')
	lt.linktohead(root)
	path = [u'中国']
	for string in dir_string_list:
		x = depth_and_clear(string)
		path = path[:x[0]]
		x[1] = x[1].decode('utf-8')
		lt.insert(path, x[1])
		path.append(x[1])
	return lt

def depth_and_clear(string_with_space):
	i=0
	for string in string_with_space:
		if string == '	':
			i += 1
	string_with_space = string_with_space.replace('	', '')
	return [i,string_with_space]

def read_dir_string_list_from(filename):
	with open(filename, 'r') as f:
		r = list(map(lambda x:x.replace('\n', ''), f.readlines()))
		return r

def at():
	''' return an address tree. '''
	print 'building address tree.'
	import pickle
	address_tree_path = os.path.join(CURRENT_DIRECTORY, 'at.t')
	address_file_path = os.path.join(CURRENT_DIRECTORY, 'lt.txt')
	
	'''
	if not os.path.exists(address_tree_path):
		d = read_dir_string_list_from(address_file_path)
		t = build_address_tree(d)
		with open(address_tree_path,'wb') as f:
			pickle.dump(t, f)
		return t
	else:
		with open(address_tree_path, 'rb') as f:
			t = pickle.load(f)
		return t
	'''
	d = read_dir_string_list_from(address_file_path)
	t = build_address_tree(d)
	return t

def generate_regular_address_string(want_num = 1000, file_path = './ra.txt', to_country = True):
	import random
	t = at()
	path = [u'中国']
	addresses = []
	while len(addresses) < 1000:
		print len(addresses)
		try:
			prov = t.search(path).getchildren()[random.randrange(len(t.search(path).getchildren()))].getdata()
			path.append(prov)
			city = t.search(path).getchildren()[random.randrange(len(t.search(path).getchildren()))].getdata()
			path.append(city)
			county = t.search(path).getchildren()[random.randrange(len(t.search(path).getchildren()))].getdata()
			path.append(county)
			if to_country:
				town = t.search(path).getchildren()[random.randrange(len(t.search(path).getchildren()))].getdata()
				path.append(town)
				country = t.search(path).getchildren()[random.randrange(len(t.search(path).getchildren()))].getdata()
				path.append(country)
			else:
				pass
			
			a =''.join(path[1:])
			addresses.append(a)
			print a
			path = path[:1]
		except:
			path = path[:1]
			print 'search error'
	with open(file_path, 'w') as f:
		for i in addresses:
			f.write(i.encode('utf8')+'\n')
'''
def validate(t, k_char_list, full_char_list):

	import sys
	sys.path.append('../../tools/')	
	import tools
	result = [full_char_list[i][0][0].decode('utf8') if k_char_list[i][0][0] == 'X' else k_char_list[i][0][0].decode('utf8') for i in range(len(full_char_list))]
	full_result = ''.join(result).encode('utf8')
	
	try:
		path = [u'中国']
		index = 0
		while index < len(full_char_list):
			addresss = t.search(path).getchildren()
			addresssc = {}
			for i in addresss:
				x = i.getdata()
				sim = tools.levenshtein(result[index:index+len(x)], x)
				addresssc[sim] = x
			address = addresssc[max(addresssc.keys())]
			path.append(address)
			index += len(address)
		result = ''.join(path[1:]).encode('utf8')
	except:
		result = full_result

	return (result, full_result)
'''
def validate(t, k_char_list, full_char_list, charlist):

	import sys
	sys.path.append('../../tools/')	
	import tools
	#result without address database
	result = [full_char_list[i][0][0].decode('utf8') if k_char_list[i][0][0] == 'X' else k_char_list[i][0][0].decode('utf8') for i in range(len(full_char_list))]
	full_result = ''.join(result).encode('utf8')
	
	#result with address database
	split_words_pos = []
	i = 0
	while i < len(key_char_list):
		j = i
		cnt = 0
		while key_char_list[j][0][0]=='X':
			cnt += 1
			j += 1
		#include key words
		split_words_pos.append([i, cnt+1])
		i = j + 1
	try:
		path = [u'中国']
		index = 0
		while index < len(split_words_pos):
			addresss = t.search(path).getchildren()
			[min, max] = t.search(path).getlen()
			init = index
			word_len = split_words_pos[init][1]
			start = split_words_pos[init][0]
			index += 1
			while word_len < min:
				split_words_pos[init][1] += split_words_pos[index][1]
				index += 1
			end = start + split_words_pos[init][1]
			word_char_list = f.recognize(charlist[start, end], input_data.full_list)
			word = ''.join(word_char_list).encode('utf8')
			addresssc = {}
			for i in addresss:
				x = i.getdata()
				sim = tools.levenshtein(word, x)
				addresssc[sim] = x
			address = addresssc[max(addresssc.keys())]
			path.append(address)
		result = ''.join(path[1:]).encode('utf8')
	except:
		result = full_result

	return (result, full_result)
