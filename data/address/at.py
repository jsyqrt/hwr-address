# coding: utf-8
# at.py
# address tree.

import os

class node:
	def __init__(self, data):
		self._data = data
		self._children = []

	def getdata(self):
		return self._data

	def getchildren(self):
		return self._children

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
	import pickle
	if not os.path.exists('./at.t'):
		d = read_dir_string_list_from('./lt.txt')
		t = build_address_tree(d)
		with open('./at.t','wb') as f:
			pickle.dump(t, f)
		return t
	else:
		with open('./at.t', 'rb') as f:
			t = pickle.load(f)
		return t

def validate(k_char_list, full_char_list):
	'''
	input: k_char_list, full_char_list
	output: recognize_result

	examples:	k_char_list = [(['省','市','县',...,],[0.8, 0.1,0.05,0.005,...,]),(),()]
	'''
	at = at()
	root = '中国'
	path = [root]
	
	at.search(path)
