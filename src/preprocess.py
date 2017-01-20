# coding: utf-8
# preprocess.py
# to get the binary matrix list of a given picture.

from PIL import Image

def preprocess(img, wanna_size):
	matrix = img_to_matrix(img)
	charlist = matrix_to_charlist(matrix)
	charlist = resize_all(charlist, wanna_size)
	return charlist

def img_to_matrix(img):
	if img.mode != '1':
		img = img.convert('1')
	d = list(map(lambda x: 0 if x == 255 else 1, img.getdata()))
	size = img.size
	print size
	m = [[d[i*size[1]+j] for j in range(size[0])] for i in range(size[1])]
	return m

def matrix_to_charlist(matrix):
	charlist = []
	'''
	for i in range(len(matrix[0])/64):
		c = []
		for j in range(64):
			c.append(matrix[j][i*64:i*64+64])
		charlist.append(c)
	'''
	charlist = []
	for i in range(64):
		charlist.append(matrix[i][:64])
	return [charlist]
	
def resize_all(charlist, size):
	def resize(character, size):
		s_size = (len(character), len(character[0]))
		a,b = size[0]/s_size[0],size[1]/s_size[1]
		if a==b==1:
			return character
		character = [[character[int(i/a)][int(j/b)] for j in range(size[1])]for i in range(size[0])]
		return character
	return [resize(i, size) for i in charlist]

if __name__ == '__main__':
	a=Image.open('../data/sample/samples/辽宁省沈阳市文化东路17号.jpg')
	x = img_to_matrix(a)
	b = matrix_to_charlist(x)
	
	import sys
	sys.path.append('../data/hcl/')
	import hcl
	hcl.input_data.show(b[0])
	print len(b)

	
	