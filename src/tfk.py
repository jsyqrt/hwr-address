# coding: utf-8
# tfk.py
# to build a cnn to recognize keyword with tensorflow.

from input_data import keyword_list

saver_path = '../data/result/keyword_recognizer.bin'
x_shape = 1024
cnn_reshape = [-1,32,32,1]
y_shape = len(keyword_list)
cnn_layer_n = 2
cnn_weights = [[3, 3, 1, 32], [3, 3, 32, 64]]
keep_prob = [1, 1, 1, 1, 0.5]
max_pooling = [1,1]
fnn_reshape = [-1, 8*8*64]
fnn_layer_n = 1
fnn_weights = [[8*8*64, 1024]]
softmax_weight = [1024, 16]
retrain = False

def __init__():
	pass

class keyword_recognizer:
	def __init__(self):
		self.keyword_recognizer = cnn.cnn('keyword_recognizer', x_shape, cnn_reshape, y_shape, cnn_layer_n, cnn_weights, keep_prob, max_pooling, fnn_reshape, fnn_layer_n, fnn_weights, softmax_weight, saver_path)
		if retrain:
			from input_data import hcl
			self.hcl = hcl.input_data(keyword_list)
		self.keyword_recognizer.prepare(retrain, self.hcl, 100000)

	def recognize(self, charlist):
		p = self.keyword_recognizer.predict(charlist)
		y = [i.index(max(i)) for i p]
		return [y, p]