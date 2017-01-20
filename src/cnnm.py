# coding: utf-8
# cnnm.py
# to build a normal cnn (several convolutional layers first, then dense layers next) with tf.

import tensorflow as tf

class cnn:
	''' to build a cnn network with n-layer-cnn and m-layer-fnn. tensorflow. '''
	def __init__(self, name, x_shape, cnn_reshape, y_shape, cnn_layer_n, cnn_weights, keep_prob, max_pooling, \
		fnn_reshape, fnn_layer_n, fnn_weights, softmax_weight, saver_path = None):
		print 'building %s cnn.' %name
		self.name = name
		self.saver_path = saver_path
		self.x = tf.placeholder(tf.float32, shape=[None, x_shape])
		self.y = tf.placeholder(tf.float32, shape=[None, y_shape])
		self.cnn_layers = cnn.make_cnn_layers(self.x, cnn_reshape, cnn_layer_n, cnn_weights, keep_prob, max_pooling)
		self.fnn_layers = cnn.make_fnn_layers(self.cnn_layers[-1], cnn_layer_n, fnn_reshape, fnn_layer_n, fnn_weights, keep_prob)
		self.y_conv = cnn.make_softmax(self.fnn_layers[-1], softmax_weight)
		self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.y_conv), reduction_indices=[1]))
		self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y,1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		
	@staticmethod
	def make_cnn_layers(x, reshape, cnn_layer_n, weights, keep_prob, max_pooling = None):
		if not max_pooling:
			max_pooling = [0 for i in range(cnn_layer_n)]
		x_image = tf.reshape(x, reshape)
		cnn_layers = []
		f = lambda x, i: cnn.max_pool_2x2(tf.nn.dropout(tf.maximum((cnn.conv2d(x, cnn.weight_variable(weights[i])) + cnn.bias_variable([weights[i][-1]]))*(0.33),(cnn.conv2d(x, cnn.weight_variable(weights[i])) + cnn.bias_variable([weights[i][-1]]))), keep_prob[i])) if max_pooling[i] else tf.nn.dropout(tf.maximum((cnn.conv2d(x, cnn.weight_variable(weights[i])) + cnn.bias_variable([weights[i][-1]]))*0.33,(cnn.conv2d(x, cnn.weight_variable(weights[i])) + cnn.bias_variable([weights[i][-1]]))), keep_prob[i]) # change relu to leaky-relu(f(x) = max(x,0)+λmin(x,0))
		x = f(x_image, 0)
		cnn_layers.append(x)
		for i in range(1,cnn_layer_n):
			x = f(x,i)
			cnn_layers.append(x)
		return cnn_layers
		
	@staticmethod
	def make_fnn_layers(x, cnn_layer_n, fnn_reshape, fnn_layer_n, fnn_weights, keep_prob):
		x = tf.reshape(x, fnn_reshape)
		f = lambda x,i: tf.nn.dropout(tf.maximum((tf.matmul(x, cnn.weight_variable(fnn_weights[i])) + cnn.bias_variable([fnn_weights[i][-1]]))*0.33, (tf.matmul(x, cnn.weight_variable(fnn_weights[i])) + cnn.bias_variable([fnn_weights[i][-1]]))), keep_prob[cnn_layer_n + i])
		fnn_layers = []
		for i in range(0,fnn_layer_n):
			x = f(x,i)
			fnn_layers.append(x)
		return fnn_layers

	@staticmethod
	def weight_variable(shape):
		initial = tf.random_normal(shape, stddev=0.01)
		return tf.Variable(initial)
	@staticmethod
	def bias_variable(shape):
		initial = tf.constant(0.0, shape=shape)
		return tf.Variable(initial)
	@staticmethod
	def conv2d(x, W):
		return tf.nn.conv2d(x,W, strides=[1,1,1,1],padding='SAME')
	@staticmethod
	def max_pool_2x2(x):
		return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	@staticmethod
	def make_softmax(x, weight):
		return tf.nn.softmax(tf.matmul(x, cnn.weight_variable(weight)) + cnn.bias_variable([weight[-1]]))
