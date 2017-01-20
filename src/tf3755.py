# coding: utf-8
# tf3755.py
# build a cnn of 3755 classes with tensorflow.

import cnnm
import tensorflow as tf

x_shape = 1024*8
cnn_reshape = [-1, 32, 32, 1]
y_shape = 100
cnn_layer_n = 8
cnn_weights = [
	[3,3,1,50],
	[3,3,50,100],
	[3,3,100,150],
	[3,3,150,200],
	[3,3,200,250],
	[3,3,250,300],
	[3,3,300,350],
	[3,3,350,400]
]
keep_prob = [0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.5, 0]
max_pooling = [0, 1, 0, 1, 0, 1, 0, 1]
fnn_reshape = [-1, 400*2*2]
fnn_layer_n = 2
fnn_weights = [
	[400*2*2, 900],
	[900, 200]
]
softmax_weight = [200, y_shape]
saver_path = '../data/result/full.bin'

from input_data import hcl
b = hcl.input_data([i for i in range(y_shape)], direct_info=True)

full_recognizer = cnnm.cnn('full_recognizer', x_shape, cnn_reshape, y_shape, cnn_layer_n, cnn_weights, keep_prob, max_pooling, fnn_reshape, fnn_layer_n, fnn_weights, softmax_weight, saver_path)

batch_size = 1000
global_step = tf.Variable(0, name='global_step', trainable=False)
start_learning_rate = 0.005
full_steps = 1000000
show_step = 20
decay_rate = 0.96
learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, full_steps, decay_rate)
momentum = 0.9
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
train_step = optimizer.minimize(full_recognizer.cross_entropy, global_step = global_step)
sess = tf.InteractiveSession()

init_op = tf.initialize_all_variables()
sess.run(init_op)

for step in range(full_steps):
	global_step  = step
	batch = b.next_batch(batch_size, 'train')
	if step%show_step == 0:
		train_accuracy = full_recognizer.accuracy.eval(feed_dict={full_recognizer.x:batch[0], full_recognizer.y_:batch[1]})
		print 'step %d, train accuracy %g' %(step, train_accuracy)
	sess.run([train_step], feed_dict={full_recognizer.x:batch[0], full_recognizer.y_:batch[1]})