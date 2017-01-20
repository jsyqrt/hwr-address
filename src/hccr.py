# coding: utf-8
# hccr.py
# to realize the network structure described by the paper 
# ( Online and Offline Handwritten Chinese Character Recognition:
#  A Comprehensive Study and New Benchmark).

import tensorflow as tf

def leaky_relu(x, alpha = 0.33):
	return tf.maximum(x*alpha, x)
def weight_variable(shape):
	initial = tf.random_normal(shape, stddev=0.01)
	return tf.Variable(initial)
def bias_variable(shape):
	initial = tf.constant(0.0, shape=shape)
	return tf.Variable(initial)
def conv2d(x, W):
	return tf.nn.conv2d(x,W, strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
def make_softmax(x, weight):
	return tf.nn.softmax(tf.matmul(x, weight_variable(weight)) + bias_variable([weight[-1]]))

x_shape = 1024*8
y_shape = 3755

cnn_weights = {
		'layer1':[3,3,8,50],
		'layer2':[3,3,50,100],
		'layer3':[3,3,100,150],
		'layer4':[3,3,150,200],
		'layer5':[3,3,200,250],
		'layer6':[3,3,250,300],
		'layer7':[3,3,300,350],
		'layer8':[3,3,350,400]
}
fnn_weights = {
		'layer9':[400*4, 900],
		'layer10':[900, 200]
}
softmax_weight = [200, y_shape]
drops = [0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.5, 0.0]

x = tf.placeholder(tf.float32, shape=[None, x_shape])
y = tf.placeholder(tf.float32, shape=[None, y_shape])

cnn_inputx = tf.reshape(x, [-1, 32, 32, 8])

layer1 = leaky_relu(conv2d(cnn_inputx, weight_variable(cnn_weights['layer1'])) + bias_variable([cnn_weights['layer1'][-1]]))
layer2 = max_pool_2x2(tf.nn.dropout(leaky_relu(conv2d(layer1, weight_variable(cnn_weights['layer2'])) + bias_variable([cnn_weights['layer2'][-1]])), drops[1]))
		
layer3 = tf.nn.dropout(leaky_relu(conv2d(layer2, weight_variable(cnn_weights['layer3'])) + bias_variable([cnn_weights['layer3'][-1]])), drops[2])
layer4 = max_pool_2x2(tf.nn.dropout(leaky_relu(conv2d(layer3, weight_variable(cnn_weights['layer4'])) + bias_variable([cnn_weights['layer4'][-1]])), drops[3]))
		
layer5 = tf.nn.dropout(leaky_relu(conv2d(layer4, weight_variable(cnn_weights['layer5'])) + bias_variable([cnn_weights['layer5'][-1]])), drops[4])
layer6 = max_pool_2x2(tf.nn.dropout(leaky_relu(conv2d(layer5, weight_variable(cnn_weights['layer6'])) + bias_variable([cnn_weights['layer6'][-1]])), drops[5]))
		
layer7 = tf.nn.dropout(leaky_relu(conv2d(layer6, weight_variable(cnn_weights['layer7'])) + bias_variable([cnn_weights['layer7'][-1]])), drops[6])
layer8 = max_pool_2x2(tf.nn.dropout(leaky_relu(conv2d(layer7, weight_variable(cnn_weights['layer8'])) + bias_variable([cnn_weights['layer8'][-1]])), drops[7]))
		
fnn_inputx = tf.reshape(layer8, [-1, 1600])

layer9 = tf.nn.dropout(leaky_relu(tf.matmul(fnn_inputx, weight_variable(fnn_weights['layer9'])) + bias_variable([fnn_weights['layer9'][-1]])), drops[8])
layer10 = leaky_relu(tf.matmul(layer9, weight_variable(fnn_weights['layer10'])) + bias_variable([fnn_weights['layer10'][-1]]))

layer11 = make_softmax(layer10, softmax_weight)

out = layer11

#loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(out), reduction_indices=[1]))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, y))

learn_rate = 0.005
decay_rate = 0.96
full_step = 100000
batch_size = 1000
num_classes = y_shape
show_step = 20

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(learn_rate, global_step, full_step, decay_rate, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)

correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

from input_data import hcl
b = hcl.input_data([i for i in range(num_classes)], direct_info = True)

for step in range(full_step):
	batch = b.next_batch(batch_size)
	if step%show_step:
		acc, cost, rate = sess.run([accuracy, loss, lr], feed_dict={x: batch[0], y: batch[1]})
		print 'lr ', rate, ' step ', step, 'Minibatch Loss= ', '{:.6f}'.format(cost), 'Training Accuracy= ', '{:.5f}'.format(acc)
	sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})
