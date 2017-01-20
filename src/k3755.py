# coding: utf-8
# k3755.py
# to build the 3755 classes recognizer network with keras framework.

from keras.models import Sequential
import numpy as np
from input_data import hcl
import random

n_classes = 3755
d = hcl.input_data([i for i in range(n_classes)], raw_data = False, direct_info = True)

def generate_data(dataset, type_of_want):
	while True:
		x,y = dataset.read_direct_info_dataset_of_3755_classes_1000_batches(type_of_want)
		yield (x, y)

model = Sequential()

from keras.layers import Convolution2D, MaxPooling2D, Dense, Activation, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2, activity_l2

model.add(Convolution2D(50, 3, 3, border_mode='same', input_shape=(8,32,32), W_regularizer=l2(0.0005)))
model.add(Activation(LeakyReLU(alpha=0.33)))
model.add(Convolution2D(100, 3, 3, border_mode='same', W_regularizer=l2(0.0005)))
model.add(Activation(LeakyReLU(alpha=0.33)))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2,2), border_mode='same'))

model.add(Convolution2D(150, 3, 3, border_mode='same', W_regularizer=l2(0.0005)))
model.add(Activation(LeakyReLU(alpha=0.33)))
model.add(Dropout(0.1))
model.add(Convolution2D(200, 3, 3, border_mode='same', W_regularizer=l2(0.0005)))
model.add(Activation(LeakyReLU(alpha=0.33)))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2,2), border_mode='same'))

model.add(Convolution2D(250, 3, 3, border_mode='same', W_regularizer=l2(0.0005)))
model.add(Activation(LeakyReLU(alpha=0.33)))
model.add(Dropout(0.2))
model.add(Convolution2D(300, 3, 3, border_mode='same', W_regularizer=l2(0.0005)))
model.add(Activation(LeakyReLU(alpha=0.33)))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2,2), border_mode='same'))

model.add(Convolution2D(350, 3, 3, border_mode='same', W_regularizer=l2(0.0005)))
model.add(Activation(LeakyReLU(alpha=0.33)))
model.add(Dropout(0.3))
model.add(Convolution2D(400, 3, 3, border_mode='same', W_regularizer=l2(0.0005)))
model.add(Activation(LeakyReLU(alpha=0.33)))
model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size=(2,2), border_mode='same'))

model.add(Flatten())
model.add(Dense(900, W_regularizer=l2(0.0005)))
model.add(Activation(LeakyReLU(alpha=0.33)))
model.add(Dropout(0.5))
model.add(Dense(200, W_regularizer=l2(0.0005)))
model.add(Activation(LeakyReLU(alpha=0.33)))

model.add(Dense(n_classes))
model.add(Activation('softmax'))

path = '../data/result/'

#model.load_weights(path+'full.h5')

from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.005, momentum=0.9, nesterov=True), metrics=['accuracy'])

'''
batch_size = 1000
step = 0
accurancy = 0

while accurancy <= 0.50:
	x, y = get_next_batch_data(d, batch_size, 'train')
	loss, accurancy = model.train_on_batch(x, y)
	step += 1
	print '<=0.50 step', step, 'loss', loss, 'accurancy', accurancy

# save and restore
print 'saving...'
model.save_weights(path+'full50.h5')

x,y = get_next_batch_data(d, 1000, 'test')
loss_and_metrics = model.evaluate(x, y)
print loss_and_metrics

while accurancy <= 0.80:
	x, y = get_next_batch_data(d, batch_size, 'train')
	loss, accurancy = model.train_on_batch(x, y)
	step += 1
	print '<=0.80 step', step, 'loss', loss, 'accurancy', accurancy

# save and restore
print 'saving...'
model.save_weights(path+'full80.h5')

x,y = get_next_batch_data(d, 1000, 'test')
loss_and_metrics = model.evaluate(x, y)
print loss_and_metrics

while accurancy <= 0.95:
	x, y = get_next_batch_data(d, batch_size, 'train')
	loss, accurancy = model.train_on_batch(x, y)
	step += 1
	print '<=0.95 step', step, 'loss', loss, 'accurancy', accurancy

# save and restore
print 'saving...'
model.save_weights(path+'full95.h5')

x,y = get_next_batch_data(d, 1000, 'test')
loss_and_metrics = model.evaluate(x, y)
print loss_and_metrics

while accurancy <= 0.98:
	x, y = get_next_batch_data(d, batch_size, 'train')
	loss, accurancy = model.train_on_batch(x, y)
	step += 1
	print '<=0.98 step', step, 'loss', loss, 'accurancy', accurancy

# save and restore
print 'saving...'
model.save_weights(path+'full98.h5')

x,y = get_next_batch_data(d, 1000, 'test')
loss_and_metrics = model.evaluate(x, y)
print loss_and_metrics

while accurancy <= 0.995:
	x, y = get_next_batch_data(d, batch_size, 'train')
	loss, accurancy = model.train_on_batch(x, y)
	step += 1
	print '<=0.995 step', step, 'loss', loss, 'accurancy', accurancy
'''

batch_size = 1000
history = model.fit_generator(generate_data(d, 'train'), samples_per_epoch=100000, nb_epoch=10)

# save and restore
print 'saving...'
model.save_weights(path+'full995.h5')

x,y = get_next_batch_data(d, 1000, 'test')
loss_and_metrics = model.evaluate(x, y)
print loss_and_metrics
