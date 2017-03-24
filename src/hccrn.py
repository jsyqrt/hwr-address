# coding: utf-8
# hccrc.py , Handwritten Chinese Character Recognition Network with keras

import os
import math
from keras.models import Sequential
import numpy as np
import input_data
import random
from keras.optimizers import SGD
from keras.layers import Convolution2D, MaxPooling2D, Dense, Activation, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2, activity_l2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping

def model_define(input_shape=(1, 32, 32), output_shape=16):	
	model = Sequential()

	model.add(Convolution2D(50, 3, 3, border_mode='same', input_shape=input_shape, W_regularizer=l2(0.0005)))
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
	model.add(Dense(500, W_regularizer=l2(0.0005)))
	model.add(Activation(LeakyReLU(alpha=0.33)))
	model.add(Dropout(0.5))
	model.add(Dense(200, W_regularizer=l2(0.0005)))
	model.add(Activation(LeakyReLU(alpha=0.33)))

	model.add(Dense(output_shape))
	model.add(Activation('softmax'))

	return model

def model_restore(model, weights_path):
	model.load_weights(weights_path)
	return model

def model_compile(model):
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	return model
		
def model_train(model, dataset, batch_size, weights_path, history_path, nb_epoch=200, samples_per_epoch=1000000):
	checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=False, save_weights_only=True)
	#lrate = LearningRateScheduler(step_decay)
	lrate = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=5, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.00001)
	#early_stop = EarlyStopping(monitor='loss', patience=5, verbose=0, mode='auto')
	history = model.fit_generator(input_data.generate_data(dataset, batch_size), samples_per_epoch, nb_epoch, callbacks=[checkpointer, lrate])
	with open(history_path, 'w') as f:
		f.write(str(history.history))
	return model
	
def step_decay(epoch):
	initial_lrate = 0.005
	drop = 0.9
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
	return lrate

def model_test(model, dataset, batch_size):
	print 'model testing...'
	x, y = dataset.next_batch(batch_size, 'test')
	loss_and_metrics = model.evaluate(x, y)
	print 'testing result: loss and accurancy: ', loss_and_metrics

def model_predict(model, x):
	proba = model.predict_proba(x)
	return proba

class recognizer:

	def __init__(self, dataset, input_shape, output_shape, weights_dir, weights_name, history_name, batch_size, nb_epoch, samples_per_epoch, retrain = False):

		print 'train info: input_shape:', input_shape, 'output_shape:', output_shape, 'weights_dir:', weights_dir, 'weights_name:', weights_name, 'history_name:', history_name, 'batch_size:', batch_size, 'nb_epoch:', nb_epoch, 'samples_per_epoch:', samples_per_epoch, 'retrain:', retrain
		
		self.weights_path = weights_dir + weights_name
		self.history_path = weights_dir + history_name
		
		self.model = model_define(input_shape, output_shape)
		if os.path.exists(self.weights_path) & (not retrain):
			self.model = model_restore(self.model, self.weights_path)
			self.model = model_compile(self.model)
		else:
			self.model = model_compile(self.model)
			self.model = model_train(self.model, dataset, batch_size, self.weights_path, self.history_path, nb_epoch, samples_per_epoch)
		#model_test(self.model, dataset, batch_size * nb_epoch)
	
	def recognize(self, x, char_list = input_data.keyword_list):
		#return input_data.index_to_char(self.model.predict_classes(x), char_list)
		return input_data.proba_to_char(model_predict(self.model, x), char_list)

if __name__ == '__main__':
	import sys
	network = sys.argv[1]
	retrain = (sys.argv[2] == 'y')
	weights_dir = '../data/result/'
	if (network == 'k'):
		input_shape = (1, 32, 32)
		output_shape = 16
		batch_size = 100
		nb_epoch = 10
		samples_per_epoch = 1000000
		weights_name = 'kwd_weights.hdf5'
		history_name = 'kwd_history.txt'
		dataset = input_data.hcl.input_data(input_data.keyword_list, raw_data = False, direct_info = False)
		
		kwd_recognizer = recognizer(dataset, input_shape, output_shape, weights_dir, weights_name, history_name, batch_size, nb_epoch, samples_per_epoch, retrain)
		print kwd_recognizer.recognize(dataset.next_batch(50, 'test')[0])

	elif (network == 'f'):
		input_shape = (1, 32, 32)
		output_shape = len(input_data.full_list)
		batch_size = 1200
		nb_epoch = 10
		samples_per_epoch = 12000000
		weights_name = 'full_weights.hdf5'
		history_name = 'full_history.txt'
		dataset = input_data.hcl.input_data(input_data.full_list, raw_data = False, direct_info = False)

		full_recognizer = recognizer(dataset, input_shape, output_shape, weights_dir, weights_name, history_name, batch_size, nb_epoch, samples_per_epoch, retrain)
	else:
		print 'wrong input! quit...'