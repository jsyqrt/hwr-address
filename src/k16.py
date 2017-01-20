# coding: utf-8
# k16.py
# to build the keyword recognizer network with keras framework.

from keras.models import Sequential
import numpy as np
from input_data import *

d = hcl.input_data(keyword_list, raw_data=False, direct_info=True)

def generate_data(dataset, batch_size, type_of_want):
	while True:
		x,y = dataset.next_batch(batch_size, type_of_want)
		yield (np.array(x), np.array(y))

model = Sequential()

from keras.layers import Convolution2D, MaxPooling2D, Dense, Activation, Dropout, Flatten

model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(8, 32, 32)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), border_mode='same'))
model.add(Dropout(1))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(1))
model.add(MaxPooling2D(pool_size=(2,2), border_mode='same'))

model.add(Flatten())
model.add(Dense(1024, init='normal'))
model.add(Activation('relu'))
model.add(Dropout(1))

model.add(Dense(16, init='normal'))
model.add(Activation('softmax'))

path = '../data/result/'

model.load_weights(path+'kwd.h5')

from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])

batch_size = 100
history = model.fit_generator(generate_data(d, batch_size, 'train'), samples_per_epoch=10000, nb_epoch=10)

print 'saving...'
#model.save_weights(path+'kwd.h5')

x,y = d.next_batch(1000, 'test')
loss_and_metrics = model.evaluate(x, y)
print loss_and_metrics

'''
classes = model.predict_classes(x[:3])
print classes
proba = model.predict_proba(x[:3])
print proba
'''
