
'''
Project: Predicting movie genres from movie posters
Course: COMPSCI 682 Neural Networks: A Modern Introduction

File: model.py
Description: Model for training the data.
Author: Nirman Dave
'''

import os
import keras
import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dense, Dropout, Flatten

def build(style, min_year, max_year, genres, ratio, epochs, lrate=1e-3, x_train=None, x_val=None, y_train=None, y_val=None, verbose=True):
	'''
	Builds a ML model using training and validation set.
	'''

	if verbose:
		print('')
		print('x_train shape:', x_train.shape)
		print('y_train shape:', y_train.shape)
		print('x_val shape:', x_val.shape)
		print('y_val shape:', y_val.shape)
		print('train samples:', x_train.shape[0])
		print('val samples:', x_val.shape[0])
		print('')

	num_classes = len(y_train[0])

	model = Sequential([

			# Layer 1: Convolution with ReLU activation
			Conv2D(32, (3, 3), input_shape=x_train.shape[1:]),
			Activation('relu'),
			MaxPooling2D(pool_size=(2,2)),

			# Layer 2: Convolution with ReLU activation
			Conv2D(32, (3, 3)),
			Activation('relu'),
			MaxPooling2D(pool_size=(2,2)),

			# Layer 3: Convolution with ReLU activation
			Conv2D(32, (3, 3)),
			Activation('relu'),
			MaxPooling2D(pool_size=(2,2)),

			# Layer 4: Convolution with ReLU activation
			Conv2D(64, (3, 3)),
			Activation('relu'),
			MaxPooling2D(pool_size=(2,2)),

			# Layer 5: Fully Connected Layer with ReLU activation
			Flatten(),
			Dense(64),
			Activation('relu'),
			Dropout(0.5),

			# Layer 6: Fully Connected Layer with ReLU activation
			Dense(32),
			Activation('relu'),
			Dropout(0.5),

			# Layer 7: Fully Connected Layer with Softmax activation
			Dense(num_classes),
			Activation('softmax'),

		])

	# Optimizer used is Adam with varying learning rate
	new_adam = keras.optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

	model.compile(

			# Using Categorical Crossentropy loss due to multi-hot encoding
			loss='categorical_crossentropy',
			optimizer=new_adam,
			metrics=['accuracy']

		)

	print(model.summary())

	# Saving model history to variable h
	h = model.fit(

			x_train,
			y_train,
			batch_size=32,
			epochs=epochs,
			validation_data=(x_val, y_val),

		)

	save_dir = os.path.join(os.getcwd(), 'cnn_model_results/models')
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)

	model_name = 'genres' \
				+ '_' + str(min_year) + '_' + str(max_year) \
				+ '_g' + str(len(genres)) \
				+ '_r' + str(ratio) \
				+ '_e' + str(epochs) \
				+ '_v' + str(style) \
				+ '_lr' + str(lrate) + '.h5'

	model_path = os.path.join(save_dir, model_name)
	model.save(model_path)

	save_dir_2 = os.path.join(os.getcwd(), 'cnn_model_results/hists')
	if os.path.isdir(save_dir_2) == False:
		os.makedirs(save_dir_2)

	hist_name = 'hists' \
				+ '_' + str(min_year) + '_' + str(max_year) \
				+ '_g' + str(len(genres)) \
				+ '_r' + str(ratio) \
				+ '_e' + str(epochs) \
				+ '_v' + str(style) \
				+ '_lr' + str(lrate) + '.pkl'

	hist_path = os.path.join(save_dir_2, hist_name)

	# Dumping model history to pickle file
	with open(hist_path, 'wb') as f:
		pickle.dump(h.history, f)

	print('Saved trained model at %s ' % model_path)

