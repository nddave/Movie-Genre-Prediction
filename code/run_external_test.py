
'''
Project: Predicting movie genres from movie posters
Course: COMPSCI 682 Neural Networks: A Modern Introduction

File: run_external_test.py
Description: Runs test for an external image from its URL on the internet.
Author: Nirman Dave
'''

import sys
import operator
import numpy as np
import data_load as dl
from os import listdir
import data_manage as dm
from os.path import isfile, join
from keras.models import load_model

models_path = str(sys.argv[1]) + '/'
eval_models = True
verbose = True
crop = 3

class TransferModel:
	'''
	TransferModel is an object that stores all saved model properties.
	'''

	min_year = 0
	max_year = 0
	genres = []
	ratio = 0
	epochs = 0
	style = 1
	file_path = ''
	model = None

	def eval(self):
		'''
		Get scores on input data.
		'''
		print('Loading test data...')
		x_test, y_test = dl.load_data(self.min_year, self.max_year, self.genres, self.ratio, set_type='test', verbose=False)
		print('Evaluating model...')
		scores = self.model.evaluate(x_test, y_test, verbose=0)
		print('Test loss =', scores[0])
		print('Test accuracy =', scores[1])

	def predict(self, movie):
		'''
		Make a prediction using this model.
		'''
		x = [movie.img_to_rgb(self.ratio)]
		x = np.array(x, dtype='float32')
		return self.model.predict(x)

	def load(self):
		'''
		Load the model for testing.
		'''
		self.model = load_model(self.file_path)

	def __str__(self):
		return (
				'Model v' + str(self.style) \
				+ ' (' + str(self.min_year) + '-' + str(self.max_year) \
				+ ' / g' + str(len(self.genres)) \
				+ ' / r' + str(self.ratio) \
				+ ' / e' + str(self.epochs) \
				+ ')'
			)

def parse_model(file_name):
	'''
	Parse the model from its name.
	'''
	split = file_name.split('_')
	parsed = TransferModel()
	parsed.min_year = int(split[3])
	parsed.max_year = int(split[4])
	parsed.genres = ['Horror', 'Romance', 'Action', 'Documentary']
	parsed.ratio = int(split[6][1:])
	parsed.epochs = int(split[7][1:])
	parsed.style = int(split[8].split('.')[0][1:])
	parsed.file_path = file_name
	return parsed

def list_models():
	return sorted([f for f in listdir(models_path) if isfile(join(models_path, f)) and f.startswith('genres_')])

def repeat_length(string, length):
	return (string * (int(length / len(string)) + 1))[:length]

def format_preds(movie, genres, preds):
	'''
	Format predictions from multi-hot encoding to human readable structure.
	'''

	preds_map = {}
	for i in range(len(genres)):
		preds_map[genres[i]] = preds[0][i]

	sorted_preds = sorted(preds_map.items(), key=operator.itemgetter(1), reverse=True)

	preds_str = []
	for genre, probability in sorted_preds:
		if genre in movie.genres:
			is_present = ''
		else:
			is_present = '[!]'
		preds_str.append(genre + is_present + ': ' + "{:.0%}".format(probability))

	spaces = repeat_length(' ', 33 - len(str(movie)))

	if crop is not None:
		return str(movie) + spaces + str(preds_str[:crop])
	else:
		return str(movie) + spaces + str(preds_str)

def main():
	'''
	Test the external movie using models in the models_path directory.
	'''

	for model_file in list_models():
		saved_model = parse_model(models_path + model_file)
		saved_model.load()
		print('')
		print('------------------------------------------------------------------------')
		print(saved_model)

		test_movies = {}

		test_movies['?'] = [
			str(sys.argv[2]),
		]

		if verbose:
			for expected_genre, movies_titles in sorted(test_movies.items()):
				for movie_title in movies_titles:
					movie = dm.search_movie_external(title=movie_title)
					if movie is not None:
						preds = saved_model.predict(movie)
						print(format_preds(movie, saved_model.genres, preds))
					else:
						print(movie_title + ' not found')
				print('------------------------------------------------------------------------')

if __name__ == '__main__':
	main()


























