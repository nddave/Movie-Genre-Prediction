
'''
Project: Predicting movie genres from movie posters
Course: COMPSCI 682 Neural Networks: A Modern Introduction

File: test.py
Description: Tests the resulting model using testing data.
Author: Nirman Dave
Adapted from: Vleminckx, Benoit. dnn-movie-posters. (2018). Github repository. https://github.com/benckx/dnn-movie-posters.
'''

import operator
import numpy as np
import data_load as dl
from os import listdir
import data_manage as dm
from os.path import isfile, join
from keras.models import load_model

models_path = 'cnn_model_results/models/'
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
	lr = ''

	def eval(self):
		'''
		Get scores on input data.
		'''
		x_test, y_test = dl.load_data(self.min_year, self.max_year, self.genres, self.ratio, set_type='test', verbose=False)
		scores = self.model.evaluate(x_test, y_test, verbose=0)
		return scores

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
				+ ' / lr' + str(self.lr) \
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
	parsed.style = int(split[8][1:])
	parsed.lr = str(split[9].split('.h5')[0][2:])
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
		print('------------------------------------------------------------------------')
		if verbose:
			print(saved_model.model.summary())
			pass
		if eval_models:
			print(saved_model, 'tloss=', s[0], 'tacc=', s[1])
		print('------------------------------------------------------------------------')
		test_movies = {}

		test_movies['Horror'] = [
			"Dracula 2000", 
			"The Blair Witch Project", 
			"The Others", 
			"Aliens", 
			"Aliens vs. Predator: Requiem", 
			"Alien: Resurrection"
		]

		test_movies['Romance'] = [
			"Notting Hill", 
			"Pretty Woman", 
			"Bridget Jones's Diary"
		]

		test_movies['Action'] = [
			"The Matrix",
			"Man of Steel",
			"X-Men: Apocalypse",
			"Lara Croft: Tomb Raider", 
			"Edge of Tomorrow", 
			"Batman Forever", 
			"Live Free or Die Hard"
		]

		test_movies['Documentary'] = [
			"Catwalk",
			"Anne Frank Remembered",
			"Jupiter's Wife",
			"Inside Job",
			"Fahrenheit 9/11"
			"The Imposter", 
			"Cave of Forgotten Dreams"
		]

		test_movies['?'] = [
			"Beastly",
			"No Strings Attached",
			"Source Code",
			"Midnight in Paris",
			"Titanic",
			"Love Actually",
			"The Proposal",
			"The Notebook",
			"50 First Dates",
			"Pride & Prejudice",

			"The Dark Knight",
			"The Avengers",
			"Mad Max: Fury Road",
			"Star Wars: Episode VII - The Force Awakens",
			"The Bourne Supremacy",
			"John Wick",
			"The Matrix",
			"Skyfall",
			"Inception",
			"Pearl Harbor"

			"The Conjuring",
			"The Cabin in the Woods",
			"Sinister",
			"The Batman vs. Dracula",
			"Paranormal Activity",
			"Saw",
			"The Ring",
			"Insidious",
			"Freddy vs. Jason",
			"Psycho Beach Party",

			"Food Stamped",
			"Climate of Change",
			"Brexit: The Movie",
			"Citizenfour",
			"Before Flying Back to Earth",
			"The Beatles: Eight Days a Week - The Touring Years",
			"A Beautiful Planet",
			"Life, Animated",
			"Inside Job",
			"Fahrenheit 9/11",
		]

		if verbose:
			print('')
			for expected_genre, movies_titles in sorted(test_movies.items()):
				print('---' + expected_genre + '---')
				for movie_title in movies_titles:
					movie = dm.search_movie(title=movie_title)
					if movie is not None:
						try:
							preds = saved_model.predict(movie)
							print(format_preds(movie, saved_model.genres, preds))
						except:
							pass
					else:
						print(movie_title + ' not found')
						print('')

if __name__ == '__main__':
	main()


























