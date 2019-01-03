
'''
Project: Predicting movie genres from movie posters
Course: COMPSCI 682 Neural Networks: A Modern Introduction

File: external_test.py
Description: Tests an image using a URL from the internet.
Author: Nirman Dave

Run this file:
$ python3 external_test.py <Movie Title>
'''

import os
import sys
import csv
import random
import urllib.request
from PIL import Image
import data_manage as dm
from os.path import isfile, join
from resizeimage import resizeimage

def log(poster_url, poster_name):
	'''
	Logs the external movie to another CSV file to avoid conflicting entries.
	'''

	# Creates unique movie id by hashing a random int
	fid = str(random.randint(0, 999999))
	fake_id = str(abs(hash(fid)))
	log_file = 'data/MovieGenre_external.csv'

	imdb_id = str(fake_id)
	imdb_url = 'http://www.imdb.com/title/et' + str(imdb_id)
	title = str(poster_name) + ' (2015)'
	imdb_score = '8.2'
	genre = ''
	poster = str(poster_url)

	data = [imdb_id, imdb_url, title, imdb_score, genre, poster]

	with open(log_file,'a') as fd:
		writer = csv.writer(fd)
		writer.writerow(data)
	fd.close()

	return fake_id

def download(poster_url, fake_id):
	'''
	Downloads the external movie poster from the link provided while calling the function.
	'''

	images_dir = 'data/images/'
	original_images_dir = 'data/images/100/'

	save_path = original_images_dir + fake_id + '.jpg'

	response = urllib.request.urlopen(poster_url)
	data = response.read()
	file = open(save_path, 'wb')
	file.write(bytearray(data))
	file.close()

	# resizes the download into different ratios
	ratios = {30: (55, 80), 40: (73, 107), 50: (91, 134), 60: (109, 161), 70: (127, 188)}

	for r in ratios:
		i, j = ratios[r]
		resize_path = images_dir + str(r) + "/" + fake_id + '.jpg'
		with open(save_path, 'r+b') as f:
			with Image.open(f) as image:
				cover = resizeimage.resize_cover(image, [i, j])
				cover.save(resize_path, image.format)

	return True

def test(models_dir, poster_name, gens):
	'''
	Runs test using run_external_test.py file
	'''

	command = "python3 run_external_test.py " + str(models_dir) + " " + str(poster_name) + " " + str(gens)
	os.system(command)

	return True

def main():
	'''
	Sets up all parameters to successfully run an external test
	'''

	models_dir = 'cnn_model_results/models'
	pic_name = str(sys.argv[1])
	pic_url = input('poster url: ')
	genres = 'four'

	fid = log(pic_url, pic_name)
	download(pic_url, fid)
	test(models_dir, pic_name, genres)

	print('FID -->', fid)

	return True

if __name__ == '__main__':
	main()








