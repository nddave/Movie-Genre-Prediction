
'''
Project: Predicting movie genres from movie posters
Course: COMPSCI 682 Neural Networks: A Modern Introduction

File: data_download.py
Description: Downloads all movie posters and reshapes them into differnt sizes.
Author: Nirman Dave
Adapted from: Vleminckx, Benoit. dnn-movie-posters. (2018). Github repository. https://github.com/benckx/dnn-movie-posters.
'''

import os
import sys
import data_manage as dm

def download(min_year, ratios, images_dir, original_images_dir):
	'''
	Takes in min_year, list of image shrink ratios and image directory
	to sort movie posters by year of origin, download them and save
	to image directory in various different sizes.
	'''	
	if os.path.isdir(original_images_dir) == False:
		os.makedirs(original_images_dir)

	dm.download_posters(min_year=min_year)

	for r in ratios:
		path = images_dir + str(r)
		if os.path.isdir(path) == False:
			os.makedirs(path)
			command = 'mogrify -path "' + path + '/" -resize ' + str(r) + '% "' + original_images_dir + '*.jpg"'
			print(command)
			os.system(command)

	return True

def main():
	'''
	Runs the download function over required params.
	'''
	min_year = 1997
	resizes = [30, 40, 50, 60, 70]
	images_dir = 'data/images/'
	original_images_dir = 'data/images/100/'
	status = download(min_year, resizes, images_dir, original_images_dir)
	return status

if __name__ == '__main__':
	main()
