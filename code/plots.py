
'''
Project: Predicting movie genres from movie posters
Course: COMPSCI 682 Neural Networks: A Modern Introduction

File: plots.py
Description: Plots model histories from pickle files.
Author: Nirman Dave
'''

import pickle
from os import listdir
import matplotlib.pyplot as plt

pkl_dir = 'cnn_model_results/hists/'

def plot_hist(hist_path, h):
	'''
	Takes model history and file name (h), plots the file and saves it with
	file name as .png
	'''
	try:
		file = open(hist_path, 'rb')
		model = pickle.load(file)
		file.close()

		train_acc = model['acc']
		train_loss = model['loss']
		val_acc = model['val_acc']
		val_loss = model['val_loss']

		info = h.split('_')
		ratio = info[4]
		lr = info[7].split('.pkl')[0]
		ver = info[6]

		title = str(ratio) + ' ' + str(lr)
		save_title = str(ratio) + ' ' + str(lr) + ' ' + str(ver)

		# plot accuracy
		print('')
		print('plotting ACC for', str(h))
		fig_name = pkl_dir + 'plots/acc/' + save_title + '.png'
		plt.plot(train_acc)
		plt.plot(val_acc)
		plt.title(title)
		plt.xlabel('epoch')
		plt.ylabel('accuracy')
		plt.legend(['train', 'val'], loc='upper left')
		plt.savefig(fig_name)
		plt.close()

		# plot loss
		print('plotting LOSS for', str(h))
		fig_name = pkl_dir + 'plots/loss/' + save_title + '.png'
		plt.plot(train_loss)
		plt.plot(val_loss)
		plt.title(title)
		plt.xlabel('epoch')
		plt.ylabel('loss')
		plt.legend(['train', 'val'], loc='upper left')
		plt.savefig(fig_name)
		plt.close()
		print('')
	except:
		pass

	return True

def main():
	'''
	Run plot on all pickle files in the pkl_dir
	'''
	all_hists = listdir(pkl_dir)
	for hist in all_hists:
		h = pkl_dir + str(hist)
		plot_hist(h, hist)
	return True

if __name__ == '__main__':
	main()






