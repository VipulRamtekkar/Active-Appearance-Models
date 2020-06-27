from shape_model import shape_model
from texture_model import texture_model
from visualize_pointset import get_coordinates
from normalize_shape import normalize_shape
from fit_test_sample import fit_shape, fit_texture

import os
import matplotlib.pyplot as plt
import numpy as np 
from skimage.color import rgb2gray
from skimage.io import imsave, imread
import cv2

def prepare_data(data_dir, precomputed_shape_normalized_texture, save_texture_data_dir = '../data/shape_normalized_images'):

	pointset_data = []
	shape_normalized_texture_data, shape_normalized_texture_data_full_res = [], []
	data_list = []

	mean_shape_img_path = os.path.join(data_dir, '08_01.jpg') # From shape modeling (Hard-coded)

	for i, img_path in enumerate(sorted(os.listdir(data_dir))):
		if not '.jpg' in img_path:
			continue

		individual_img_path = os.path.join(data_dir, img_path)
		x_coordinates, y_coordinates, connect_from, connect_to = get_coordinates(individual_img_path)
		x_coordinates = 600*np.expand_dims(x_coordinates, 1); y_coordinates = 800*(1-np.expand_dims(y_coordinates, 1))
		
		coordinates = np.expand_dims(np.concatenate((x_coordinates, y_coordinates), axis=1), 0)
		pointset_data.append(coordinates)

		# Align individual image to shape normalized coordinates

		if not precomputed_shape_normalized_texture:
			shape_normalized_texture = normalize_shape(individual_img_path, mean_shape_img_path)
			imsave(os.path.join(save_texture_data_dir, img_path), shape_normalized_texture)
		else:
			shape_normalized_texture = plt.imread(os.path.join(save_texture_data_dir, img_path))

		shape_normalized_texture_resize = cv2.resize(shape_normalized_texture, 
			(shape_normalized_texture.shape[0]//4, shape_normalized_texture.shape[1]//4))
		shape_normalized_texture_data.append(np.expand_dims(rgb2gray(shape_normalized_texture_resize), 0))
		shape_normalized_texture_data_full_res.append(np.expand_dims(shape_normalized_texture,0))

		data_list.append(individual_img_path)

	pointset_data = np.concatenate(pointset_data, axis=0)
	shape_normalized_texture_data = np.concatenate(shape_normalized_texture_data, axis=0)
	shape_normalized_texture_data_full_res = np.concatenate(shape_normalized_texture_data_full_res, axis=0)

	return pointset_data, shape_normalized_texture_data, shape_normalized_texture_data_full_res, \
	 connect_from, connect_to, np.array(data_list)


def main(data_dir = '../data/imm3943/IMM-Frontal Face DB SMALL/', precomputed_shape_normalized_texture=False):

	# Prepare data
	pointset_data, shape_normalized_texture_data, shape_normalized_texture_data_full_res, \
	connect_from, connect_to, data_list = prepare_data(data_dir, precomputed_shape_normalized_texture)

	N = pointset_data.shape[0] # Total Data size
	Ntest = 10

	# Split Data into Train and Test Set
	test_indices = np.random.choice(range(N), size=(Ntest,), replace=False)
	train_indices = list(set(range(N)) - set(test_indices))

	# Train Shape Model
	print('----------------------------------Train Shape Model----------------------------------')
	shape_model(pointset_data[train_indices], connect_from, connect_to, save_plot_dir='../results/faces/shapes', 
		data_list=data_list[train_indices])

	# Test fitting ability of shape model
	print('----------------------------------Test Shape Model Fits----------------------------------')
	shape_mean = np.load('../results/shape_mean.npy')
	shape_cov_matrix = np.load('../results/shape_cov.npy')
	shape_eig_values = np.load('../results/shape_eigvalues.npy')
	shape_eig_vecs = np.load('../results/shape_eigvecs.npy')

	fit_shape(pointset_data[test_indices],shape_mean, shape_cov_matrix, shape_eig_values, shape_eig_vecs, 
		connect_from, connect_to, save_plot_dir='../results/faces/shapes')

	# Train Texture Model
	print('----------------------------------Train Texture Model----------------------------------')
	texture_model(shape_normalized_texture_data_full_res[train_indices], shape_normalized_texture_data[train_indices], connect_from, 
		connect_to, save_plot_dir = '../results/faces/texture')

	shape_mean = np.load('../results/shape_mean.npy')
	shape_cov_matrix = np.load('../results/shape_cov.npy')
	shape_eig_values = np.load('../results/shape_eigvalues.npy')
	shape_eig_vecs = np.load('../results/shape_eigvecs.npy')

	print('----------------------------------Test Texture Model Fits----------------------------------')
	fit_shape(pointset_data[test_indices],shape_mean, shape_cov_matrix, shape_eig_values, shape_eig_vecs, 
		connect_from, connect_to, save_plot_dir='../results/faces/shapes')

	print('----------------------------------Obtain combined modes of variations----------------------------------')


if __name__ == '__main__':
	np.random.seed(14)
	main(precomputed_shape_normalized_texture=True)



