from shape_model import shape_model
from texture_model import texture_model
from visualize_pointset import get_coordinates
from functions import *

import os
import numpy as np 

def prepare_data(data_dir, precomputed_shape_normalized_texture):

	pointset_data = []
	shape_normalized_texture_data, shape_normalized_texture_data_full_res = [], []

	mean_shape_img_path = os.path.join(data_dir, '08_01.jpg') # From shape modeling (Hard coded)

	for i, img_path in enumerate(sorted(os.listdir(data_dir))):
		if not '.jpg' in img_path:
			continue
		complete_path = os.path.join(data_dir, img_path)
		x_coordinates, y_coordinates, connect_from, connect_to = get_coordinates(complete_path)
		x_coordinates = 600*np.expand_dims(x_coordinates, 1); y_coordinates = 800*(1-np.expand_dims(y_coordinates, 1))
		
		coordinates = np.expand_dims(np.concatenate((x_coordinates, y_coordinates), axis=1), 0)
		pointset_data.append(coordinates)

		# Align individual image to shape normalized coordinates
		individual_shape_img_path = os.path.join(data_dir, img_path)

		if not precomputed_shape_normalized_texture:
			shape_normalized_texture = normalize_shape(individual_shape_img_path, mean_shape_img_path)
			imsave(os.path.join(save_data_dir, img_path), shape_normalized_texture)
		else:
			shape_normalized_texture = plt.imread(os.path.join(save_data_dir, img_path))
			shape_normalized_texture_resize = cv2.resize(shape_normalized_texture, 
				(shape_normalized_texture.shape[0]//4, shape_normalized_texture.shape[1]//4))

		shape_normalized_texture_data.append(np.expand_dims(rgb2gray(shape_normalized_texture_resize), 0))
		shape_normalized_texture_data_full_res.append(np.expand_dims(shape_normalized_texture,0))

	pointset_data = np.concatenate(pointset_data, axis=0)
	shape_normalized_texture_data = np.concatenate(shape_normalized_texture_data, axis=0)
	shape_normalized_texture_data_full_res = np.concatenate(shape_normalized_texture_data_full_res, axis=0)

	return pointset_data, shape_normalized_texture_data, shape_normalized_texture_data_full_res


def main(data_dir = '../data/imm3943/IMM-Frontal Face DB SMALL/', precomputed_shape_normalized_texture=False):
	
	N = len(os.listdir(data_dir))-3

	# Prepare data
	pointset_data, shape_normalized_texture_data, shape_normalized_texture_data_full_res = \
	prepare_data(data_dir, precomputed_shape_normalized_texture)

	shape_model(pointset_data, connect_from, connect_to, save_plot_dir='../results/faces', data_dir=data_dir)

	shape_cov_matrix = np.load('../results/shape_cov.npy')
	shape_eig_values = np.save('../results/shape_eigvalues.npy')
	shape_eig_vecs = np.save('../results/shape_eigvecs.npy')
	# Largest t eigenvalues

	total = sum(shape_eig_values)

	total = 0.9999 * total

	p = 0
	t = 0 

	for i in range(len(eig_values)):
		p = p + eig_values[i]

		if p >= total:
			t = i
			break
	t = t + 1
	s0 = mean[0] 
	s = compute_preshape_space(pointset_data[:5])
	for i in range(5):
		R = compute_optimal_rotation(s[i],s0)
		s[i] = np.matmul(R,s[i])

	for i in range(5):

		phi = eig_vecs[:t]
		y = s[i] - s0 
		y = np.array(y)
		y = np.reshape(y,(146,1))
		phi = np.transpose(phi)

		b = np.linalg.lstsq(phi,y)

		print (b[0])


if __name__ == '__main__':
	main(precomputed_shape_normalized_texture=True)




