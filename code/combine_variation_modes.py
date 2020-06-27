from visualize_pointset import get_coordinates
from normalize_shape import apply_shape
from shape_utils import preshape_to_image_space

import os
import numpy as np 
import matplotlib.pyplot as plt
import cv2

def get_individual_variation_mode(eig_values, eig_vecs, mean, i, scale=3):

	var_plus = mean + scale*np.sqrt(np.real(eig_values[i]))*np.real(eig_vecs[:,i]).reshape(mean.shape)
	var_minus = mean - scale*np.sqrt(np.real(eig_values[i]))*np.real(eig_vecs[:,i]).reshape(mean.shape)
	return var_plus, var_minus

def combine_shape_with_normalized_texture(shape_pointset, normalized_texture, shape_mean_x_coord, shape_mean_y_coord):

	'''
	shape_pointset: 1xnxd array in preshape space 
	normalized_texture: 1xHxW in shape-normalized frame (mean shape frame)
	'''

	W,H = normalized_texture.shape[0], normalized_texture.shape[1]
	
	shape_pointset_in_img_space = preshape_to_image_space(shape_pointset, shape_mean_x_coord, shape_mean_y_coord)

	shape_mean_x_coord *= W; shape_mean_y_coord *= H
	shape_pointset_in_img_space[0,:,0] *= W; shape_pointset_in_img_space[0,:,1] *= H

	warped_texture = apply_shape(shape_pointset_in_img_space, normalized_texture, shape_mean_x_coord, shape_mean_y_coord)

	return warped_texture


def get_combine_variation_modes(shape_mean, shape_eig_values, shape_eig_vecs, texture_mean, texture_eig_values, 
	texture_eig_vecs, save_plot_dir='../results/faces/combine_modes'):

	os.makedirs(save_plot_dir, exist_ok=True)

	shape_mean_img_path = os.path.join('../data/imm3943/IMM-Frontal Face DB SMALL/', '08_01.jpg') # From shape modeling (Hard-coded)
	shape_mean_x_coord, shape_mean_y_coord, connect_from, connect_to = get_coordinates(shape_mean_img_path)
	shape_mean_y_coord = 1-shape_mean_y_coord

	shape_var_1_plus, shape_var_1_minus = get_individual_variation_mode(shape_eig_values, shape_eig_vecs, shape_mean, 0, scale=4)
	texture_var_1_plus, texture_var_1_minus = get_individual_variation_mode(texture_eig_values, texture_eig_vecs, texture_mean, 0)

	shape_var_2_plus, shape_var_2_minus = get_individual_variation_mode(shape_eig_values, shape_eig_vecs, shape_mean, 1, scale=10)
	texture_var_2_plus, texture_var_2_minus = get_individual_variation_mode(texture_eig_values, texture_eig_vecs, texture_mean, 1)

	f, axarr = plt.subplots(4,4)
	for s, shape_mode in enumerate([shape_var_1_minus, shape_var_1_plus, shape_var_2_minus, shape_var_2_plus]):		 
		for t, texture_mode in enumerate([texture_var_1_minus, texture_var_1_plus, texture_var_2_minus, texture_var_2_plus]):

			combine_mode = combine_shape_with_normalized_texture(shape_mode, cv2.resize(texture_mode.squeeze(0),
				(texture_var_1_plus.shape[1]*4, texture_var_1_plus.shape[2]*4) ), shape_mean_x_coord, shape_mean_y_coord)
			axarr[s, t].imshow(combine_mode , cmap='gray')
			axarr[s, t].axis('off')

	plt.suptitle('Combine Modes of Variations')
	plt.subplots_adjust(wspace=None, hspace=None)
	plt.savefig(os.path.join(save_plot_dir, 'subplot.png'), bbox_inches='tight')
	plt.clf()

if __name__ == '__main__':

	shape_mean = np.load('../results/shape_mean.npy')
	shape_eig_values = np.load('../results/shape_eigvalues.npy')
	shape_eig_vecs = np.load('../results/shape_eigvecs.npy')

	texture_mean = np.load('../results/texture_mean.npy')
	texture_eig_values = np.load('../results/texture_eigvalues.npy')
	texture_eig_vecs = np.load('../results/texture_eigvecs.npy')

	get_combine_variation_modes(shape_mean, shape_eig_values, shape_eig_vecs, texture_mean, texture_eig_values, texture_eig_vecs)

