from texture_utils import compute_mean_texture
from image_warp import normalize_shape
from visualize_pointset import *

import os
import numpy as np 
import scipy 
from skimage.color import rgb2gray
from skimage.io import imsave, imread
from skimage.transform import resize
import cv2
import matplotlib.pyplot as plt

def texture_model(full_res_data, data, connect_from=None, connect_to=None, save_plot_dir = '../results/faces', data_dir = None):

	'''
	data: (N, W, H): N grayscale images of width W and height H in shape normalized coordinates

	'''
	os.makedirs(save_plot_dir, exist_ok=True)
	N, H, W = data.shape

	# -----------------------Compute and plot mean texture-------------------------- #

	mean_texture, beta, alpha, normalized_data = compute_mean_texture(data)
	plt.imshow(cv2.resize(mean_texture[0], (mean_texture[0].shape[0]*4,mean_texture[0].shape[1]*4 )), cmap='gray')
	plt.title('Mean Average Appearance')
	plt.savefig(os.path.join(save_plot_dir, 'average_normalized_appearace.png'))
	plt.clf()

	np.save('../results/texture_mean.npy', mean_texture)

	# ------------------------Plot eigenvalues-------------------------- #

	g_normalized = normalized_data.reshape(N, -1) # Consistent with the original AAM paper notations
	

	try:
		cov_matrix = np.load('../results/texture_cov.npy')
		eig_values = np.load('../results/texture_eigvalues.npy')
		eig_vecs = np.load('../results/texture_eigvecs.npy')
	except:
		cov_matrix = np.cov(g_normalized.T - mean_texture.reshape(mean_texture.size, 1))
		eig_values, eig_vecs = scipy.linalg.eig(cov_matrix)
		np.save('../results/texture_cov.npy', cov_matrix)
		np.save('../results/texture_eigvalues.npy', eig_values)
		np.save('../results/texture_eigvecs.npy', eig_vecs)
	
	idx = eig_values.argsort()[::-1]
	eig_values = eig_values[idx]
	eig_vecs = eig_vecs[:,idx]

	plt.plot(np.real(eig_values[::-1]))
	plt.title('Texture Analysis: Eigenvalues (in y axis) plot (sorted in ascending order)')
	plt.savefig(os.path.join(save_plot_dir, 'texture-eigen-values.png'))
	plt.clf()

	# ----------------------Plot modes of variations------------------------- #

	def get_modes_of_variation(i, scale=3):

		var_plus = mean_texture + scale*np.sqrt(np.real(eig_values[i]))*np.real(eig_vecs[:,i]).reshape(mean_texture.shape)
		var_minus = mean_texture - scale*np.sqrt(np.real(eig_values[i]))*np.real(eig_vecs[:,i]).reshape(mean_texture.shape)
		return var_plus, var_minus

	var_1_plus, var_1_minus = get_modes_of_variation(0)

	f, axarr = plt.subplots(1,3) 
	axarr[0].imshow(cv2.resize(var_1_minus[0],(mean_texture[0].shape[0]*4,mean_texture[0].shape[1]*4 )) , cmap='gray')
	axarr[0].title.set_text('Mean - 3 S.D')
	axarr[0].axis('off')

	axarr[1].imshow(cv2.resize(mean_texture[0],(mean_texture[0].shape[0]*4,mean_texture[0].shape[1]*4 )) , cmap='gray')
	axarr[1].title.set_text('Mean')
	axarr[1].axis('off')

	axarr[2].imshow(cv2.resize(var_1_plus[0],(mean_texture[0].shape[0]*4,mean_texture[0].shape[1]*4 )) , cmap='gray')
	axarr[2].title.set_text('Mean + 3 S.D')
	axarr[2].axis('off')

	plt.subplots_adjust(wspace=None, hspace=None)
	plt.suptitle('1st mode of variation')
	plt.savefig(os.path.join(save_plot_dir, 'texture_first_mode_variation.png'), bbox_inches='tight')
	plt.clf()

	var_2_plus, var_2_minus = get_modes_of_variation(1)

	f, axarr = plt.subplots(1,3) 
	axarr[0].imshow(cv2.resize(var_2_minus[0], (mean_texture[0].shape[0]*4,mean_texture[0].shape[1]*4 )), cmap='gray')
	axarr[0].title.set_text('Mean - 3 S.D')
	axarr[0].axis('off')

	axarr[1].imshow(cv2.resize(mean_texture[0], (mean_texture[0].shape[0]*4,mean_texture[0].shape[1]*4 )), cmap='gray')
	axarr[1].title.set_text('Mean')
	axarr[1].axis('off')

	axarr[2].imshow(cv2.resize(var_2_plus[0], (mean_texture[0].shape[0]*4,mean_texture[0].shape[1]*4 )), cmap='gray')
	axarr[2].title.set_text('Mean + 3 S.D')
	axarr[2].axis('off')

	plt.subplots_adjust(wspace=None, hspace=None)
	plt.suptitle('2nd mode of variation')
	plt.axis('off')
	plt.savefig(os.path.join(save_plot_dir, 'texture_second_mode_variation.png'), bbox_inches='tight')
	plt.clf()

	var_3_plus, var_3_minus = get_modes_of_variation(2)

	f, axarr = plt.subplots(1,3) 
	axarr[0].imshow(cv2.resize(var_3_minus[0], (mean_texture[0].shape[0]*4,mean_texture[0].shape[1]*4 )), cmap='gray')
	axarr[0].title.set_text('Mean - 3 S.D')
	axarr[0].axis('off')

	axarr[1].imshow(cv2.resize(mean_texture[0], (mean_texture[0].shape[0]*4,mean_texture[0].shape[1]*4 )), cmap='gray')
	axarr[1].title.set_text('Mean')
	axarr[1].axis('off')

	axarr[2].imshow(cv2.resize(var_3_plus[0], (mean_texture[0].shape[0]*4,mean_texture[0].shape[1]*4 )), cmap='gray')
	axarr[2].title.set_text('Mean + 3 S.D')
	axarr[2].axis('off')

	plt.subplots_adjust(wspace=None, hspace=None)
	plt.suptitle('3rd mode of variation')
	plt.axis('off')
	plt.savefig(os.path.join(save_plot_dir, 'texture_third_mode_variation.png'), bbox_inches='tight')
	plt.clf()

	# -----------------------Compare modes of variations------------------------- #

	closest_mean_index = np.argmin(np.sum( (normalized_data-mean_texture), axis=(1,2) ))
	plt.imshow(full_res_data[closest_mean_index])
	plt.title('Image closest to mean texture')
	plt.savefig(os.path.join(save_plot_dir, 'texture_closest_mean.png'))
	plt.clf()

	closest_var_1_plus_index = np.argmin(np.sum( (normalized_data-var_1_plus), axis=(1,2) ))
	plt.imshow(full_res_data[closest_var_1_plus_index])
	plt.title('Image closest to Mean + 3 S.D. (first mode)')
	plt.savefig(os.path.join(save_plot_dir, 'texture_closest_var_1_plus.png'))
	plt.clf()

	closest_var_1_minus_index = np.argmin(np.sum( (normalized_data-var_1_minus), axis=(1,2) ))
	plt.imshow(full_res_data[closest_var_1_minus_index])
	plt.title('Image closest to Mean - 3 S.D. (first mode)')
	plt.savefig(os.path.join(save_plot_dir, 'texture_closest_var_1_minus.png'))
	plt.clf()


def compute(precomputed_shape_normalized_texture=False):

	data_dir = '../data/imm3943/IMM-Frontal Face DB SMALL/'
	save_data_dir = '../data/shape_normalized_images'
	N = len(os.listdir(data_dir))-3

	mean_shape_img_path = os.path.join(data_dir, '08_01.jpg') # From shape modeling

	os.makedirs(save_data_dir, exist_ok=True)

	shape_normalized_texture_data, shape_normalized_texture_data_full_res = [], []

	for i, img_path in enumerate(sorted(os.listdir(data_dir))):
		if not '.jpg' in img_path:
			continue

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

	shape_normalized_texture_data = np.concatenate(shape_normalized_texture_data, axis=0)
	shape_normalized_texture_data_full_res = np.concatenate(shape_normalized_texture_data_full_res, axis=0)
	print(shape_normalized_texture_data.shape)

	texture_model(shape_normalized_texture_data_full_res, shape_normalized_texture_data)

if __name__ == '__main__':
	np.random.seed(1)
	compute(True)


		

	