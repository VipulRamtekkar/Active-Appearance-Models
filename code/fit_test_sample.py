from visualize_pointset import plot_pointset_with_connections
from shape_utils import compute_preshape_space, compute_optimal_rotation

from scipy.optimize import leastsq, fsolve

from texture_utils import *

import os
import numpy as np 
import matplotlib.pyplot as plt
import cv2

# t refers to the number of eigenvalues that are preserved. We preserve the top t out of n eigenvalues
# that roughly sum up to a certain percent of sum of all n eigenvalues.
def findt(eig_values, percent):

	total = sum(eig_values)
	total = percent * total
	p = 0
	t = 0 
	for i in range(len(eig_values)):
		p = p + eig_values[i]
		if p >= total:
			t = i
			break
	return t

def get_rrmse(A,B):
	rrmse = np.sqrt(np.sum((A-B)**2)/np.sum(A**2))
	return rrmse

def fit_shape(test_pointset_data, mean, cov_matrix, eig_values, eig_vecs, connect_from, connect_to, save_dir):
	# Largest t eigenvalues

	eig_values = np.real(eig_values)
	eig_vecs = np.real(eig_vecs)

	t = findt(eig_values, 0.9999)
	t = t + 1

	s0 = mean[0] 
	s = compute_preshape_space(test_pointset_data)

	for i in range(10):
		R = compute_optimal_rotation(s[i],s0)
		s[i] = np.matmul(R,s[i])

	phi = eig_vecs[:t]
	phi = np.transpose(phi)

	rrmse_values = []
	test_set_params = np.zeros((test_pointset_data.shape[0], t))

	for i in range(10):

		y = s[i] - s0 
		y = np.array(y)
		y = np.reshape(y,(146,1))
		

		b = np.linalg.lstsq(phi,y)

		sfit = s0 + np.reshape(np.matmul(phi,b[0]), (73,2))
		sorg = s[i]

		rrmse = get_rrmse(sorg, sfit)
		rrmse_values.append(rrmse)
		test_set_params[i] = b[0].squeeze()

		plot_pointset_with_connections(sorg[:,0], sorg[:,1], connect_from, connect_to, label='original annotation', color='red')
		plot_pointset_with_connections(sfit[:,0], sfit[:,1], connect_from, connect_to, label='Shape model fit', color='blue')
		plt.title("Comparing fit against annotation")
		plt.legend()
		plt.savefig(os.path.join(save_dir, 'shape_model_fit_{}.png'.format(i)))
		plt.clf()

	print('Average Reconstruction Error on Test Set: Mean: {} Std-Dev: {}'.format(np.mean(rrmse_values), np.std(rrmse_values)))
	np.save(os.path.join(save_dir, 'shape_model_fit_param.npy'), test_set_params)


def fit_texture(test_texture_data, mean, cov_matrix, eig_values, eig_vecs, connect_from, connect_to, save_dir):

	t = findt(eig_values, 0.985)
	eig_values = np.real(eig_values[0:t]) 
	eig_vecs = np.real(eig_vecs[:, 0:t])

	beta = np.mean(test_texture_data, axis=(1,2), keepdims=True)
	alpha = np.mean(test_texture_data*mean, axis=(1,2), keepdims=True)
	
	test_texture_data = (test_texture_data-beta)/alpha

	rrmse_values = []
	test_set_params = np.zeros((test_texture_data.shape[0], t))

	for i in range(test_texture_data.shape[0]):

		model_fit_coeff = np.dot(np.linalg.pinv(eig_vecs), test_texture_data[i].reshape(mean.size,1) - mean.reshape(mean.size,1) )
		texture_recon = (mean.reshape(mean.size,1) + np.dot(eig_vecs, model_fit_coeff)).reshape(mean.shape).squeeze(0)

		rrmse = get_rrmse(test_texture_data[i], texture_recon) 
		rrmse_values.append(rrmse)
		test_set_params[i] = model_fit_coeff.squeeze()

		f, axarr = plt.subplots(1,2) 
		axarr[0].imshow(cv2.resize(test_texture_data[i], (test_texture_data[i].shape[0]*4,test_texture_data[i].shape[1]*4 )), cmap='gray')
		axarr[0].title.set_text('Test Image')
		axarr[0].axis('off')

		axarr[1].imshow(cv2.resize(texture_recon, (texture_recon.shape[0]*4,texture_recon.shape[1]*4 )), cmap='gray')
		axarr[1].title.set_text('Model Fit')
		axarr[1].axis('off')

		plt.subplots_adjust(wspace=None, hspace=None)
		plt.suptitle('Analysis of Texture Model fitting: Test Image: {}'.format(i))
		plt.savefig(os.path.join(save_dir, 'texture_model_fit_{}.png'.format(i)), bbox_inches='tight')
		plt.clf()

	print('Average Reconstruction Error on Test Set: Mean: {} Std-Dev: {}'.format(np.mean(rrmse_values), np.std(rrmse_values)))
	np.save(os.path.join(save_dir, 'texture_model_fit_param.npy'), test_set_params)
	




