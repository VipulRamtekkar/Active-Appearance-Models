from visualize_pointset import plot_pointset_with_connections
from shape_utils import compute_preshape_space, compute_optimal_rotation

from scipy.optimize import leastsq

from texture_utils import *

import os
import numpy as np 
import matplotlib.pyplot as plt
import cv2

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

def fit_shape(test_pointset_data, mean, cov_matrix, eig_values, eig_vecs, connect_from, connect_to, save_plot_dir):
	# Largest t eigenvalues

	t = findt(eig_values, 0.9999)
	t = t + 1

	s0 = mean[0] 
	s = compute_preshape_space(test_pointset_data)

	for i in range(10):
		R = compute_optimal_rotation(s[i],s0)
		s[i] = np.matmul(R,s[i])

	phi = eig_vecs[:t]
	phi = np.transpose(phi)


	for i in range(10):

		y = s[i] - s0 
		y = np.array(y)
		y = np.reshape(y,(146,1))
		

		b = np.linalg.lstsq(phi,y)

		sfit = s0 + np.reshape(np.matmul(phi,b[0]), (73,2))
		sorg = s[i]

		plot_pointset_with_connections(sorg[:,0], sorg[:,1], connect_from, connect_to, label='original annotation', color='red')
		plot_pointset_with_connections(sfit[:,0], sfit[:,1], connect_from, connect_to, label='Shape model fit', color='blue')
		plt.title("Comparing fit against annotation")
		plt.legend()
		plt.savefig(os.path.join(save_plot_dir, 'shape_model_fit_{}.png'.format(i)) )
		plt.clf()


def fit_texture(test_texture_data, mean, cov_matrix, eig_values, eig_vecs, connect_from, connect_to, save_plot_dir):

	t = findt(eig_values, 0.98)

	s0 = mean
	phi = eig_vecs[:t]
	phi = np.transpose(phi)

	
	beta = np.mean(test_texture_data, axis=(1,2), keepdims=True)
	alpha = np.mean(test_texture_data*mean, axis=(1,2), keepdims=True)

	test_texture_data = (test_texture_data-beta)/alpha
	s = test_texture_data

	for i in range(10):

		y = s[i] - s0
		y = np.reshape(y,(3480,1))

		b = np.linalg.lstsq(phi,y, rcond = None)

		sfit = s0 + np.real(np.reshape(np.matmul(phi,b[0]), (1,58,60)))
		sorg = np.reshape(s[i],(1,58,60))

		f, axarr = plt.subplots(1,2) 
		axarr[0].imshow(cv2.resize(sorg[0], (sorg[0].shape[0]*4,sorg[0].shape[1]*4 )), cmap='gray')
		axarr[0].title.set_text('data')
		axarr[0].axis('off')

		axarr[1].imshow(cv2.resize(sfit[0], (sfit[0].shape[0]*4,sfit[0].shape[1]*4 )), cmap='gray')
		axarr[1].title.set_text('fit')
		axarr[1].axis('off')

		plt.subplots_adjust(wspace=None, hspace=None)
		plt.suptitle('Comparison between fit and data')
		plt.savefig(os.path.join(save_plot_dir, 'texture_model_fit_{}.png'.format(i)), bbox_inches='tight')
		plt.clf()