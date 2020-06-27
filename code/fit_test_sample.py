from visualize_pointset import plot_pointset_with_connections
from shape_utils import compute_preshape_space, compute_optimal_rotation

import os
import numpy as np 
import matplotlib.pyplot as plt

def fit_shape(test_pointset_data, mean, cov_matrix, eig_values, eig_vecs, connect_from, connect_to, save_plot_dir):
	# Largest t eigenvalues
	total = sum(eig_values)
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
	s = compute_preshape_space(test_pointset_data)

	for i in range(10):
		R = compute_optimal_rotation(s[i],s0)
		s[i] = np.matmul(R,s[i])

	for i in range(10):

		phi = eig_vecs[:t]
		y = s[i] - s0 
		y = np.array(y)
		y = np.reshape(y,(146,1))
		phi = np.transpose(phi)

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
	pass

