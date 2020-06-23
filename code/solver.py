import os
import numpy as np 
import matplotlib.pyplot as plt 
import h5py

from functions import *
from visualize_pointset import *

def solve(data, save_plot_dir = '../results/hand', data_dir = None):

	# Shape of Data is Nxnxd. N images. n points per each image pointset. d dimensions of the points.

	os.makedirs(save_plot_dir, exist_ok=True)
	N = data.shape[0]
	colors = np.random.rand(N,3)
	
	# -------------------------- Part A --------------------------- #
	for i in range(N):
		plt.plot(data[i,:,0], data[i,:,1], 'o', color=colors[i])
	plt.title('Plot of all initital pointsets')
	plt.savefig(os.path.join(save_plot_dir, 'initial-all-data-scatter.png'))

	for i in range(N):
		plt.plot(data[i,:,0], data[i,:,1], color=colors[i])
		plt.plot([data[i,0,0], data[i,-1,0]], [data[i,0,1], data[i,-1,1]], color=colors[i])
	plt.savefig(os.path.join(save_plot_dir, 'initial-all-data-polyline.png'))
	plt.clf()

	# ------------------------------------------------------------- #

	# -------------------------- Part B --------------------------- #
	mean, z_aligned = compute_mean(data)
	for i in range(40):
		plt.plot(z_aligned[i,:,0], z_aligned[i,:,1], 'o', alpha=0.4)
	plt.plot(mean[0,:,0], mean[0,:,1], 'ko-')
	plt.plot([mean[0,-1,0], mean[0,0,0]], [mean[0,-1,1], mean[0,0,1]], 'ko-')

	plt.title('Computed shape mean (in black), together with all the aligned pointsets')
	plt.savefig(os.path.join(save_plot_dir, 'mean-and-aligned-data.png'))
	plt.clf()
	# ------------------------------------------------------------- #

	# -------------------------- Part C --------------------------- #

	cov_matrix = compute_covariance_matrix(z_aligned, mean) # ndxnd matrix
	eig_values, eig_vecs = np.linalg.eig(cov_matrix)

	idx = eig_values.argsort()[::-1]
	eig_values = eig_values[idx]
	eig_vecs = eig_vecs[:,idx]

	plt.plot(np.real(eig_values[::-1]))
	plt.title('Eigenvalues (in y axis) plot (sorted in ascending order)')
	plt.savefig(os.path.join(save_plot_dir, 'eigen-values.png'))
	plt.clf()

	# -------------------------- Part D --------------------------- #

	def get_modes_of_variation(i):

		var_plus = mean + 3*np.sqrt(np.real(eig_values[i]))*np.real(eig_vecs[:,i]).reshape(mean.shape)
		var_minus = mean - 3*np.sqrt(np.real(eig_values[i]))*np.real(eig_vecs[:,i]).reshape(mean.shape)
		return var_plus, var_minus

	for i in range(40):
		plt.plot(z_aligned[i,:,0], z_aligned[i,:,1], 'o', alpha=0.15)
	plt.plot(mean[0,:,0], mean[0,:,1], 'ko-', label='Mean')
	plt.plot([mean[0,-1,0], mean[0,0,0]], [mean[0,-1,1], mean[0,0,1]], 'ko-')

	var_1_plus, var_1_minus = get_modes_of_variation(0)

	plt.plot(var_1_plus[0,:,0], var_1_plus[0,:,1], 'ro-', label='Mean + 3 S.D')
	plt.plot([var_1_plus[0,-1,0], var_1_plus[0,0,0]], [var_1_plus[0,-1,1], var_1_plus[0,0,1]], 'ro-')

	plt.plot(var_1_minus[0,:,0], var_1_minus[0,:,1], 'bo-', label='Mean - 3 S.D')
	plt.plot([var_1_minus[0,-1,0], var_1_minus[0,0,0]], [var_1_minus[0,-1,1], var_1_minus[0,0,1]], 'bo-')

	plt.title('1st Mode of Variation with all the aligned pointsets')
	plt.legend()
	plt.savefig(os.path.join(save_plot_dir, 'mean-and-first-mode.png'))
	plt.clf()

	var_2_plus, var_2_minus = get_modes_of_variation(1)

	for i in range(40):
		plt.plot(z_aligned[i,:,0], z_aligned[i,:,1], 'o', alpha=0.15)
	plt.plot(mean[0,:,0], mean[0,:,1], 'ko-', label='Mean')
	plt.plot([mean[0,-1,0], mean[0,0,0]], [mean[0,-1,1], mean[0,0,1]], 'ko-')

	plt.plot(var_2_plus[0,:,0], var_2_plus[0,:,1], 'ro-', label='Mean + 3 S.D')
	plt.plot([var_2_plus[0,-1,0], var_2_plus[0,0,0]], [var_2_plus[0,-1,1], var_2_plus[0,0,1]], 'ro-')

	plt.plot(var_2_minus[0,:,0], var_2_minus[0,:,1], 'bo-', label='Mean - 3 S.D')
	plt.plot([var_2_minus[0,-1,0], var_2_minus[0,0,0]], [var_2_minus[0,-1,1], var_2_minus[0,0,1]], 'bo-')

	plt.title('2nd Mode of Variation with all the aligned pointsets')
	plt.legend()
	plt.savefig(os.path.join(save_plot_dir, 'mean-and-second-mode.png'))
	plt.clf()

	var_3_plus, var_3_minus = get_modes_of_variation(2)

	for i in range(40):
		plt.plot(z_aligned[i,:,0], z_aligned[i,:,1], 'o', alpha=0.15)
	plt.plot(mean[0,:,0], mean[0,:,1], 'ko-', label='Mean')
	plt.plot([mean[0,-1,0], mean[0,0,0]], [mean[0,-1,1], mean[0,0,1]], 'ko-')

	plt.plot(var_3_plus[0,:,0], var_3_plus[0,:,1], 'ro-', label='Mean + 3 S.D')
	plt.plot([var_3_plus[0,-1,0], var_3_plus[0,0,0]], [var_3_plus[0,-1,1], var_3_plus[0,0,1]], 'ro-')
	plt.plot(var_3_minus[0,:,0], var_3_minus[0,:,1], 'bo-', label='Mean - 3 S.D')
	plt.plot([var_3_minus[0,-1,0], var_3_minus[0,0,0]], [var_3_minus[0,-1,1], var_3_minus[0,0,1]], 'bo-')

	plt.title('3rd Mode of Variation with all the aligned pointsets')
	plt.legend()
	plt.savefig(os.path.join(save_plot_dir, 'mean-and-third-mode.png'))
	plt.clf()

	# ------------------------------------------------------------- #

	# -------------------------- Part E --------------------------- #
	

	z_closest_mean, index_mean = get_closest_pointset(data, mean)
	z_closest_var_1_plus, index_mean_plus = get_closest_pointset(data, var_1_plus)
	z_closest_var_1_minus, index_mean_minus = get_closest_pointset(data, var_1_minus)
	# import pdb; pdb.set_trace()

	if data_dir is not None:

		visualize_checkpoints(np.expand_dims(data[index_mean], 0), os.path.join(data_dir, sorted(os.listdir(data_dir))[index_mean]))
		plt.title('Image closest to the the mean shape')
		plt.savefig(os.path.join(save_plot_dir, 'closest_mean.png'))
		plt.clf()

		visualize_checkpoints(np.expand_dims(data[index_mean_plus], 0), os.path.join(data_dir, sorted(os.listdir(data_dir))[index_mean_plus]))
		plt.title('Image closest to Mean shape +3 S.D along the top mode of variation')
		plt.savefig(os.path.join(save_plot_dir, 'closest_var_plus.png'))
		plt.clf()

		visualize_checkpoints(np.expand_dims(data[index_mean_minus], 0), os.path.join(data_dir, sorted(os.listdir(data_dir))[index_mean_minus]))
		plt.title('Image closest to Mean shape -3 S.D along the top mode of variation')
		plt.savefig(os.path.join(save_plot_dir, 'closest_var_minus.png'))
		plt.clf()

	else: # Hand Datset
		plt.plot(data[index_mean, :, 0],data[index_mean, :, 1], 'ro-' )
		plt.plot([data[index_mean, -1, 0], data[index_mean, 0, 0]],[data[index_mean, -1, 1], data[index_mean, 0, 1]], 'ro-' )
		plt.title('Image closest to the the mean shape')
		plt.savefig(os.path.join(save_plot_dir, 'closest_mean.png'))
		plt.clf()

		plt.plot(data[index_mean_plus, :, 0],data[index_mean_plus, :, 1], 'ro-' )
		plt.plot([data[index_mean_plus, -1, 0], data[index_mean_plus, 0, 0]],[data[index_mean_plus, -1, 1], data[index_mean_plus, 0, 1]], 'ro-' )
		plt.title('Image closest to Mean shape +3 S.D along the top mode of variation')
		plt.savefig(os.path.join(save_plot_dir, 'closest_var_plus.png'))
		plt.clf()

		plt.plot(data[index_mean_minus, :, 0],data[index_mean_minus, :, 1], 'ro-' )
		plt.plot([data[index_mean_minus, -1, 0], data[index_mean_minus, 0, 0]],[data[index_mean_minus, -1, 1], data[index_mean_minus, 0, 1]], 'ro-' )
		plt.title('Image closest to Mean shape -3 S.D along the top mode of variation')
		plt.savefig(os.path.join(save_plot_dir, 'closest_var_minus.png'))
		plt.clf()







