import os
import numpy as np 
import matplotlib.pyplot as plt 
import h5py

from functions import *
from visualize_pointset import *

def solve(data, connect_from, connect_to, save_plot_dir = '../results/hand', data_dir = None):

	# Shape of Data is Nxnxd. N images. n points per each image pointset. d dimensions of the points.

	os.makedirs(save_plot_dir, exist_ok=True)
	N = data.shape[0]
	colors = np.random.rand(N,3)
	
	# -------------------------- Part A --------------------------- #
	for i in range(N):
		plt.plot(data[i,:,0], data[i,:,1], 'o', color=colors[i])
	plt.title('Plot of all initital pointsets')
	plt.savefig(os.path.join(save_plot_dir, 'initial-all-data-scatter.png'))

	plt.clf()

	# ------------------------------------------------------------- #

	# -------------------------- Part B --------------------------- #
	mean, z_aligned = compute_mean(data)
	for i in range(40):
		plt.plot(z_aligned[i,:,0], z_aligned[i,:,1], 'o', alpha=0.4)
	plot_pointset_with_connections(mean[0,:,0], mean[0,:,1], connect_from, connect_to)
	# # plt.plot(mean[0,:,0], mean[0,:,1], 'ko-')
	# # plt.plot([mean[0,-1,0], mean[0,0,0]], [mean[0,-1,1], mean[0,0,1]], 'ko-')

	plt.title('Aligned pointsets')
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


	def get_modes_of_variation(i, scale=3):

		var_plus = mean + scale*np.sqrt(np.real(eig_values[i]))*np.real(eig_vecs[:,i]).reshape(mean.shape)
		var_minus = mean - scale*np.sqrt(np.real(eig_values[i]))*np.real(eig_vecs[:,i]).reshape(mean.shape)
		return var_plus, var_minus

	for i in range(40):
		plt.plot(z_aligned[i,:,0], z_aligned[i,:,1], 'o', alpha=0.15)

	var_1_plus, var_1_minus = get_modes_of_variation(0)

	plot_pointset_with_connections(mean[0,:,0], mean[0,:,1], connect_from, connect_to, label='Mean')
	plot_pointset_with_connections(var_1_plus[0,:,0], var_1_plus[0,:,1], connect_from, connect_to, label= 'Mean + 3 S.D', color='red')
	plot_pointset_with_connections(var_1_minus[0,:,0], var_1_minus[0,:,1], connect_from, connect_to, label='Mean - 3 S.D',color='blue')

	plt.title('1st Mode of Variation with all the aligned pointsets')
	plt.legend()
	plt.savefig(os.path.join(save_plot_dir, 'mean-and-first-mode.png'))
	plt.clf()

	var_2_plus, var_2_minus = get_modes_of_variation(1, scale=5)

	for i in range(40):
		plt.plot(z_aligned[i,:,0], z_aligned[i,:,1], 'o', alpha=0.15)

	plot_pointset_with_connections(mean[0,:,0], mean[0,:,1], connect_from, connect_to, label='Mean')
	plot_pointset_with_connections(var_2_plus[0,:,0], var_2_plus[0,:,1], connect_from, connect_to, label= 'Mean + 5 S.D', color='red')
	plot_pointset_with_connections(var_2_minus[0,:,0], var_2_minus[0,:,1], connect_from, connect_to, label='Mean - 5 S.D',color='blue')

	plt.title('2nd Mode of Variation with all the aligned pointsets')
	plt.legend()
	plt.savefig(os.path.join(save_plot_dir, 'mean-and-second-mode.png'))
	plt.clf()

	var_3_plus, var_3_minus = get_modes_of_variation(2, scale=7)

	for i in range(40):
		plt.plot(z_aligned[i,:,0], z_aligned[i,:,1], 'o', alpha=0.15)
	
	plot_pointset_with_connections(mean[0,:,0], mean[0,:,1], connect_from, connect_to, label='Mean')
	plot_pointset_with_connections(var_3_plus[0,:,0], var_3_plus[0,:,1], connect_from, connect_to, label= 'Mean + 7 S.D', color='red')
	plot_pointset_with_connections(var_3_minus[0,:,0], var_3_minus[0,:,1], connect_from, connect_to, label='Mean - 7 S.D',color='blue')

	plt.title('3rd Mode of Variation with all the aligned pointsets')
	plt.legend()
	plt.savefig(os.path.join(save_plot_dir, 'mean-and-third-mode.png'))
	plt.clf()

	# ------------------------------------------------------------- #

	# -------------------------- Part E --------------------------- #
	
	z_closest_mean, index_mean = get_closest_pointset(data, mean)
	z_closest_var_1_plus, index_mean_plus = get_closest_pointset(data, var_1_plus)
	z_closest_var_1_minus, index_mean_minus = get_closest_pointset(data, var_1_minus)

	# from image_piecewise_affine.warp import PiecewiseAffineTransform
	# dst_point = get_coordinates(os.path.join(data_dir, sorted(os.listdir(data_dir))[index_mean] ))
	# dst_img = plt.imread(os.path.join(data_dir, sorted(os.listdir(data_dir))[index_mean] ))

	# src_point = get_coordinates(os.path.join(data_dir, sorted(os.listdir(data_dir))[index_mean_plus] ))
	# src_img = plt.imread(os.path.join(data_dir, sorted(os.listdir(data_dir))[index_mean_plus] ))

	# src_pts, dst_pts = [], []
	# for x,y in zip(src_point[0]*600, src_point[1]*800):
	# 	src_pts.append((x,y))
	# for x,y in zip(dst_point[0]*600, dst_point[1]*800):
	# 	dst_pts.append((x,y))

	print(os.path.join(data_dir, sorted(os.listdir(data_dir))[index_mean] ))
	print(os.path.join(data_dir, sorted(os.listdir(data_dir))[index_mean_plus] ))
	print(os.path.join(data_dir, sorted(os.listdir(data_dir))[index_mean_minus] ))

	visualize_checkpoints(os.path.join(data_dir, sorted(os.listdir(data_dir))[index_mean] ), show=False)
	plt.title('Image closest to the the mean shape')
	plt.savefig(os.path.join(save_plot_dir, 'closest_mean.png'))
	plt.clf()

	visualize_checkpoints(os.path.join(data_dir, sorted(os.listdir(data_dir))[index_mean_plus] ), show=False)
	plt.title('Image closest to Mean shape +3 S.D along the top mode of variation')
	plt.savefig(os.path.join(save_plot_dir, 'closest_var_plus.png'))
	plt.clf()

	visualize_checkpoints(os.path.join(data_dir, sorted(os.listdir(data_dir))[index_mean_minus] ), show=False)
	plt.title('Image closest to Mean shape -3 S.D along the top mode of variation')
	plt.savefig(os.path.join(save_plot_dir, 'closest_var_minus.png'))
	plt.clf()

	return eig_vecs, eig_values, mean




