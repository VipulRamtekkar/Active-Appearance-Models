import numpy as np 

def compute_preshape_space(z):

	centroid = np.mean(z, axis=1, keepdims=True)

	# Put points on the same hyperplane
	z_prime = z - centroid

	z_prime_norm = np.linalg.norm(z_prime, axis=(1,2), keepdims=True)

	# Put points on the same hypersphere
	z_preshape = z_prime/z_prime_norm 

	return z_preshape

def compute_optimal_rotation(z1, z2):

	X = z1 # (Nx2)
	Y = z2 # (2xN)
	# import pdb; pdb.set_trace()

	U, S, Vt = np.linalg.svd(np.matmul(X, Y.T))

	R = np.matmul(Vt.T, U.T)

	if np.linalg.det(R) == 1:
		return R 
	else:
		M = np.eye(U.shape[0])
		M[-1,-1] = -1
		R = np.matmul(Vt.T, np.matmul(M, U.T))
		return R

def compute_mean(pointsets):

	z = pointsets
	z_mean = np.expand_dims(z[0], axis=0)

	prev_z_mean = z_mean

	while True:

		# For a given Mean, Find optimal transformation parameters
		# import pdb; pdb.set_trace()

		z = compute_preshape_space(z)
		z_mean = compute_preshape_space(z_mean)

		for i in range(z.shape[0]):
			R = compute_optimal_rotation(z[i], z_mean[0])
			z[i] = np.matmul(R, z[i])

		# Find mean for a given theta
		z_mean = np.mean(z, axis=0, keepdims=True)
		z_mean = z_mean/np.linalg.norm(z_mean)

		if np.linalg.norm(prev_z_mean-z_mean) < 0.00001:
			break 

		prev_z_mean = z_mean

	return z_mean, z


def compute_covariance_matrix(Z, mean):
	# mean - 1xnxd
	# z - Nxnxd

	N = Z.shape[0]
	z_vec_dim = Z.shape[1]*Z.shape[2]
	mean = mean.reshape((1,z_vec_dim))
	mean = mean.T
	Z = Z.reshape((N,z_vec_dim))
	Z = Z.T

	# mean - ndx1
	# Z - ndxN

	# cov_matrix = np.matmul(Z-mean, (Z-mean).T)
	cov_matrix = np.cov(Z)

	return cov_matrix

def get_closest_pointset(z, mean):

	z = compute_preshape_space(z)
	mean = compute_preshape_space(mean)

	min_dist = 1000
	min_dist_index = -1

	for i in range(z.shape[0]):

		R = compute_optimal_rotation(z[i], mean[0])
		z_i_align = np.matmul(R, z[i])
		dist = np.linalg.norm(mean-z_i_align)

		if dist < min_dist:
			min_dist = dist
			min_dist_index = i

	return z[min_dist_index], min_dist_index





