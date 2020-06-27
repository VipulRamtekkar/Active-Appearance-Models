import numpy as np 
import matplotlib.pyplot as plt

def compute_mean_texture(data):

	N = data.shape[0]
	H = data.shape[1]
	W = data.shape[2]

	# Parameters for lighting normalization
	alpha = np.zeros((N,1)) # Scaling 
	beta = np.zeros((N,1)) # Offset 

	# initialize one of the images as the mean to begin the optimization process
	mean_texture = np.expand_dims(data[np.random.randint(N)], 0)

	prev_texture = mean_texture

	counter = 0
	while True:

		beta = np.mean(data, axis=(1,2), keepdims=True)
		alpha = np.mean(data*mean_texture, axis=(1,2), keepdims=True)

		normalized_data = (data-beta)/alpha

		mean_texture = np.mean(normalized_data, axis=0, keepdims=True)
		# Standardize mean

		mean_texture -= np.mean(mean_texture, axis=(1,2), keepdims=True)
		mean_texture /= np.mean((mean_texture**2), axis=(1,2), keepdims=True)
		# 
		# mean_texture =  (mean_texture-beta)/alpha

		if np.linalg.norm(prev_texture-mean_texture) < 0.0000001:
			break

		prev_texture = mean_texture
		counter += 1

	print('Texture mean found in {} iterations'.format(counter))

	return mean_texture, beta, alpha, normalized_data


