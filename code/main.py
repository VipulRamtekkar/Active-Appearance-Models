from solver import solve
from visualize_pointset import get_coordinates
from functions import *

import os
import h5py
import numpy as np 
from scipy.spatial import Delaunay

# ------------------------- Face Dataset -------------------------
data_dir = '../imm3943/IMM-Frontal Face DB SMALL/'
N = len(os.listdir(data_dir))-3

# pointset_data = np.zeros((N, 73, 2))
pointset_data = []
for i, img_path in enumerate(sorted(os.listdir(data_dir))):
	if not '.jpg' in img_path:
		continue
	complete_path = os.path.join(data_dir, img_path)
	x_coordinates, y_coordinates, connect_from, connect_to = get_coordinates(complete_path)
	x_coordinates = 600*np.expand_dims(x_coordinates, 1); y_coordinates = 800*(1-np.expand_dims(y_coordinates, 1))
	
	coordinates = np.expand_dims(np.concatenate((x_coordinates, y_coordinates), axis=1), 0)
	pointset_data.append(coordinates)

pointset_data = np.concatenate(pointset_data, axis=0)
eig_vecs, eig_values, mean = solve(pointset_data, connect_from, connect_to, save_plot_dir='../results/faces', data_dir=data_dir)

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







