from solver import solve
from visualize_pointset import get_coordinates

import os
import h5py
import numpy as np 

# ------------------------- Face Dataset -------------------------
data_path = '../imm3943/IMM-Frontal Face DB SMALL/'
N = len(os.listdir(data_path))-3

# pointset_data = np.zeros((N, 73, 2))
pointset_data = []
for i, img_path in enumerate(os.listdir(data_path)):
	if not '.jpg' in img_path:
		continue
	complete_path = os.path.join(data_path, img_path)
	x_coordinates, y_coordinates, connect_from, connect_to = get_coordinates(complete_path)
	x_coordinates = 600*np.expand_dims(x_coordinates, 1); y_coordinates = 800*(1-np.expand_dims(y_coordinates, 1))
	
	coordinates = np.expand_dims(np.concatenate((x_coordinates, y_coordinates), axis=1), 0)
	pointset_data.append(coordinates)

pointset_data = np.concatenate(pointset_data, axis=0)
solve(pointset_data, connect_from, connect_to, save_plot_dir = '../results/faces', data_dir= '../data/faces/data')





