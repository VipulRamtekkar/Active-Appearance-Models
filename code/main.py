from solver import solve

import h5py
import numpy as np 

# ------------------------- Ellipse dataset -------------------------
data_path = '../data/ellipse/ellipse2.npy'
data = np.load(data_path)
solve(data, save_plot_dir = '../results/ellipse', data_dir= '../data/ellipse/data')

# ------------------------- Hand dataset -------------------------
data_path = '../data/hand/data.mat'
f = h5py.File(data_path)
for k, v in f.items():
	data = np.array(v)

solve(data, save_plot_dir = '../results/hand')

# ------------------------- Leaf dataset -------------------------
data_path = '../data/leaf/leaf.npy'
data = np.load(data_path)
solve(data, save_plot_dir = '../results/leaf', data_dir= '../data/leaf/data')

# ------------------------- Brain MRI dataset -------------------------
data_path = '../data/brain/brain.npy'
data = np.load(data_path)
solve(data, save_plot_dir = '../results/brain', data_dir= '../data/brain/data')





