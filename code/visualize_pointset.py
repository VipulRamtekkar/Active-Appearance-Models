import numpy as np 
import os 
import matplotlib.pyplot as plt 

def visualize_checkpoints(pointset, img_path):
	img = plt.imread(img_path)

	fig, ax = plt.subplots()
	ax.imshow(img, cmap='gray')
	
	x_values = list(pointset[0,:,0]); y_values = list(pointset[0,:,1])

	ax.plot(x_values, y_values, 'ro-')
	ax.plot([x_values[-1], x_values[0]], [y_values[-1], y_values[0]], 'ro-')
	# plt.show()



# pointset = np.load('../leaf.npy')

img = '../data/leaf/data/leaf_1.png'
# visualize_checkpoints(img, pointset)