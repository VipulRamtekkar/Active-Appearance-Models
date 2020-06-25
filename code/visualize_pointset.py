import numpy as np 
import os 
import cv2
import matplotlib.pyplot as plt 

def get_coordinates(img_path):
	img_name = img_path.split('/')[-1].split('.')[0]

	coord_path = '../imm3943/IMM-Frontal Face DB SMALL/asf/{}.asf'.format(img_name)

	with open(coord_path) as f:
		content = f.readlines()
	points = content[16:89]

	x_values = np.array([float(point.split('\t')[2]) for point in points])
	y_values = np.array([float(point.split('\t')[3]) for point in points])

	connect_from = np.array([int(point.split('\t')[5]) for point in points])
	connect_to = np.array([int(point.split('\t')[6]) for point in points])

	return x_values, y_values, connect_from, connect_to

def plot_pointset_with_connections(x_values, y_values, connect_from, connect_to, label='connections', color='black'):
	plt.scatter(x_values, y_values)
	for i, (c_f, c_t) in enumerate(zip(connect_from, connect_to)):
		if i == 0:		
			plt.plot([x_values[c_f], x_values[i]], [y_values[c_f], y_values[i]],label=label, color=color)
		else:
			plt.plot([x_values[c_f], x_values[i]], [y_values[c_f], y_values[i]], color=color)
		plt.plot([x_values[i], x_values[c_t]], [y_values[i], y_values[c_t]], color=color)


def visualize_checkpoints(img_path, pointset=None, show=True):
	img = plt.imread(img_path)
	
	if pointset == None:
		x_values, y_values, connect_from, connect_to = get_coordinates(img_path)
		x_values *= img.shape[1]; y_values *= img.shape[0]
	else:
		x_values = pointset[0,:,0]; y_values = pointset[0,:,1]

	fig, ax = plt.subplots()
	ax.imshow(img)
	ax.scatter(x_values, y_values)

	for i, (c_f, c_t) in enumerate(zip(connect_from, connect_to)):
		ax.plot([x_values[c_f], x_values[i]], [y_values[c_f], y_values[i]], color='black')
		ax.plot([x_values[i], x_values[c_t]], [y_values[i], y_values[c_t]], color='black')
	plt.show()


if __name__ == '__main__':
	img_path = '../imm3943/IMM-Frontal Face DB SMALL/05_10.jpg'
	visualize_checkpoints(img_path)

