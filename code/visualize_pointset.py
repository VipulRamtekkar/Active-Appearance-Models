import numpy as np 
import os 
import cv2
import matplotlib.pyplot as plt 

def get_coordinates_lfpw(img_path):
	img_name = img_path.split('/')[-1].split('.')[0]
	coord_path = '../data/lfpw/trainset/{}.pts'.format(img_name)

	with open(coord_path) as f:
		content = f.readlines()
	points = content[3:71]
	x_values = np.array([float(point.split(' ')[0]) for point in points])
	y_values = np.array([float(point.split(' ')[1]) for point in points])

	base = np.arange(68)

	connect_from = np.roll(base, -1, 0)
	connect_to = np.roll(base, 1, 0)

	return x_values, y_values, connect_from, connect_to

def get_coordinates(img_path):
	img_name = img_path.split('/')[-1].split('.')[0]

	coord_path = os.path.join(os.path.join(*img_path.split('/')[0:-1]), 'asf/{}.asf'.format(img_name))

	with open(coord_path) as f:
		content = f.readlines()
	points = content[16:89]

	x_values = np.array([float(point.split('\t')[2]) for point in points])
	y_values = np.array([float(point.split('\t')[3]) for point in points])

	connect_from = np.array([int(point.split('\t')[5]) for point in points])
	connect_to = np.array([int(point.split('\t')[6]) for point in points])

	return x_values, y_values, connect_from, connect_to

def plot_pointset_with_connections(x_values, y_values, connect_from, connect_to, label='connections', color='black'):
	if color == 'black':
		plt.scatter(x_values, y_values)
	else:
		plt.scatter(x_values, y_values, color=color)
	for i, (c_f, c_t) in enumerate(zip(connect_from, connect_to)):
		if i == 0:		
			plt.plot([x_values[c_f], x_values[i]], [y_values[c_f], y_values[i]],label=label, color=color)
		else:
			plt.plot([x_values[c_f], x_values[i]], [y_values[c_f], y_values[i]], color=color)
		plt.plot([x_values[i], x_values[c_t]], [y_values[i], y_values[c_t]], color=color)


def visualize_checkpoints(img_path, show=True):
	img = plt.imread(img_path)
	
	x_values, y_values, connect_from, connect_to = get_coordinates(img_path)
	x_values *= img.shape[1]; y_values *= img.shape[0]

	fig, ax = plt.subplots()
	ax.imshow(img)
	ax.scatter(x_values, y_values)

	for i, (c_f, c_t) in enumerate(zip(connect_from, connect_to)):
		ax.plot([x_values[c_f], x_values[i]], [y_values[c_f], y_values[i]], color='black')
		ax.plot([x_values[i], x_values[c_t]], [y_values[i], y_values[c_t]], color='black')
	if show:
		plt.show()

def plot(s1,s2,j):

	x1 = []
	y1 = []
	x2 = []
	y2 = []

	for i in range(len(s1)):
		x1.append(s1[i][0])
		y1.append(s1[i][1])

	for i in range(len(s2)):
		x2.append(s2[i][0])
		y2.append(s2[i][1])

	plt.scatter(x1,y1)
	plt.scatter(x2,y2)
	plt.title("Comparing fit against annotation")
	plt.savefig('../results/faces/comparison'+str(j)+'.png')
	plt.clf()



if __name__ == '__main__':
	img_path = '../imm3943/IMM-Frontal Face DB SMALL/05_10.jpg'
	visualize_checkpoints(img_path)

