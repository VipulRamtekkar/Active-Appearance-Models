from solver import solve
from visualize_pointset import get_coordinates

import os
import numpy as np 
import cv2 
# ------------------------- Face Dataset -------------------------
data_path = '../imm3943/IMM-Frontal Face DB SMALL/'
N = len(os.listdir(data_path))-3

def getpoints(triangleList):

	points = []
	
	for t in triangleList:
		points.append((t[0],t[1]))
		points.append((t[2],t[3]))
		points.append((t[4],t[5]))	

	points = np.array(points)

	return points 

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

mean = pointset_data[58][0]

img = cv2.imread("../imm3943/IMM-Frontal Face DB SMALL/01_01.jpg")
size = img.shape
rectangle = (0,0, size[1], size[2])
subdiv = cv2.Subdiv2D()
for p in mean:
	subdiv.insert((int(p[0]),int(p[1])))

triangleList = subdiv.getTriangleList()

mean_points = getpoints(triangleList)


for i in range(len(pointset_data)):

	points = pointset_data[i][0]

	subdiv = cv2.Subdiv2D()
	for p in points:
		subdiv.insert((int(p[0]),int(p[1])))

	SampletraingleList = subdiv.getTriangleList()

	Samplepoints = getpoints(SampletraingleList)

	
