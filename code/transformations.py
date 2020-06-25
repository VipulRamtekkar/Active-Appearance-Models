#Code for piecewise affine wrap, bilinear transform and Delauney Triangulation
#References: https://www.learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/

import numpy as np 
import cv2 


img = cv2.imread("../imm3943/IMM-Frontal Face DB SMALL/01_01.jpg")
size = img.shape
rectangle = (0,0, size[1], size[2])
subdiv = cv2.Subdiv2D()

points = []

with open("mean.txt") as file:
	for line in file:
		x, y = line.split()
		points.append((x,y))

for p in points:
	subdiv.insert(p)

triangleList = subdiv.getTriangleList()

# The points in the traingle are as follows (t[0],t[1]), (t[2],t[3]) and (t[4],t[5])

#The delauney traingles are found from the mean points

#Finding whether a point lies inside the delaunay triangle

#Barycentric Coordinates

def Return_Triangle(x,y,triangleList):

	for t in triangleList:
		x1, y1, x2, y2, x3, y3 = [t[i] for i in range(6)]

		beta = (y*x3 - y*x1 - y1*x3 - y3*x + x1*y3 + x*y1)/(-x2*y3 + x2*y1+x1*y3+x3*y2-x3*y1-x1*y2)

		if beta > 1 or beta < 0:
			continue

		gamma = (x*y2 - x*y1 - x1*y2 - x2*y + x2*y1 + x1*y)/(-x2*y3 + x2*y1 + x1*y3 + x3*y2 -x3*y1 - x1*y2)

		if gamma > 1 or gamma < 0:
			continue

		alpha = 1 - (beta + gamma)

		if alpha <= 1 and alpha >= 0:
			return t








