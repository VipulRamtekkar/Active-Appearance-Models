'''
Piecewise affine warping script taken from: https://github.com/TimSC/image-piecewise-affine
'''

from PIL import Image
import numpy as np
import math
import scipy.spatial as spatial
import matplotlib.pyplot as plt 

def GetBilinearPixel(imArr, posX, posY, out):

	#Get integer and fractional parts of numbers
	modXi = int(posX)
	modYi = int(posY)
	modXf = posX - modXi
	modYf = posY - modYi

	#Get pixels in four corners
	for chan in range(imArr.shape[2]):
		bl = imArr[modYi, modXi, chan]
		br = imArr[modYi, modXi+1, chan]
		tl = imArr[modYi+1, modXi, chan]
		tr = imArr[modYi+1, modXi+1, chan]
	
		#Calculate interpolation
		b = modXf * br + (1. - modXf) * bl
		t = modXf * tr + (1. - modXf) * tl
		pxf = modYf * t + (1. - modYf) * b
		out[chan] = int(pxf+0.5) #Do fast rounding to integer

	return out #Helps with profiling view

def WarpProcessing(inIm, inArr, 
		outArr, 
		inTriangle, 
		triAffines, shape):
	
	blank_img = np.zeros_like(inArr)
	#Ensure images are 3D arrays
	px = np.empty((inArr.shape[2],), dtype=np.int32)
	homogCoord = np.ones((3,), dtype=np.float32)

	#Calculate ROI in target image
	xmin = shape[:,0].min()
	xmax = shape[:,0].max()
	ymin = shape[:,1].min()
	ymax = shape[:,1].max()
	xmini = int(xmin)
	xmaxi = int(xmax+1.)
	ymini = int(ymin)
	ymaxi = int(ymax+1.)
	#print xmin, xmax, ymin, ymax

	#Synthesis shape norm image	

	x_target, y_target = [], []
	for i in range(xmini, xmaxi):
		for j in range(ymini, ymaxi):
			homogCoord[0] = i
			homogCoord[1] = j

			#Determine which tesselation triangle contains each pixel in the shape norm image
			if i < 0 or i >= outArr.shape[1]: continue
			if j < 0 or j >= outArr.shape[0]: continue

			#Determine which triangle the destination pixel occupies
			tri = inTriangle[i,j]
			if tri == -1: 
				continue
				
			#Calculate position in the input image
			affine = triAffines[tri]
			outImgCoord = np.dot(affine, homogCoord)

			#Check destination pixel is within the image
			if outImgCoord[0] < 0 or outImgCoord[0] >= inArr.shape[1]:
				for chan in range(px.shape[0]): outArr[j,i,chan] = 0
				continue
			if outImgCoord[1] < 0 or outImgCoord[1] >= inArr.shape[0]:
				for chan in range(px.shape[0]): outArr[j,i,chan] = 0
				continue

			#Nearest neighbour
			#outImgL[i,j] = inImgL[int(round(inImgCoord[0])),int(round(inImgCoord[1]))]

			#Copy pixel from source to destination by bilinear sampling
			#print i,j,outImgCoord[0:2],im.size
			px_out = GetBilinearPixel(inArr, outImgCoord[0], outImgCoord[1], px)

			x_target.append(px_out[0])
			y_target.append(px_out[1])

			for chan in range(px.shape[0]):
				outArr[j,i,chan] = px[chan]
				blank_img[j,i,chan] = px[chan]
			#print outImgL[i,j]

	return xmini, xmaxi, ymini, ymaxi, blank_img

def PiecewiseAffineTransform(srcIm, srcPoints, dstIm, dstPoints):

	#Convert input to correct types
	srcArr = np.asarray(srcIm, dtype=np.float32)
	dstPoints = np.array(dstPoints)
	srcPoints = np.array(srcPoints)

	#Split input shape into mesh
	tess = spatial.Delaunay(dstPoints)

	#Calculate ROI in target image
	xmin, xmax = dstPoints[:,0].min(), dstPoints[:,0].max()
	ymin, ymax = dstPoints[:,1].min(), dstPoints[:,1].max()
	#print xmin, xmax, ymin, ymax

	#Determine which tesselation triangle contains each pixel in the shape norm image
	inTessTriangle = np.ones(dstIm.size, dtype=np.int) * -1
	for i in range(int(xmin), int(xmax+1.)):
		for j in range(int(ymin), int(ymax+1.)):
			if i < 0 or i >= inTessTriangle.shape[0]: continue
			if j < 0 or j >= inTessTriangle.shape[1]: continue
			normSpaceCoord = (float(i),float(j))
			simp = tess.find_simplex([normSpaceCoord])
			inTessTriangle[i,j] = simp

	#Find affine mapping from input positions to mean shape
	triAffines = []
	for i, tri in enumerate(tess.vertices):
		meanVertPos = np.hstack((srcPoints[tri], np.ones((3,1)))).transpose()
		shapeVertPos = np.hstack((dstPoints[tri,:], np.ones((3,1)))).transpose()

		affine = np.dot(meanVertPos, np.linalg.inv(shapeVertPos)) 
		triAffines.append(affine)

	#Prepare arrays, check they are 3D	
	targetArr = np.copy(np.asarray(dstIm, dtype=np.uint8))
	srcArr = srcArr.reshape(srcArr.shape[0], srcArr.shape[1], len(srcIm.mode))
	targetArr = targetArr.reshape(targetArr.shape[0], targetArr.shape[1], len(dstIm.mode))

	#Calculate pixel colours
	warp_out = WarpProcessing(srcIm, srcArr, targetArr, inTessTriangle, triAffines, dstPoints)
	
	#Convert single channel images to 2D
	if targetArr.shape[2] == 1:
		targetArr = targetArr.reshape((targetArr.shape[0],targetArr.shape[1]))
	dstIm.paste(Image.fromarray(targetArr))

	return warp_out

def normalize_shape(individual_img_path, target_img_path, show=False):
	#Load source image
	srcIm = Image.open(individual_img_path)	
	dstIm = Image.open(target_img_path)

	from visualize_pointset import get_coordinates

	src_pts = []
	x_values, y_values, connect_from, connect_to = get_coordinates(individual_img_path)
	for x,y in zip(x_values, y_values):
		src_pts.append((800*x,600*y ))

	dst_pts = []
	x_values, y_values, connect_from, connect_to = get_coordinates(target_img_path)
	for x,y in zip(x_values, y_values):
		dst_pts.append((800*x,600*y ))

	x_max = x_values.max(); x_min = x_values.min()
	y_max = y_values.max(); y_min = y_values.min()

	#Define control points for warp
	srcCloud = [(100,100),(400,100),(400,400),(100,400)]
	dstCloud = [(150,120),(374,105),(410,267),(105,390)]

	#Perform transform
	mean_array = np.array(dstIm.convert('L'))
	ymin, ymax, xmin, xmax, img_mask = PiecewiseAffineTransform(srcIm, src_pts, dstIm, dst_pts)
	shape_normalized_img = np.array(dstIm)
	mean_array = mean_array

	# import pdb; pdb.set_trace()
	try:
		shape_normalized_img = (shape_normalized_img*(img_mask!=0))[xmin:xmax, ymin:ymax]
	except:
		shape_normalized_img = (shape_normalized_img[xmin:xmax, ymin:ymax]*(img_mask!=0))

	if show:
		plt.imshow(shape_normalized_img)
		plt.show()
	else:
		return shape_normalized_img

	#Visualise result
	# srcIm.show()
	# dstIm.show()

if __name__ == "__main__":
	normalize_shape(individual_img_path="../imm3943/IMM-Frontal Face DB SMALL/11_07.jpg", 
		target_img_path="../imm3943/IMM-Frontal Face DB SMALL/08_01.jpg",show=True)

