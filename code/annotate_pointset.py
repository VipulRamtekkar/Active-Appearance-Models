import argparse
import cv2
import os
import numpy as np 
import matplotlib.pyplot as plt

blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)

def annotate_click(event, x, y, flags, param):
	global refPt, cropping

	if event == cv2.EVENT_LBUTTONDOWN:
		param[0].append([x,y])

		image[y,x] = red
		cv2.putText(image, str(len(param[0])), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 1)
		cv2.imshow("image", image)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', '-d', default='sample.png' , required=False, help="Path to the image")
parser.add_argument('--pointset', '-p', required=True, help='path to store the pointsets')
parser.add_argument('--overwrite', '-o', action='store_true', help='if overwriting existing pointset file is allowed')
args = parser.parse_args()

num_points = 32
image_counter = 1
loop_exit = False

final_pointsets = [] 

if os.path.exists(args.pointset) and not args.overwrite:
	print('Pointset path exits! Please change the pointset file path or allow overwrite')
	exit(1)

for image_name in sorted(os.listdir(args.data_path)):

	image_pointset = []
	image_path = os.path.join(args.data_path, image_name)
	print('Reading', image_path)

	image = cv2.imread(image_path)
	print(image.shape)
	cv2.namedWindow("image")

	param = [image_pointset, image_counter]
	cv2.setMouseCallback("image", annotate_click, param)

	while True:
		cv2.imshow("image", image)
		key = cv2.waitKey(1) & 0xFF

		if key == 27 or len(image_pointset) == num_points:
			break
		if key == ord('q'):
			loop_exit = True
			break

	image_counter += 1
	if loop_exit:
		break

	final_pointsets.append(image_pointset)

cv2.destroyAllWindows()

final_pointsets = np.array(final_pointsets)

print(final_pointsets.shape)
np.save(args.pointset, final_pointsets, allow_pickle=True)

