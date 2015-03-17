import cv2
from matplotlib import pyplot as plt
import numpy as np
from time import time
import os
import sys

# Check whether the script is being called from the command line properly
def check_usage(script_name):
	if len(sys.argv) != 2:
		print 'Usage: python %s.py <filename>' % script_name
		exit(1)

# Code in this method modified from:
# http://docs.opencv.org/trunk/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
def read_image():
	filename = sys.argv[1]
	if not os.path.isfile(filename):
		print 'No such file:', filename
		exit(1)

	img = cv2.imread(filename)
	gray = np.float32(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
	return img, gray

def wait_until_done():
	if cv2.waitKey(0) & 0xff == 27:
	    cv2.destroyAllWindows()

# Same argument set as cv2.cornerHarris
def generate_gradient_matrix(src, blockSize, ksize, k, score):
	o = time()

	print "TRY: generate gradient matrix and score"
	size_y, size_x = src.shape
	# Return value of the method
	scored_image_gradient = np.zeros(src.shape)

	# Use depth constants based on this StackOverflow article:
	# http://stackoverflow.com/q/11331830/
	gradient_x = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=ksize)
	gradient_y = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=ksize)
	
	gradient_xx = np.square(gradient_x)
	gradient_yy = np.square(gradient_y)
	gradient_xy = np.multiply(gradient_x, gradient_y)
	for y in xrange(size_y):
		oy = time()
		for x in xrange(size_x):
			M = np.zeros((2, 2))

			# Use a blockSize x blockSize window (except for pixels too close to the edges)
			ymin = max(0, y - blockSize / 2)
			ymax = min(size_y, y + blockSize / 2)
			xmin = max(0, x - blockSize / 2)
			xmax = min(size_x, x + blockSize / 2)

			for v in xrange(ymin, ymax):
				for u in xrange(xmin, xmax):
					M[0, 0] += gradient_xx[v, u]
					M[0, 1] += gradient_xy[v, u]
					M[1, 1] += gradient_yy[v, u]
			M[1, 0] = M[0, 1]

			# Calculate score as given in the paper
			scored_image_gradient[y, x] = score(M)
	
	print "SUCCESS (%.3f secs): generate gradient matrix and score" % (time() - o)
	return scored_image_gradient

def distance(p1, p2):
	y, x = p1
	v, u = p2
	return ((y - v) ** 2 + (x - u) ** 2) ** 0.5
