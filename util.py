import cv2
from matplotlib import pyplot as plt
import numpy as np
from time import time
import os
import sys

# Same argument set as cv2.cornerHarris
def generate_gradient_matrix(src, blockSize, ksize, k):
	o = time()

	size_y, size_x = src.shape
	# Return value of the method
	dst = {}

	# Use depth constants based on this StackOverflow article:
	# http://stackoverflow.com/q/11331830/
	gradient_x = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=ksize)
	gradient_y = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=ksize)
	
	gradient_xx = np.square(gradient_x)
	gradient_yy = np.square(gradient_y)
	gradient_xy = np.multiply(gradient_x, gradient_y)
	print 'time so far: %.3f' % (time() - o)
	print 'total iterations', size_y * size_x
	raw_input("Continue?")
	
	for y in xrange(size_y):
		# oy = time()
		for x in xrange(size_x):
			M = np.zeros((2, 2))

			# Use a blockSize x blockSize window (except for pixels too close to the edges)
			for v in xrange(max(0, y - blockSize / 2), min(size_y, y + blockSize / 2)):
				for u in xrange(max(0, x - blockSize / 2), min(size_x, x + blockSize / 2)):
					M[0, 0] += gradient_xx[v, u]
					M[0, 1] += gradient_xy[v, u]
					M[1, 1] += gradient_yy[v, u]
			M[1, 0] = M[0, 1]

			# Calculate score as given in the paper
			dst[(y, x)] = M
		# print 'Time for this iteration: %.3f' % (time() - oy)
			
	print 'Total time: %.3f' % (time() - o)
	return dst

def distance(p1, p2):
	y, x = p1
	v, u = p2
	return ((y - v) ** 2 + (x - u) ** 2) ** 0.5
