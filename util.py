import cv2
from matplotlib import pyplot as plt
import numpy as np
from time import time
import os
import sys

# Same argument set as cv2.cornerHarris
def generate_gradient_matrix(src, blockSize, ksize, k):
	o = time()

	# Return value of the method
	dst = {}

	# Use depth constants based on this StackOverflow article:
	# http://stackoverflow.com/q/11331830/
	gradient_x = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=ksize)
	gradient_y = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=ksize)

	size_y, size_x = src.shape
	for y in xrange(size_y):
		for x in xrange(size_x):
			M = np.zeros((2, 2))

			# Use a blockSize x blockSize window (except for pixels too close to the edges)
			for v in xrange(max(0, y - blockSize / 2), min(size_y, y + blockSize / 2)):
				for u in xrange(max(0, x - blockSize / 2), min(size_x, x + blockSize / 2)):
					cross_term = np.dot(gradient_x[v, u], gradient_y[v, u])
					M[0, 0] += np.dot(gradient_x[v, u], gradient_x[v, u])
					M[0, 1] += cross_term
					M[1, 0] += cross_term
					M[1, 1] += np.dot(gradient_y[v, u], gradient_y[v, u])

			# Calculate score as given in the paper
			dst[(y, x)] = M
			
	print 'Total time: %.3f' % (time() - o)
	return dst

def distance(p1, p2):
	y, x = p1
	v, u = p2
	return ((y - v) ** 2 + (x - u) ** 2) ** 0.5
