import cv2
import numpy as np
from time import time
import os
import sys

# Useful links:
# http://stackoverflow.com/questions/3862225/implementing-a-harris-corner-detector
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html

# Same argument set as cv2.cornerHarris
def harris(src, blockSize, ksize, k):
	o = time()

	# Return value of the method
	dst = np.zeros(src.shape)

	# Use depth constants based on this StackOverflow article:
	# http://stackoverflow.com/q/11331830/
	gradient_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
	gradient_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)

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
			dst[y, x] = np.linalg.det(M) - k * (np.trace(M) ** 2)
	print 'Total time: %.3f' % (time() - o)
	return dst

using_own_implementation = True
corner_harris = harris if using_own_implementation else cv2.cornerHarris

# Below source code modified from:
# http://docs.opencv.org/trunk/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html

filename = sys.argv[1]
if not os.path.isfile(filename):
	print 'No such file:', filename
	exit(1)

img = cv2.imread(filename)
gray = np.float32(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
dst = corner_harris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
