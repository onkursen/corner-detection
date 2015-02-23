import cv2
import numpy as np

def onkurHarris(src, blockSize, ksize, k):
	size_y, size_x = src.shape
	dst = np.zeros(src.shape)
	gradient_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
	gradient_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)
	for y in xrange(size_y):
		for x in xrange(size_x):
			M = np.zeros((2, 2))
			for v in xrange(max(0, y - blockSize/2), min(size_y, y + blockSize/2 + 1)):
				for u in xrange(max(0, x - blockSize/2), min(size_x, x + blockSize/2 + 1)):
					gradients_at_uv = np.zeros((2, 2))
					gradients_at_uv[0, 0] = np.dot(gradient_x[v, u], gradient_x[v, u])
					gradients_at_uv[0, 1] = gradients_at_uv[1, 0] = np.dot(gradient_x[v, u], gradient_y[v, u])
					gradients_at_uv[1, 1] = np.dot(gradient_y[v, u], gradient_y[v, u])
					M += gradients_at_uv
			print y,x
			dst[y, x] = np.linalg.det(M) - k * (np.trace(M) ** 2)
	return dst

test = True
harris = onkurHarris if test else cv2.cornerHarris
	 
# filename = sys.argv[1]
filename = 'chessboard.jpg'
if not os.path.isfile(filename):
	print 'No such file:', filename
	exit(1)

img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = harris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()