from util import *

# Useful links:
# http://stackoverflow.com/questions/3862225/implementing-a-harris-corner-detector
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html

def harris(src, blockSize, ksize, k):
	def harris_score(M):
		return np.linalg.det(M) - k * (np.trace(M) ** 2)

	return generate_gradient_matrix(src, blockSize, ksize, k, harris_score)

using_own_implementation = True
corner_harris = harris if using_own_implementation else cv2.cornerHarris

# Below source code modified from:
# http://docs.opencv.org/trunk/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html

img, gray = read_image()
dst = corner_harris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
