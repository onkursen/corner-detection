from util import *

# Useful links:
# http://stackoverflow.com/questions/3862225/implementing-a-harris-corner-detector
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html

def harris(src, blockSize, ksize, k):
	def harris_score(M):
		return np.linalg.det(M) - k * (np.trace(M) ** 2)

	return generate_gradient_matrix(src, blockSize, ksize, k, harris_score)

# Code in this method modified from:
# http://docs.opencv.org/trunk/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
def write_image_with_corners_and_show(name, image, corners):
	#result is dilated for marking the corners, not important
	result = cv2.dilate(corners,None)

	# Threshold for an optimal value, it may vary depending on the image.
	image[result>0.01*result.max()]=[0,0,255]

	cv2.imwrite('%s.png' % name, image)
	cv2.imshow(name, image)

check_usage('harris')
img, gray = read_image()

t_onkur = time()
print "TRY: Running Onkur's Harris corner detector"
onkurs = harris(gray,2,3,0.04)
print "SUCCESS (%.3f secs): Running Onkur's Harris corner detector" % (time() - t_onkur)

t_opencv = time()
print "TRY: Running OpenCV Harris corner detector"
opencv = cv2.cornerHarris(gray,2,3,0.04)
print "SUCCESS (%.3f secs): Running OpenCV Harris corner detector" % (time() - t_opencv)

write_image_with_corners_and_show('onkurs', img, onkurs)
write_image_with_corners_and_show('opencv', img, opencv)
wait_until_done()