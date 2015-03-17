from util import *

def shi_tomasi(src, maxCorners, qualityLevel, minDistance, blockSize=3, ksize=3, k=0.04):
	def shi_tomasi_score(M):
		return min(M[0,0], M[1,1])
	gradient_matrix = generate_gradient_matrix(src, blockSize, ksize, k, shi_tomasi_score)

	t = time()
	print "TRY: picking best corners"

	good_corners = []
	size_y, size_x = src.shape
	for y in xrange(size_y):
		for x in xrange(size_x):
			if gradient_matrix[y, x] >= qualityLevel:
				good_corners.append((y,x))
	
	good_corners.sort(key=lambda k:-gradient_matrix[k])
	top_corners = [good_corners.pop(0)]
	for c in good_corners:
		distance_curr = lambda corner: distance(c, corner)
		if max(map(distance_curr, top_corners)) >= minDistance:
			top_corners.append(c)
		if len(top_corners) == maxCorners:
			break

	print "SUCCESS (%.3f secs): picking best corners" % (time() - t)
	return np.int0(top_corners)

# Code in this method modified from:
# http://docs.opencv.org/trunk/doc/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html
def write_image_with_corners_and_show(name, image, corners):
	for i in corners:
	    x,y = i.ravel()
	    cv2.circle(img,(x,y),3,(0,0,255),-1)
	cv2.imwrite('%s.png' % name, image)
	cv2.imshow(name, image)

check_usage('shi-tomasi')
img, gray = read_image()

t_onkur = time()
print "TRY: Running Onkur's Shi-Tomasi corner detector"
onkurs = shi_tomasi(gray, 25, 0.01, 10)
print "SUCCESS (%.3f secs): Running Onkur's Shi-Tomasi corner detector" % (time() - t_onkur)

t_opencv = time()
print "TRY: Running OpenCV Shi-Tomasi corner detector"
opencv = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
print "SUCCESS (%.3f secs): Running OpenCV Shi-Tomasi corner detector" % (time() - t_opencv)

write_image_with_corners_and_show('onkurs', img, onkurs)
write_image_with_corners_and_show('opencv', img, opencv)
wait_until_done()
