from util import *

def shi_tomasi(src, maxCorners, qualityLevel, minDistance, blockSize=3, ksize=3, k=0.04):
	def shi_tomasi_score(M):
		# return min(np.linalg.eigvals(M))
		return min(M[0,0], M[1,1])

	print "TRY: generating gradient matrix and score"
	gradient_matrix = generate_gradient_matrix(src, blockSize, ksize, k, shi_tomasi_score)
	print "SUCCESS: generating gradient matrix and score"

	t = time()

	print "TRY: getting good corners"
	good_corners = []
	size_y, size_x = src.shape
	for y in xrange(size_y):
		for x in xrange(size_x):
			if gradient_matrix[y, x] >= qualityLevel:
				good_corners.append((y,x))
	print "SUCCESS: getting good corners"
	
	print "TRY: picking best corners"
	good_corners.sort(key=lambda k:-gradient_matrix[k])
	top_corners = [good_corners.pop(0)]
	for c in good_corners:
		distance_curr = lambda corner: distance(c, corner)
		if max(map(distance_curr, top_corners)) >= minDistance:
			top_corners.append(c)
		if len(top_corners) == maxCorners:
			break
	print "SUCCESS: picking best corners"

	print "Time taken to pick corners: %.3f" % (time() - t)
	return top_corners

using_own_implementation = True
corner_st = shi_tomasi if using_own_implementation else cv2.goodFeaturesToTrack

img, gray = read_image()
corners = np.int0(corner_st(gray, 25, 0.01, 10))

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

plt.imshow(img),plt.show()
