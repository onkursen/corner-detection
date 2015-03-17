from util import *

def shi_tomasi(src, maxCorners, qualityLevel, minDistance, blockSize=3, ksize=3, k=0.04):
	top_corners = np.ndarray(shape=(maxCorners, 1, 2))

	print "TRY: generating gradient matrix"
	gradient_matrix = generate_gradient_matrix(src, blockSize, ksize, k)
	print "SUCCESS: generating gradient matrix"

	print "TRY: scoring corners"
	good_corners = {}
	size_y, size_x = src.shape
	for y in xrange(size_y):
		for x in xrange(size_x):
			M = gradient_matrix[(y, x)]
			w, v = np.linalg.eig(M)
			score = min(w)
			if score >= qualityLevel:
				good_corners[(y, x)] = score
	print "SUCCESS: scoring corners"
	
	print "TRY: obtaining best corners"
	num_corners_obtained = 0
	while num_corners_obtained < maxCorners and bool(good_corners):
		best, score = max(good_corners.iteritems(), key=lambda k: k[1])
		del good_corners[best]
		top_corners[num_corners_obtained, 0] = best
		num_corners_obtained += 1

		corners_to_remove = []
		for c in good_corners:
			if distance(best, c) < minDistance:
				corners_to_remove.append(c)
		for c in corners_to_remove:
			del good_corners[c]
	print "SUCCESS: obtaining best corners"

	return top_corners[:num_corners_obtained]

using_own_implementation = True
corner_st = shi_tomasi if using_own_implementation else cv2.goodFeaturesToTrack

img = cv2.imread('test_images/ansel.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = corner_st(gray,25,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

plt.imshow(img),plt.show()