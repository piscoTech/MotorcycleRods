import cv2 as cv
import numpy as np
import sys
import math

def showImage(title, image, scale=1):
	image = cv.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))
	cv.imshow(title, image)
	cv.waitKey()
	cv.destroyAllWindows()

def humanLabels(labels):
	# Map component labels to hue value for human display
	labelHue = np.uint8(179 * labels / np.max(labels))
	blackCh = 255 * np.ones_like(labelHue)
	visualLabels = cv.merge([labelHue, blackCh, blackCh])
	visualLabels = cv.cvtColor(visualLabels, cv.COLOR_HSV2BGR)
	visualLabels[labelHue == 0] = 0

	return visualLabels

print("Orientation is given relative to the positive x semiaxis, counter-clockwise")

IMG  = [0, 1, 12, 21, 31, 33]
#IMG += [44, 47, 48, 49] # Other objects
#IMG += [50, 51] # Contact points
#IMG += [90, 92, 98] # Iron dust

#IMG = IMG[3:4]

for img in IMG:
	title = "Image %02d - " % img
	print("\nAnalysing image %02d..." % img)
	img = cv.imread("images/TESI%02d.BMP" % img, cv.IMREAD_GRAYSCALE)
	if img == None:
		print("Image %02d not found!" % img)
		sys.exit()
	showImage(title + "Original", img)

	# Binarize
	th, binary = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
	print("Threshold used: %d" % th)
	# Clean by closing
	strEl = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
	binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, strEl)
	showImage(title + "Binarized", binary)

	# Label
	_, labels, stats, centers = cv.connectedComponentsWithStats(binary)
	print("Found %d object(s)" % np.max(labels))

	# Start from 1 to ignore background
	for l in range(1, np.max(labels) + 1):
		print("Object %d at (%.2f, %.2f)" % (l, centers[l][0], centers[l][1]))
		print("\tArea: %d" % stats[l][4])

		component = np.array([[255 if pixel == l else 0 for pixel in row] for row in labels], dtype=np.uint8)
		_, _, holesStats, holesCenters = cv.connectedComponentsWithStats(255 - component)
		holes = []
		for hStat, hCenter in zip(holesStats, holesCenters):
			if hStat[0] == 0 or hStat[1] == 0:
				# This is the main background, not relevant
				continue
			elif math.sqrt((centers[l][0] - hCenter[0])**2 + (centers[l][1] - hCenter[1])**2) < 1:
				# This is the object, ie the background of this labelling
				continue
			else:
				holes += [(hCenter[0], hCenter[1], 2 * math.sqrt(hStat[4] / math.pi))]

		print("\tHole count: %d" % len(holes))
		rType = "A" if len(holes) == 1 else ("B" if len(holes) == 2 else "Not a rod")
		print("\tRod Type: %s" % rType)
		if len(holes) < 1 or len(holes) > 2:
			continue

		# Compute moments for the component, central moments are 'mu20', 'mu11', 'mu02', etc. up to third order
		moments = cv.moments(component, True)
		# The difference is the other way around wrt to the studied algorithm as for openCV the first index is for the horizontal rather than the vertical
		theta = -0.5 * math.atan(2 * moments['mu11'] / (moments['mu20'] - moments['mu02']))
		d2theta = 2 * (moments['mu20'] - moments['mu02']) * math.cos(2 * theta) - 4 * moments['mu11'] * math.sin(2 * theta)
		theta = theta if d2theta > 0 else theta + math.pi / 2

		print("\tOrientation: %.1f deg" % (theta * 180 / math.pi))
		for i, hData in enumerate(holes):
			hX, hY, hD = hData
			print("\tHole %d: At (%.2f, %.2f) with diameter %.2f" % (i+1, hX, hY, hD))

	showImage(title + "Labelled", humanLabels(labels))
