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
	visualLabels = cv.cvtColor(visualLabels, cv.COLOR_HSV2RGB)
	visualLabels[labelHue == 0] = 0

	return visualLabels

print("Orientation is given relative to the positive x semiaxis, counter-clockwise")

DUST_AREA_THRESHOLD = 50

IMG  = [0, 1, 12, 21, 31, 33]
IMG += [44, 47, 48, 49] # Other objects
#IMG += [50, 51] # Contact points
IMG += [90, 92, 98] # Iron dust

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
	# Remove dust by opening
	strEl = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
	binary = cv.morphologyEx(binary, cv.MORPH_OPEN, strEl)
	# Clean the result by closing
	strEl = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
	binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, strEl)
	showImage(title + "Binarized", binary)
	
	# Label
	_, labels, stats, centers = cv.connectedComponentsWithStats(binary)
	print("Found %d object(s)" % np.max(labels))

	labelsMER = np.zeros_like(labels, dtype=np.uint8)
	finalLabel = 1
	for l in range(0, np.max(labels) + 1):
		if stats[l][0] == 0 and stats[l][1] == 0:
			# Background component
			continue

		print("Object %d at (%.2f, %.2f)" % (l, centers[l][0], centers[l][1]))
		print("\tArea: %d" % stats[l][4])

		if stats[l][4] < DUST_AREA_THRESHOLD:
			print("\tDust detected!")
			continue

		component = np.array([[255 if pixel == l else 0 for pixel in row] for row in labels], dtype=np.uint8)
		_, _, holesStats, holesCenters = cv.connectedComponentsWithStats(255 - component)
		holes = []
		for hStat, hCenter in zip(holesStats, holesCenters):
			if hStat[0] == 0 and hStat[1] == 0:
				# This is the main background, not relevant
				continue
			elif math.sqrt((centers[l][0] - hCenter[0])**2 + (centers[l][1] - hCenter[1])**2) < 1:
				# This is the object, ie the background of this labelling
				# It could also be a hole with center at the center of the object, which is not interesting for the rods
				continue
			else:
				holes += [(hCenter[0], hCenter[1], 2 * math.sqrt(hStat[4] / math.pi))]

		print("\tInteresting hole count: %d" % len(holes))
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
		alpha = -math.sin(theta)
		beta = math.cos(theta)
		major = (alpha, -beta,  beta*centers[l][1] - alpha*centers[l][0])
		minor = (beta,  alpha, -beta*centers[l][0] - alpha*centers[l][1])

		# Compute oriented MER
		# c1 largest positive distance (third component) from major axis
		c1 = (0,0,0)
		# c2 largest negative distance from major axis
		c2 = (0,0,0)
		# c3 largest positive distance from minor axis
		c3 = (0,0,0)
		# c4 largest negative distance from minor axis
		c4 = (0,0,0)
		# Points on the minor axis with the largest positive and negative distance from the major one
		wb1 = (0,0,0)
		wb2 = (0,0,0)
		for x in range(stats[l][0], stats[l][0] + stats[l][2]):
			for y in range(stats[l][1], stats[l][1] + stats[l][3]):
				if component[y][x] != 255:
					continue

				distMaj = (major[0]*x + major[1]*y + major[2]) / math.sqrt(major[0]**2 + major[1]**2)
				distMin = (minor[0]*x + minor[1]*y + minor[2]) / math.sqrt(minor[0]**2 + minor[1]**2)

				c1 = (x,y,distMaj) if distMaj > c1[2] else c1
				c2 = (x,y,distMaj) if distMaj < c2[2] else c2

				c3 = (x,y,distMin) if distMin > c3[2] else c3
				c4 = (x,y,distMin) if distMin < c4[2] else c4

				if abs(distMin) < 0.5:
					# The point is on the minor axis, use it to compute the width at the barycenter
					wb1 = (x,y,distMaj) if distMaj > wb1[2] else wb1
					wb2 = (x,y,distMaj) if distMaj < wb2[2] else wb2

		# Contact points found, compute MER vertices
		line1 = (alpha/beta, c1[1] - alpha/beta*c1[0])
		line2 = (alpha/beta, c2[1] - alpha/beta*c2[0])
		line3 = (-beta/alpha, c3[1] + beta/alpha*c3[0])
		line4 = (-beta/alpha, c4[1] + beta/alpha*c4[0])

		x = (line1[1] - line3[1]) / (line3[0] - line1[0])
		v1 = (x, line1[0]*x + line1[1])
		x = (line1[1] - line4[1]) / (line4[0] - line1[0])
		v2 = (x, line1[0]*x + line1[1])

		x = (line2[1] - line3[1]) / (line3[0] - line2[0])
		v3 = (x, line2[0]*x + line2[1])
		x = (line2[1] - line4[1]) / (line4[0] - line2[0])
		v4 = (x, line2[0]*x + line2[1])
		mer = cv.line(component, (int(v1[0]), int(v1[1])), (int(v3[0]), int(v3[1])), (255,255,255))
		mer = cv.line(mer, (int(v3[0]), int(v3[1])), (int(v4[0]), int(v4[1])), (255,255,255))
		mer = cv.line(mer, (int(v4[0]), int(v4[1])), (int(v2[0]), int(v2[1])), (255,255,255))
		mer = cv.line(mer, (int(v2[0]), int(v2[1])), (int(v1[0]), int(v1[1])), (255,255,255))
		mer = cv.line(mer, (wb1[0], wb1[1]), (wb2[0], wb2[1]), (0,0,0))
		labelsMER = cv.max(labelsMER, np.array([[finalLabel if pixel == 255 else 0 for pixel in row] for row in mer], dtype=np.uint8))
		finalLabel += 1

		length = math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)
		width = math.sqrt((v1[0] - v3[0])**2 + (v1[1] - v3[1])**2)
		widthBar = math.sqrt((wb1[0] - wb2[0])**2 + (wb1[1] - wb2[1])**2)
		print("\tLength: %.2f" % length)
		print("\tWidth: %.2f" % width)
		print("\tWidth at the barycenter: %.2f" % widthBar)

		for i, hData in enumerate(holes):
			hX, hY, hD = hData
			print("\tHole %d: At (%.2f, %.2f) with diameter %.2f" % (i+1, hX, hY, hD))

		showImage(title + "Component %d" % l, mer)

	showImage(title + "Labelled", humanLabels(labelsMER))
