import numpy as np
import cv2

path = r"C:\Users\Daniel\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\calvingfrontmachine\landsat_preds\8_pred.png"
path2 = r"C:\Users\Daniel\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\calvingfrontmachine\landsat_preds\8_contours.png"

im = cv2.imread(path)
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray,127,255,0)
thresh = 255 - thresh
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
backtorgb = cv2.cvtColor(im2,cv2.COLOR_GRAY2RGB)
cnt = contours[0]
epsilon = 0.0025 * cv2.arcLength(cnt, False)
approx = cv2.approxPolyDP(cnt, epsilon, False)
	
#cv2.drawContours(backtorgb, contours, -1,(255,255,0),3)
cv2.drawContours(backtorgb, [approx], -1,(255,255,0),3)
cv2.imshow("Keypoints", backtorgb)
cv2.waitKey(1)