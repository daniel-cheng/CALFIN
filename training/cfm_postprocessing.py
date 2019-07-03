# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 14:51:21 2018

@author: Daniel
"""

from __future__ import print_function
from collections import defaultdict

import os, cv2, skimage, random, glob
import numpy as np

from skimage.io import imsave, imread
from skimage.transform import resize, rotate, rescale
from skimage.util import invert
from scipy.ndimage.filters import median_filter
from scipy import ndimage
import scipy.ndimage.morphology as m

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize

import warnings
import traceback


def skeletonize(img):
	h1 = np.array([[0, 0, 0],[0, 1, 0],[1, 1, 1]]) 
	m1 = np.array([[1, 1, 1],[0, 0, 0],[0, 0, 0]]) 
	h2 = np.array([[0, 0, 0],[1, 1, 0],[0, 1, 0]]) 
	m2 = np.array([[0, 1, 1],[0, 0, 1],[0, 0, 0]])	
	hit_list = [] 
	miss_list = []
	for k in range(4): 
		hit_list.append(np.rot90(h1, k))
		hit_list.append(np.rot90(h2, k))
		miss_list.append(np.rot90(m1, k))
		miss_list.append(np.rot90(m2, k))	
	img = img.copy()
	while True:
		last = img
		for hit, miss in zip(hit_list, miss_list): 
			hm = m.binary_hit_or_miss(img, hit, miss) 
			img = np.logical_and(img, np.logical_not(hm)) 
		if np.all(img == last):  
			break
	img = np.where(img == True, 255, 0).astype(np.uint8)
	return img


def mask_to_points(img_mask):
	skel = skeletonize(img_mask)
	x, y = skel.nonzero()
	return x, y


def make_video(input_path, image_paths, output_path, fps=1.0, size=None,
			   is_color=True, format='MJPG'):
	"""
	Create a video from a list of images.

	@param	  outvid	 	output video
	@param	  image_paths	list of images to use in the video
	@param	  fps		 	frame per second
	@param	  size			size of each frame
	@param	  is_color		color
	@param	  format	 	see http://www.fourcc.org/codecs.php
	@return					see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

	The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
	By default, the video will have the size of the first image.
	It will resize every image to this size before adding them to the video.
	"""
	
	video_name = output_path + '.avi'
	mp4_name = output_path + '_cf.avi'
	fourcc = VideoWriter_fourcc(*'MJPG')
	vid = None
	for image_path in image_paths:
		img = imread(image_path)
		if not os.path.exists(image_path):
			raise FileNotFoundError(image_path)
		if vid is None:
			if size is None:
				size = img.shape[1], img.shape[0]
			vid = VideoWriter(video_name, fourcc, float(fps), size, is_color)
		if size[0] != img.shape[1] or size[1] != img.shape[0]:
			img = resize(img, size)

		vid.write(img)
	vid.release()
	return

def clusterCloseContours(contours):
	endpoints = []
	clustered = []
	index_mapper = dict(int)
	threshold = 50
	#Look for 
	sorting_key = lambda x: cv2.arcLength(x, False)
	list_of_merges = []
	
	#Start with largest contour
	##Attach all that match with itself
	#Repeat until no other changes
	
	for i in range(len(contours)):		
		start1 = np.array(contours[i][0])
		end1 = np.array(contours[i][-1])
		index_mapper[i] = i
		clustered.append(contours[i])
		for j in range(i + 1, len(contours)):
			start2 =  np.array(contours[j][0])
			end2 =  np.array(contours[j][-1])
			dist1 = np.linalg.norm(end1 - start2)
			dist2 = np.linalg.norm(start1 - end2)
			if dist1 < threshold:
				print('contour ', i, ' extended by contour', j)
#				list_of_merges
				index_mapper[i] = j
				clustered.append(contours[i].extend(contours[j]))
			elif dist2 < threshold:
				print('contour ', j, ' extended by contour', i)
				index_mapper[j] = i
	return sorted(contours, key=sorting_key)

def overlay_front(raw_path, mask_path, overlay_path, date):
	raw = imread(raw_path, 0).astype(np.uint8)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	raw = clahe.apply(raw)
	
#	raw = cv2.convertScaleAbs(raw, alpha=(255.0/65535.0))
	raw_rgb = np.stack((raw,)*3, axis=-1)
	img_mask = imread(mask_path, 0).astype(np.uint8)
	img_mask = np.where(img_mask > np.mean(img_mask)/2, 255, 0).astype(np.uint8)
	
	skeleton = skeletonize(img_mask)
	
	im2, contours, hierarchy = cv2.findContours(skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	if (len(contours) == 0):
		print('No contours for:', mask_path)
		return
		
#	mergedContours = clusterCloseContours(contours)
#	cf_contour = max(contours, key=len)
	overlaid = cv2.drawContours(raw_rgb, contours, -1, (255,0,0), 2)
	
#	perimeter = cv2.arcLength(cf_countour, True)
#	approx = cv2.approxPolyDP(cf_countour, 0.0005 * perimeter, False) 
#	imgApprox = np.zeros(img_mask.shape,dtype=np.uint8)
#	imgApprox = cv2.drawContours(imgApprox, [approx], -1, (255,255,255), 2)

	#Write date to image
	widthScale = overlaid.shape[0] / 500 #(font is scaled for Upernavik, which is 500 pixels wide)
	heightScale = overlaid.shape[1] / 500 #(font is scaled for Upernavik, which is 500 pixels wide)

	bottomLeftCornerOfText = (4, overlaid.shape[1] - int(34 * heightScale))
	font = cv2.FONT_HERSHEY_SIMPLEX
	fontScale = 1 * widthScale
	fontColor = (0, 192, 216)
	fontColorBorder = (0, 0, 0)
	lineType = 2
	thickness = 2
	thicknessBorder = 8
	#text_width, text_height = cv2.getTextSize(date, font, fontScale, lineType)[0]
	cv2.putText(overlaid, date, bottomLeftCornerOfText, font, fontScale, fontColorBorder, thickness=thicknessBorder, lineType=lineType)
	cv2.putText(overlaid, date, bottomLeftCornerOfText, font, fontScale, fontColor, thickness=thickness, lineType=lineType)
	
	imsave(overlay_path, overlaid)
		
def overlay_images(resolutions):
	input_path = r"landsat_frame_boundaries"
	output_path = r"landsat_overlay_boundaries"
	domains = os.listdir(input_path)
	print('Overlaying front and date on raw image...')
	for domain in domains:
		domain_path = os.path.join(input_path, domain)
		images = os.listdir(domain_path)

		overlay_domain_path = os.path.join(output_path, domain)
		if (not os.path.exists(overlay_domain_path)):
			os.mkdir(overlay_domain_path)

		for name in images:
			if '_pred' not in name:
				print('overlaying:', name)
				try:
					#Determine name and domain
					name_split = name.split('_')
					domain = name_split[0]
					date = name_split[3]
					basename = name[0:-8]

					#Determine paths
					overlay_domain_path = os.path.join(output_path, domain)
					raw_path = os.path.join(domain_path, basename + '_raw.png')
					pred_path = os.path.join(domain_path, basename + '_pred.png')
					overlay_path = os.path.join(overlay_domain_path, basename + '_overlay.png')

					#Overlay the frame
					overlay_front(raw_path, pred_path, overlay_path, date)
				except Exception as e:
					print('Error in:', name)
					traceback.print_exc()
	
def preprocess():
	'''Resize images to median resolution and put them into folders'''
	input_path = r"landsat_preds_boundaries"
	frame_path = r"landsat_frame_boundaries"
	overlay_path = r"landsat_overlay_boundaries"
	images = os.listdir(input_path)
	raw_resolutions = defaultdict(list)
	print('Determining median domain image sizes...')
	for name in images:
		if '_pred' not in name:
			try:
				#Determine name and domain
				name_split = name.split('_')
				domain = name_split[0]
				basename = name[0:-8]

				#Get resolution of each image and add to resolutions for median analysis later
				raw_path = os.path.join(input_path, name)
				resolution = imread(raw_path, 0).shape
				raw_resolutions[domain].append(resolution)
			except Exception as e:
				print('Error in:', name)
				traceback.print_exc()

	resolutions = defaultdict(tuple)
	for domain in raw_resolutions:
		#Save median resolutions
		resolution = np.median(raw_resolutions[domain], axis=0)
		resolutions[domain] = (int(resolution[0]), int(resolution[1]))
		print(domain, 'median resolution:', resolution)

		
		#Generate folder for images
		frame_domain_path = os.path.join(frame_path, domain)
		if (not os.path.exists(frame_domain_path)):
			os.mkdir(frame_domain_path)
		overlay_domain_path = os.path.join(overlay_path, domain)
		if (not os.path.exists(overlay_domain_path)):
			os.mkdir(overlay_domain_path)

	print('Resizing images domain image sizes...')
	images = os.listdir(input_path)
	for name in images:
		if '_pred' not in name:
			try:
				print('resizing:', name)
				#Determine name and domain
				name_split = name.split('_')
				domain = name_split[0]
				date = name_split[3]

				#Determine input paths
				basename = name[0:-8]
				raw_path = os.path.join(input_path, basename + '_raw.png')
				pred_path = os.path.join(input_path, basename + '_pred.png')

				#Resize images
				raw_resized = resize(imread(raw_path, 0), resolutions[domain])
				pred_resized = resize(imread(pred_path, 0), resolutions[domain])

				#Determine save paths
				frame_domain_path = os.path.join(frame_path, domain)
				frame_raw_path = os.path.join(frame_domain_path, basename + '_raw.png')
				frame_pred_path = os.path.join(frame_domain_path, basename + '_pred.png')

				#Save resized images
				imsave(frame_raw_path, raw_resized)
				imsave(frame_pred_path, pred_resized)
			except Exception as e:
				print('Error in:', name)
				traceback.print_exc()
	return resolutions

def date_sort_key(name):
	date = name.split('/')[-1].split('_')[3]
	return date

def images_to_video():
	output_path = r"landsat_overlay_boundaries"
	video_output_path = "../public_html"
	domains = os.listdir(output_path)
	for domain in domains:
		try:
			domain_path = os.path.join(output_path, domain)
			image_paths = glob.glob(domain_path + "/*.png")
			image_paths.sort(key=date_sort_key)
			make_video(domain_path, image_paths, os.path.join(video_output_path, domain))
		except Exception as e:
			print('Error in:', domain)
			traceback.print_exc()
		
		
def	evaluate_error():
	pass	
def fxn():
    warnings.warn("UserWarning", UserWarning)

if __name__ == "__main__":

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		fxn()

		#evaluate_error()
		#resolutions = preprocess()
	#	resolutions = {'Upernavik-NE': (503, 475), 'Kangiata-Nunata': (172, 226), 'Kong-Oscar':(326, 304), 'Rink-Isbrae': (332, 328), 'Hayes': (287, 350), 'Jakobshavn': (644, 912), 'Akugdlerssup': (144, 224), 'Cornell': (174, 173), 'Dietrichson': (173, 241), 'Docker-Smith': (157, 155), 'Docker-Smith-N': (184, 202), 'Eqip': (92, 100), 'Gade': (240, 226), 'Hayes-M': (159, 176), 'Hayes-S': (355, 232), 'Igssussarssuit': (231, 225), 'Illullip': (242, 189), 'Inngia': (391, 405), 'Kangerlussuaq': (183, 261),'Kangilernata': (87, 112), 'Kjer': (302, 286), 'Morell': (233, 224), 'Nansen': (280, 378), 'Narsap': (291, 403), 'Nordenskiold': (362, 281), 'Rink-Gletscher': (204, 257), 'Saqqarliup': (87, 122), 'Sermeq-Avannarleq-N': (70, 122), 'Sermeq-Avannarleq-S': (84, 108), 'Sermeq-Kujalleq': (98, 86), 'Steenstrup': (387, 304), 'Sverdrup': (306, 356), 'Umiammakku': (362, 351), 'Upernavik-SE': (466, 287)}
		#print(resolutions)
		#overlay_images(resolutions)
		images_to_video()
		#os.system("source deactivate")
		#os.system("~/public_html/video4web.sh avi 1280")
		#os.system("chmod 644 ~/public_html/*")
				
		#Cluster close endpoints
		#Make movies
		# work on poster?
