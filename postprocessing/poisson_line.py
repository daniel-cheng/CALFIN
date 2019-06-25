# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:28:22 2019

@author: Daniel
"""
import numpy as np
import os, glob, cv2
import matplotlib.pyplot as plt
from scipy.misc import imresize
from scipy.ndimage import distance_transform_edt
from redistribute_points import redistribute_points
from skimage.morphology import skeletonize
import meshcut
from scipy import ndimage
from ordered_line_from_unordered_points import ordered_line_from_unordered_points, ordered_line_from_unordered_points_tree
from PIL import Image


steps = ['1', '2', '3', '4']
#steps = ['2', '3']
#steps = ['3']
#steps = ['4']
steps = ['2', '4']
steps = ['A']

#Load domain to process
if '1' in steps:
	def landsat_sort(file_path):
		return file_path.split(os.path.sep)[-1].split('_')[3]
	
	def get_paths(root, domain):
		raw_paths = glob.glob(os.path.join(root, domain, '*_raw.png'))
		raw_paths.sort(key=landsat_sort)
		
		mask_paths = glob.glob(os.path.join(root, domain, '*_pred.png'))
		mask_paths.sort(key=landsat_sort)
		
		return raw_paths, mask_paths
	
	def step_1(root, domain):
		return get_paths(root, domain)
		
	raw_paths = glob.glob(r'D:\Daniel\Documents\Github\CALFIN Repo\postprocessing\testing\*[0-9].png')
	raw_paths_calfin = glob.glob(r'D:\Daniel\Documents\Github\CALFIN Repo\postprocessing\testing_calfin\*_raw.png')
	raw_paths_all = raw_paths + raw_paths_calfin
	raw_paths.sort(key=landsat_sort)
	raw_paths_calfin.sort(key=landsat_sort)
	raw_paths_all.sort(key=landsat_sort)
	
	mask_paths = glob.glob(r'D:\Daniel\Documents\Github\CALFIN Repo\postprocessing\testing\*_mask.png')
	mask_paths_calfin = glob.glob(r'D:\Daniel\Documents\Github\CALFIN Repo\postprocessing\testing_calfin\*_pred.png')
	mask_paths_all = mask_paths + mask_paths_calfin
	mask_paths.sort(key=landsat_sort)
	mask_paths_calfin.sort(key=landsat_sort)
	mask_paths_all.sort(key=landsat_sort)
	
#	first_mask_path = mask_paths[0]
#	first_mask_img = cv2.imread(first_mask_path, 0)
	

	

if '2' in steps:
	def extract_front_indicators(raw_path, mask_path, index, resolution):
		width, height = resolution
		#Extract known masks
		raw_img = np.array(Image.fromarray(cv2.imread(raw_path, 0)).resize((width, height), Image.BICUBIC)).astype('uint8')
		mask_img = np.array(Image.fromarray(cv2.imread(mask_path, 0)).resize((width, height), Image.BICUBIC)).astype('uint8')
		raw_rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
#		edges = cv2.Canny(mask_img,100,200)
		edge_bianry = np.where(mask_img > 127, 1, 0)
		skeleton = skeletonize(edge_bianry)
		front_pixels = np.nonzero(skeleton)
		
		#Require a minimum of 2 points
		if len(front_pixels[0]) < 2:
			return None
		dpi = 80
		fig = plt.figure(index, figsize=(width/dpi, height/dpi), dpi=dpi)
#		fig.tight_layout(pad=0)
#		fig.add_subplot(111)

		plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
		            hspace = 0, wspace = 0)
#		plt.margins(0,0)
#		ax.xaxis.set_major_locator(plt.NullLocator())
#		ax.yaxis.set_major_locator(plt.NullL1ocator())

		plt.imshow(raw_rgb_img)
		ax = plt.gca()
		ax.axis('tight')
		ax.axis('off')
		ax.set_xlim(0.0, width)
		ax.set_ylim(height, 0.0)
		front_line = np.array(ordered_line_from_unordered_points_tree(front_pixels, raw_img.shape))
		number_of_points = front_line.shape[1]
		front_normals = np.zeros((2, number_of_points))
		
		#Require a minimum of 2 points
		if len(front_line) < 2:
			return None
		i = 0
		p1 = front_line[:, i]
		p2 = front_line[:, i + 1]
		d21 = p2 - p1
		n21 = np.array([-d21[1], d21[0]])
		n1 = n21
		n1 = n1 / np.linalg.norm(n1)
		front_normals[:,i] = n1
		
		i = number_of_points - 1
		p0 = front_line[:, i - 1]
		p1 = front_line[:, i]
		d10 = p1 - p0
		n10 = np.array([-d10[1], d10[0]])
		n1 = n10
		n1 = n1 / np.linalg.norm(n1)
		front_normals[:,i] = n1
		
		for i in range(1, number_of_points - 1):
			p0 = front_line[:, i - 1]
			p1 = front_line[:, i]
			p2 = front_line[:, i + 1]
			d10 = p1 - p0
			d21 = p2 - p1
			n10 = np.array([d10[1], -d10[0]])
			n21 = np.array([d21[1], -d21[0]])
			n1 = n10 + n21
			n1 = n1 / np.linalg.norm(n1)
			front_normals[:,i] = n1
		
#		raw_rgb_img = np.zeros((512, 512, 3)) + 0.5
#		for i in range(len(front_line[0])):
#			raw_rgb_img[front_line[0, i], front_line[1, i]] = [front_normals[0,i] / 2 + 0.5, front_normals[1,i] / 2 + 0.5, 0.5]
		
		plt.show()
		fig.canvas.draw()
		data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
		overlay = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

		return overlay, front_line, front_normals
	
#	base_date = 
	overlays = []
	fronts_lines = []
	fronts_normals = []
	processed_paths = []
	
	#Determine resolution
	img_sizes = []
	for i in range(0, len(mask_paths_calfin)):
		img = Image.open(raw_paths_calfin)
		img_sizes.append(img.size)
	img_sizes = np.array(img_sizes)
	resolution = np.median(img_sizes, axis=1)
	
	for i in range(0, 5):
		raw_path = raw_paths_calfin[i]
		mask_path = mask_paths_calfin[i]
		print(i, mask_path)
		
		result = extract_front_indicators(raw_path, mask_path, i, resolution)

		if result != None:
			overlay = result[0]
			front_lines = result[1]
			front_normals = result[2]
			overlays.append(overlay)
			fronts_lines.append(front_lines)
			fronts_normals.append(front_normals)
			processed_paths.append(raw_path)
			
			
#			plt.figure(1)
#			plt.imshow(overlay)
#			plt.show()
if '3' in steps:
	start_points = []
	end_points = []
	for i in range(len(fronts_lines)):
		start_points.append(fronts_lines[i][:,0])
		end_points.append(fronts_lines[i][:,-1])
	start_points = np.array(start_points)
	end_points = np.array(end_points)
	median_start_point = np.median(start_points, axis=0)
	median_end_point = np.median(end_points, axis=0)
	
if '4' in steps:
	from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
	
	def make_video(images, images_paths, output_path, fps=1.0, size=None,
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
		fourcc = VideoWriter_fourcc(*'MJPG')
		vid = None
		for i in range(len(images_paths)):
			image_path = images_paths[i]
			img = images[i]
			if vid is None:
				if size is None:
					size = img.shape[1], img.shape[0]
				vid = VideoWriter(video_name, fourcc, float(fps), size, is_color)
			if size[0] != img.shape[1] or size[1] != img.shape[0]:
				img = resize(img, size)
			
			widthScale = img.shape[0] / 500 #(font is scaled for Upernavik, which is 500 pixels wide)
			heightScale = img.shape[1] / 500 #(font is scaled for Upernavik, which is 500 pixels wide)
		
			bottomLeftCornerOfText = (4, img.shape[1] - int(34 * heightScale))
			font = cv2.FONT_HERSHEY_SIMPLEX
			fontScale = 1 * widthScale
			fontColor = (0, 192, 216)
			fontColorBorder = (0, 0, 0)
			lineType = 2
			thickness = 2
			thicknessBorder = 8
			
			name_split = image_path.split('\\')[-1].split('_')
			date = name_split[3]
					
			#text_width, text_height = cv2.getTextSize(date, font, fontScale, lineType)[0]
			cv2.putText(img, date, bottomLeftCornerOfText, font, fontScale, fontColorBorder, thickness=thicknessBorder, lineType=lineType)
			cv2.putText(img, date, bottomLeftCornerOfText, font, fontScale, fontColor, thickness=thickness, lineType=lineType)
			img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
			vid.write(img)
		vid.release()
		return
	
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
				
	video_output_path = "Upernavik"
	make_video(overlays, processed_paths, video_output_path, fps=2.0)
	
if 'A' in steps:
	root_path = r'D:\Daniel\Pictures\CALFIN Imagery\landsat_preds'
	for domain in os.listdir(root_path):
		 raw_paths, mask_paths = step_1(root, domain)
	