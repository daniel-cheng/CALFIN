# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:28:22 2019

@author: Daniel
"""
import numpy as np
import os, glob, cv2, shutil
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from redistribute_points import redistribute_points
from skimage.morphology import skeletonize
import meshcut
from scipy import ndimage
from ordered_line_from_unordered_points import ordered_line_from_unordered_points_tree
from PIL import Image
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
from collections import defaultdict
plt.ioff()

steps = ['1', '2', '3', '4']
#steps = ['2']
#steps = ['3']
steps = ['4']
steps = ['2', '4']
steps = ['A']


"""Load domain to process"""
def landsat_sort(file_path):
	"""Sorting key function derives date from landsat file path."""
	return file_path.split(os.path.sep)[-1].split('_')[3]

def get_paths(root, domain, suffix):
	"""Gets a sorted list of file paths from the specified root directory, subdirectory, and file name regex/glob pattern."""
	paths = glob.glob(os.path.join(root, domain, '*' + suffix + '.png'))
	paths.sort(key=landsat_sort)
	return paths

def step_1(root, domain):
	"""Gets the sorted lists of files for the specified root and subdirectory."""
	raw_paths = get_paths(root, domain, '_raw')
	mask_paths =  get_paths(root, domain, '_pred')
	return raw_paths, mask_paths

if '1' in steps and __name__ == "__main__":
	#For testing - performs step 1 with default parameters.
	root = r'D:\Daniel\Documents\Github\CALFIN Repo\postprocessing'
	domain = 'testing_calfin'
	raw_paths, mask_paths = step_1(root, domain)
	


"""Front Extraction"""
def extract_front_indicators(raw_path, mask_path, index, resolution):
	"""Extracts an ordered polyline from the processed mask. Also returns an overlay of the extracted polyline and the raw image. Draws to the indexed figure and specified resolution."""
	width = resolution[0]
	height = resolution[1]
	minimum_points = 4
	
	#Extract known masks
	raw_img = np.array(Image.fromarray(cv2.imread(raw_path, 0)).resize((width, height), Image.BICUBIC)).astype('uint8')
	mask_img = np.array(Image.fromarray(cv2.imread(mask_path, 0)).resize((width, height), Image.BICUBIC)).astype('uint8')
	raw_rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
	#edges = cv2.Canny(mask_img,100,200)
	edge_bianry = np.where(mask_img > 127, 1, 0)
	skeleton = skeletonize(edge_bianry)
	front_pixels = np.nonzero(skeleton)
	
	#Require a minimum number of points
	if len(front_pixels[0]) < minimum_points:
		return None
	
	#Prepare figure for direct image output/saving
	dpi = 80
	fig = plt.figure(index, figsize=(width/dpi, height/dpi), dpi=dpi)
	plt.clf()
	plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
	            hspace = 0, wspace = 0)
	plt.imshow(raw_rgb_img)
	ax = plt.gca()
	ax.axis('tight')
	ax.axis('off')
	ax.set_xlim(0.0, width)
	ax.set_ylim(height, 0.0)
	
	#Perform mask to polyline extraction.
	front_line = np.array(ordered_line_from_unordered_points_tree(front_pixels, raw_img.shape, minimum_points))
	number_of_points = front_line.shape[1]
	front_normals = np.zeros((2, number_of_points))
	
	#Require a minimum number of points
	if len(front_line[0]) < minimum_points:
		return None
	
	#Calculate normals for endpoints.
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
	
	#Calculate normals for all other points.
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
	
	#Draw normals over raw image.
#		raw_rgb_img = np.zeros((512, 512, 3)) + 0.5
#		for i in range(len(front_line[0])):
#			raw_rgb_img[front_line[0, i], front_line[1, i]] = [front_normals[0,i] / 2 + 0.5, front_normals[1,i] / 2 + 0.5, 0.5]
	
	#Save figure as image matrix.
	fig.canvas.draw()
	data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
	overlay = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

	return overlay, front_line, front_normals


def step_2(raw_paths, mask_paths, reprocessing_path, source_root_path):
	"""Extracts image overlays, polyline fronts, polyline normals, and valid return raw paths for all images in paths."""
	#base_date = 
	overlays = []
	fronts_lines = []
	fronts_normals = []
	processed_paths = []
	
	#Determine resolution
	img_sizes = []
	for i in range(0, len(mask_paths)):
		img = Image.open(raw_paths[i])
		img_sizes.append(img.size)
	img_sizes = np.array(img_sizes)
	resolution = np.median(img_sizes, axis=0)
	
	for i in range(0, len(mask_paths)):
#	for i in range(50, 51):
		raw_path = raw_paths[i]
		mask_path = mask_paths[i]
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
		else:
			name_split = raw_path.split('\\')[-1].split('_')
			domain = name_split[0]
			basename = '_'.join(name_split[0:-1])
			raw_name = basename + '.png'
			mask_name = basename + '_mask.png'
			source_path = os.path.join(source_root_path, domain)
			domain_path = os.path.join(reprocessing_path, domain)
			if not os.path.exists(domain_path):
				os.mkdir(domain_path)
			shutil.copy2(os.path.join(source_path, raw_name), os.path.join(domain_path, raw_name))
			shutil.copy2(os.path.join(source_path, raw_name), os.path.join(domain_path, mask_name))
	return overlays, fronts_lines, fronts_normals, processed_paths

if '2' in steps and __name__ == "__main__":
	#For testing - performs step 2 with default parameters.
	overlays, fronts_lines, fronts_normals, processed_paths = step_2(raw_paths, mask_paths)
	
	
"""Endpoint determination and Shapefile output"""
def step_3(fronts_lines, processed_paths, reprocessing_path, source_root_path):
	"""Determines endpoints by finding the median"""
	start_points = []
	end_points = []
	for i in range(len(fronts_lines)):
		start_points.append(fronts_lines[i][:,0])
		end_points.append(fronts_lines[i][:,-1])
	start_points = np.array(start_points)
	end_points = np.array(end_points)
	median_start_point = np.median(start_points, axis=0)
	median_end_point = np.median(end_points, axis=0)
	
	#Reprocess images that do not conform
	for i in range(len(start_points)):
		start_diff = np.linalg.norm(median_start_point - start_points[i])
		end_diff = np.linalg.norm(median_end_point - end_points[i])
		if start_diff > 10 or end_diff > 10:
			raw_path = processed_paths[i]
			name_split = raw_path.split('\\')[-1].split('_')
			domain = name_split[0]
			basename = '_'.join(name_split[0:-1])
			raw_name = basename + '.png'
			mask_name = basename + '_mask.png'
			source_path = os.path.join(source_root_path, domain)
			domain_path = os.path.join(reprocessing_path, domain)
			if not os.path.exists(domain_path):
				os.mkdir(domain_path)
			shutil.copy2(os.path.join(source_path, raw_name), os.path.join(domain_path, raw_name))
			shutil.copy2(os.path.join(source_path, raw_name), os.path.join(domain_path, mask_name))
	
	return median_start_point, median_end_point

if '3' in steps and __name__ == "__main__":
	#For testing - performs step 3 with default parameters.
	median_start_point, median_end_point = step_3(fronts_lines)
	

"""Visualize/Movie output"""
def get_date(image_path):
	"""Takes in a Landsat image path, and returns date string as well as individual year, month, and day strings."""
	name_split = image_path.split('\\')[-1].split('_')
	date = name_split[3]
	date_parts = date.split('-')
	year = date_parts[0]
	month = date_parts[1]
	day = date_parts[2]
	return date, year, month, day

def make_video(images, images_paths, output_path, fps=1.0, size=None, is_color=True, format='MJPG'):
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
		
		date, year, month, day = get_date(image_path)
				
		#text_width, text_height = cv2.getTextSize(date, font, fontScale, lineType)[0]
		cv2.putText(img, date, bottomLeftCornerOfText, font, fontScale, fontColorBorder, thickness=thicknessBorder, lineType=lineType)
		cv2.putText(img, date, bottomLeftCornerOfText, font, fontScale, fontColor, thickness=thickness, lineType=lineType)
		img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
		vid.write(img)
	vid.release()
	return

def make_calendar(processed_paths):
	"""Generates year/month bins where measurements exist for given domain."""
	max_date, max_year, max_month, max_day = get_date(processed_paths[-1])
	min_year = 1972 #Landsat start
	max_year = int(max_year)
	year_range = max_year - min_year + 1
	yearly_bins = np.zeros((year_range, 1))
	monthly_bins = np.zeros((year_range, 12))
	
	#Populate year/month bins
	for i in range(len(processed_paths)):
		date, year, month, day = get_date(processed_paths[i])
		yearly_bins[int(year) - min_year, 0] += 1
		monthly_bins[int(year) - min_year, int(month)] += 1
			
	return yearly_bins, monthly_bins

def step_4(overlays, processed_paths, output_path, name):
	file_path = os.path.join(output_path, name)
	yearly_bins, monthly_bins = make_calendar(processed_paths)
	make_video(overlays, processed_paths, file_path, fps=1.0)
	return yearly_bins, monthly_bins

if '4' in steps and __name__ == "__main__":
	#For testing - performs step 4 with default parameters.
	output_path = '.'
	name = "Upernavik-NE"
	yearly_bins, monthly_bins = step_4(overlays, processed_paths, output_path, name)
	
	
"""Perform steps 1-4 for all domains"""
def save_domain(domain, raw_paths, mask_paths, overlays, fronts_lines, fronts_normals, processed_paths, yearly_bins, monthly_bins):
	"""Saves all domain specific global variables to npz file."""
	np.savez_compressed(domain + '.npz', 
	   raw_paths=raw_paths, 
	   mask_paths=mask_paths, 
	   overlays=overlays, 
	   fronts_lines=fronts_lines, 
	   fronts_normals=fronts_normals, 
	   processed_paths=processed_paths,
	   yearly_bins=yearly_bins,
	   monthly_bins=monthly_bins)

def load_domain(domain):
	"""Loads all domain specific global variables from npz file."""
	data = np.load(domain + '.npz')
	raw_paths = data['raw_paths'] 
	mask_paths = data['mask_paths'] 
	overlays = data['overlays'] 
	fronts_lines = data['fronts_lines'] 
	fronts_normals = data['fronts_normals'] 
	processed_paths = data['processed_paths']
	yearly_bins = data['yearly_bins']
	monthly_bins = data['monthly_bins']

if 'A' in steps and __name__ == "__main__":
	#For testing - performs all steps with default parameters.
	root_path = r'D:\Daniel\Pictures\CALFIN Imagery\landsat_preds'
	output_path = r'D:\Daniel\Documents\Github\CALFIN Repo\postprocessing\videos'
	reprocessing_path = r'D:\Daniel\Documents\Github\CALFIN Repo\reprocessing\images_full'
	source_root_path = r'D:\Daniel\Pictures\CALFIN Imagery\test_full'
	yearly_bins_all = dict()
	monthly_bins_all = dict()
	for domain in os.listdir(root_path):
		try:
			raw_paths, mask_paths = step_1(root_path, domain)
			overlays, fronts_lines, fronts_normals, processed_paths = step_2(raw_paths, mask_paths, reprocessing_path, source_root_path)
			median_start_point, median_end_point = step_3(fronts_lines, processed_paths, reprocessing_path, source_root_path)
			yearly_bins, monthly_bins = step_4(overlays, processed_paths, output_path, domain)
			yearly_bins_all[domain] = yearly_bins
			monthly_bins_all[domain] = monthly_bins
#			save_domain(domain, raw_paths, mask_paths, overlays, fronts_lines, fronts_normals, processed_paths, yearly_bins, monthly_bins)
#			break
		except Exception as e:
			print('Error with domain:', domain, '-', e)
	np.savez_compressed('time-bins.npz', yearly_bins_all=yearly_bins_all, monthly_bins_all=monthly_bins_all)
	
	#automatically save all images that don't have detections, and create masks
