# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 18:06:26 2019

@author: Daniel
"""

#from skimage.io import imsave
import numpy as np
from keras import backend as K

import os, cv2, gdal

os.environ["CUDA_VISIBLE_DEVICES"]="0" #Only make first GPU visible on multi-gpu setups
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

from skimage.transform import resize
from skimage.io import imread
from skimage.morphology import skeletonize

from aug_generators_dual import aug_validation


def preprocess(i, settings, metrics):
	read_image(i, settings, metrics)
	preprocess_image(settings, metrics)


def read_image(i, settings, metrics):
	"""Load image into settings. Only called once per raw image. Can switch between CALFIN and Mohajerani style
		 image inputs."""
	if settings['driver'] == 'calfin':	
		read_image_calfin(i, settings, metrics)
	elif settings['driver'] == 'calfin_on_mohajerani':
		read_image_mohajerani(i, settings, metrics)
	elif settings['driver'] == 'calfin_on_zhang':	
		read_image_calfin(i, settings, metrics)
	elif settings['driver'] == 'calfin_on_zhang_esa_cci':	
		read_image_calfin_shapefile_mask(i, settings, metrics)
	elif settings['driver'] == 'mohajerani_on_calfin':
		read_image_calfin(i, settings, metrics)
	elif settings['driver'] == 'mask_extractor':	
		read_image_calfin(i, settings, metrics)
	elif settings['driver'] == 'production':
		read_image_production(i, settings, metrics)
	else:
		raise Exception('Input driver must be "calfin" or "mohajerani"')


def preprocess_image(settings, metrics):
	"""Perform preprocessing on image loaded into settings. Can be called many times per image, based on number
		of subsets/fronts per raw image. Can switch between CALFIN and Mohajerani style image inputs."""
	if settings['driver'] == 'calfin':	
		preprocess_image_calfin(settings, metrics)
	elif settings['driver'] == 'calfin_on_mohajerani':	
		preprocess_image_mohajerani(settings, metrics)
	elif settings['driver'] == 'calfin_on_zhang':	
		preprocess_image_calfin(settings, metrics)
	elif settings['driver'] == 'calfin_on_zhang_esa_cci':	
		preprocess_image_calfin(settings, metrics)
	elif settings['driver'] == 'mohajerani_on_calfin':
		preprocess_image_calfin(settings, metrics)
	elif settings['driver'] == 'mask_extractor':	
		preprocess_image_calfin(settings, metrics)
	elif settings['driver'] == 'production':
		preprocess_image_production(settings, metrics)
	else:
		raise Exception('Input driver must be "calfin" or "mohajerani"')


def read_image_calfin(i, settings, metrics):
	"""Reads CALFIN style image inputs into memory. Also extracts resolution info from GeoTiff for error analysis."""
	validation_files = settings['validation_files']
	tif_source_path = settings['tif_source_path']
	fjord_boundaries_path = settings['fjord_boundaries_path']
	image_settings = settings['image_settings']
	date_index = settings['date_index']
	
	
	image_path = validation_files[i]
	image_dir = os.path.dirname(image_path) 
	image_name = os.path.basename(image_path)
	image_name_base = os.path.splitext(image_name)[0]
	image_name_base_parts = image_name_base.split('_')
	domain = image_name_base_parts[0]
	date = image_name_base_parts[date_index]
	year = date.split('-')[0]
	
	#initialize paths
	mask_path = os.path.join(image_dir, image_name_base + '_mask.png')
	tif_path = os.path.join(tif_source_path, domain, year, image_name_base + '.tif')
#	raw_save_path = os.path.join(save_path, image_name_base + '_raw.png')
#	mask_save_path = os.path.join(save_path, image_name_base + '_mask.png')
#	pred_save_path = os.path.join(save_path, image_name_base + '_pred.png')
	fjord_boundary_path = os.path.join(fjord_boundaries_path, domain + "_fjord_boundaries.png")
	
	#Read in raw/mask image pair
	img_3_uint8 = imread(image_path) #np.uint8 [0, 255]
	mask_uint8 = imread(mask_path) #np.uint8 [0, 255]
	fjord_boundary = imread(fjord_boundary_path) #np.uint8 [0, 255]
	if img_3_uint8.shape[2] != 3:
		img_3_uint8 = np.concatenate((img_3_uint8, img_3_uint8, img_3_uint8))
		
	#Detect if this image is supposed to have a front or not by seeing if its standard deviation is below threshold epsilon
	epsilon = 1e-4
	if np.nan_to_num(np.std(mask_uint8)) < epsilon: #and there is no front, it's a true negative
		settings['negative_image_names'].append(image_name_base)
#		print(image_name_base, 'is negative image')
		
	
	#Get bounds and transform vertices
	geotiff = gdal.Open(tif_path)
	geoTransform = geotiff.GetGeoTransform()
	meters_per_native_pixel = (np.abs(geoTransform[1]) + np.abs(geoTransform[5])) / 2
	resolution_1024 = (img_3_uint8.shape[0] + img_3_uint8.shape[1])/2
	resolution_native = (geotiff.RasterXSize + geotiff.RasterYSize) / 2
	meters_per_1024_pixel = resolution_native / resolution_1024 * meters_per_native_pixel
	
	image_settings['unprocessed_raw_image'] = img_3_uint8
	image_settings['unprocessed_mask_image'] = mask_uint8
	image_settings['unprocessed_fjord_boundary'] = fjord_boundary
	image_settings['resolution_1024'] = resolution_1024
	image_settings['meters_per_1024_pixel'] = meters_per_1024_pixel
	image_settings['image_name_base'] = image_name_base
	image_settings['domain'] = domain
	image_settings['year'] = int(year)
	image_settings['i'] = i


def preprocess_image_calfin(settings, metrics):
	"""Prepares images by resizing, slicing, and normalizing images."""
	#Initalize variables
	full_size = settings['full_size']
	image_settings = settings['image_settings']
	img_3_uint8 = image_settings['unprocessed_raw_image']
	mask_uint8 = image_settings['unprocessed_mask_image']
	fjord_boundary = image_settings['unprocessed_fjord_boundary']
	
	#Rescale mask while perserving continuity of edges
	img_3_f64 = resize(img_3_uint8, (full_size, full_size), preserve_range=True)  #np.float64 [0.0, 65535.0]
	mask_f64 = resize(mask_uint8, (full_size, full_size), order=0, preserve_range=True) #np.float64 [0.0, 65535.0]
	fjord_boundary_final_f32 = resize(fjord_boundary, (full_size, full_size), order=0, preserve_range=True) #np.float64 [0.0, 255.0]
	
	#Ensure correct scaling and no clipping of data values
	img_max = img_3_f64.max()
	img_min = img_3_f64.min()
	img_range = img_max - img_min
	if (img_max != 0.0 and img_range > 255.0):
		img_3_f64 = np.round(img_3_f64 / img_max * 255.0).astype(np.float32) #np.float32 [0, 65535.0]
	else:
		img_3_f64 = img_3_f64.astype(np.float32)
	img_final_f32 = img_3_f64 #np.float32 [0.0, 255.0]
	mask_f32 = (mask_f64 / mask_f64.max()).astype(np.float32)
	mask_uint8 = (mask_f32 * 255).astype(np.uint8)
			
	#Calculate edge from original resolution mask
	thickness = 3
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
	mask_edge_f64 = cv2.Canny(mask_uint8, 250, 255 * 2).astype('float64') #thresholds = Use diagonals to detect strong edges, then connect anything with at least a single edge
	mask_edge_f32 = cv2.dilate(mask_edge_f64, kernel, iterations = 1).astype(np.float32) #np.float32 [0.0, 255.0]
	mask_edge_f32 = mask_edge_f32 / mask_edge_f32.max()
	mask_final_f32 = np.stack((mask_edge_f32, mask_f32), axis = -1)
	
	image_settings['raw_image'] = img_final_f32
	image_settings['mask_image'] = mask_final_f32
	image_settings['fjord_boundary_final_f32'] = fjord_boundary_final_f32


def read_image_calfin_shapefile_mask(i, settings, metrics):
	"""Reads CALFIN style image inputs and converts Shapefile into standard masks. Also extracts resolution info from GeoTiff for error analysis."""
	validation_files = settings['validation_files']
	tif_source_path = settings['tif_source_path']
	fjord_boundaries_path = settings['fjord_boundaries_path']
	image_settings = settings['image_settings']
	date_index = settings['date_index']
	
	
	image_path = validation_files[i]
	image_dir = os.path.dirname(image_path) 
	image_name = os.path.basename(image_path)
	image_name_base = os.path.splitext(image_name)[0]
	image_name_base_parts = image_name_base.split('_')
	domain = image_name_base_parts[0]
	date = image_name_base_parts[date_index]
	year = date.split('-')[0]
	
	#initialize paths
	mask_path = os.path.join(image_dir, image_name_base + '_mask.png')
	tif_path = os.path.join(tif_source_path, domain, year, image_name_base + '.tif')
#	raw_save_path = os.path.join(save_path, image_name_base + '_raw.png')
#	mask_save_path = os.path.join(save_path, image_name_base + '_mask.png')
#	pred_save_path = os.path.join(save_path, image_name_base + '_pred.png')
	fjord_boundary_path = os.path.join(fjord_boundaries_path, domain + "_fjord_boundaries.png")
	
	#Read in raw/mask image pair
	img_3_uint8 = imread(image_path) #np.uint8 [0, 255]
	mask_uint8 = imread(mask_path) #np.uint8 [0, 255]
	fjord_boundary = imread(fjord_boundary_path) #np.uint8 [0, 255]
	if img_3_uint8.shape[2] != 3:
		img_3_uint8 = np.concatenate((img_3_uint8, img_3_uint8, img_3_uint8))
		
	#Detect if this image is supposed to have a front or not by seeing if its standard deviation is below threshold epsilon
	epsilon = 1e-4
	if np.nan_to_num(np.std(mask_uint8)) < epsilon: #and there is no front, it's a true negative
		settings['negative_image_names'].append(image_name_base)
#		print(image_name_base, 'is negative image')
		
	
	#Get bounds and transform vertices
	geotiff = gdal.Open(tif_path)
	geoTransform = geotiff.GetGeoTransform()
	meters_per_native_pixel = (np.abs(geoTransform[1]) + np.abs(geoTransform[5])) / 2
	resolution_1024 = (img_3_uint8.shape[0] + img_3_uint8.shape[1])/2
	resolution_native = (geotiff.RasterXSize + geotiff.RasterYSize) / 2
	meters_per_1024_pixel = resolution_native / resolution_1024 * meters_per_native_pixel
	
	image_settings['unprocessed_raw_image'] = img_3_uint8
	image_settings['unprocessed_mask_image'] = mask_uint8
	image_settings['unprocessed_fjord_boundary'] = fjord_boundary
	image_settings['resolution_1024'] = resolution_1024
	image_settings['meters_per_1024_pixel'] = meters_per_1024_pixel
	image_settings['image_name_base'] = image_name_base
	image_settings['domain'] = domain
	image_settings['year'] = int(year)
	image_settings['i'] = i


def read_image_mohajerani(i, settings, metrics):
	"""Reads Mohajerani style image inputs into memory. Uses static scaling for Helheim error analysis."""
	validation_files = settings['validation_files']
	fjord_boundaries_path = settings['fjord_boundaries_path']
	image_settings = settings['image_settings']
	date_index = settings['date_index']
	
	image_path = validation_files[i]
	image_dir = os.path.dirname(image_path) 
	image_name = os.path.basename(image_path)
	image_name_base = os.path.splitext(image_name)[0]
	image_name_base_parts = image_name_base.split('_')
	domain = 'Helheim'
	date = image_name_base_parts[date_index]
	year = date[0:4]
	
	#initialize paths
	mask_path = os.path.join(image_dir, image_name_base.replace('Subset', 'Front') + '.png')
#	raw_save_path = os.path.join(save_path, image_name_base + '_raw.png')
#	mask_save_path = os.path.join(save_path, image_name_base + '_mask.png')
#	pred_save_path = os.path.join(save_path, image_name_base + '_pred.png')
	fjord_boundary_path = r"D:\Daniel\Documents\Github\CALFIN Repo Intercomp\training\data\fjord_boundaries\Helheim_fjord_boundaries.png"
	
	#Read in raw/mask image pair
	img_3_uint8 = imread(image_path) #np.uint8 [0, 255]
	mask_uint8 = imread(mask_path) #np.uint8 [0, 255]
	fjord_boundary = imread(fjord_boundary_path) #np.uint8 [0, 255]
	if img_3_uint8.shape[2] != 3:
		img_3_uint8 = np.concatenate((img_3_uint8, img_3_uint8, img_3_uint8))
	
	#Retrieve pixel to meter scaling ratio
	meters_per_1024_pixel = 96.3 / 1.97
	image_settings['unprocessed_raw_image'] = img_3_uint8
	image_settings['unprocessed_mask_image'] = mask_uint8
	image_settings['unprocessed_fjord_boundary'] = fjord_boundary
	image_settings['resolution_1024'] = 256
	image_settings['meters_per_1024_pixel'] = meters_per_1024_pixel
	image_settings['image_name_base'] = image_name_base
	image_settings['domain'] = domain
	image_settings['year'] = int(year)
	image_settings['i'] = i


def preprocess_image_mohajerani(settings, metrics):
	"""Prepares images by resizing, slicing, and normalizing images. Resizing adds in a dilation and skeletonization
		operations to preserve continuity of Mohajerani style edge masks."""
	#Initalize variables
	full_size = settings['full_size']
	image_settings = settings['image_settings']
	img_3_uint8 = image_settings['unprocessed_raw_image']
	mask_uint8 = image_settings['unprocessed_mask_image']
	fjord_boundary = image_settings['unprocessed_fjord_boundary']
	
	#Rescale mask while perserving continuity of edges
	thickness = 3
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
	mask_uint8_dilated = cv2.dilate((255 - mask_uint8).astype('float64'), kernel, iterations = 1).astype(np.uint8) #np.uint8 [0, 255]
	
	augs = aug_validation(img_size=full_size)
	dat = augs(image=img_3_uint8, mask=mask_uint8_dilated)
	img_3_aug_f32 = dat['image'].astype('float32') #np.float32 [0.0, 255.0]
	mask_aug_f32 = dat['mask'].astype('float32') #np.float32 [0.0, 255.0]
	
	#Reskeletonize mask once at desired resolution to ensure 3-pixel wide target
	mask_rescaled_f32 = np.where(mask_aug_f32[:,:,0] > 127.0, 1.0, 0.0)
	mask_edge_f32 = skeletonize(mask_rescaled_f32) #np.float32 [0.0, 1.0]
	mask_edge_f32 = cv2.dilate(mask_edge_f32.astype(np.float64), kernel, iterations = 1).astype(np.float32) #np.float32 [0.0, 255.0]
	mask_edge_f32 = mask_edge_f32 / mask_edge_f32.max()	
	mask_final_f32 = np.stack((mask_edge_f32, np.zeros(mask_edge_f32.shape)), axis = -1)
	
	#Resize fjord boundary
	dat = augs(image=fjord_boundary)
	fjord_boundary_final_f32 = dat['image'].astype('float32') #np.float32 [0.0, 1.0]
	
	#Ensure correct scaling and no clipping of data values
	img_max = img_3_aug_f32.max()
	img_min = img_3_aug_f32.min()
	img_range = img_max - img_min
	if (img_max != 0.0 and img_range > 255.0):
		img_3_aug_f32 = np.round(img_3_aug_f32 / img_max * 255.0) #np.float32 [0, 65535.0]
	else:
		img_3_aug_f32 = img_3_aug_f32.astype(np.float32)
	img_final_f32 = img_3_aug_f32.astype(np.float32) #np.float32 [0.0, 255.0]
	
	image_settings['raw_image'] = img_final_f32
	image_settings['mask_image'] = mask_final_f32
	image_settings['fjord_boundary_final_f32'] = fjord_boundary_final_f32


def read_image_production(i, settings, metrics):
	"""Reads CALFIN style image inputs into memory. Also extracts resolution info from GeoTiff for error analysis."""
	validation_files = settings['validation_files']
	tif_source_path = settings['tif_source_path']
	fjord_boundaries_path = settings['fjord_boundaries_path']
	image_settings = settings['image_settings']
	date_index = settings['date_index']
	
	
	image_path = validation_files[i]
	image_name = os.path.basename(image_path)
	image_name_base = os.path.splitext(image_name)[0]
	image_name_base_parts = image_name_base.split('_')
	domain = image_name_base_parts[0]
	date = image_name_base_parts[date_index]
	year = date.split('-')[0]
	
	#initialize paths
	tif_path = os.path.join(tif_source_path, domain, year, image_name_base + '.tif')
	fjord_boundary_path = os.path.join(fjord_boundaries_path, domain + "_fjord_boundaries.png")
	
	#Read in raw/mask image pair
	img_3_uint8 = imread(image_path) #np.uint8 [0, 255]
	fjord_boundary = imread(fjord_boundary_path) #np.uint8 [0, 255]
	if img_3_uint8.shape[2] != 3:
		img_3_uint8 = np.concatenate((img_3_uint8, img_3_uint8, img_3_uint8))
	
	img_3_uint8 = resize(img_3_uint8, (fjord_boundary.shape[0], fjord_boundary.shape[1]), preserve_range=True).astype(np.uint8)  #np.float64 [0.0, 65535.0]
	
	#Get bounds and transform vertices
	print(tif_path, img_3_uint8.max())
	geotiff = gdal.Open(tif_path)
	geoTransform = geotiff.GetGeoTransform()
	meters_per_native_pixel = (np.abs(geoTransform[1]) + np.abs(geoTransform[5])) / 2
	resolution_1024 = (img_3_uint8.shape[0] + img_3_uint8.shape[1])/2
	resolution_native = (geotiff.RasterXSize + geotiff.RasterYSize) / 2
	meters_per_1024_pixel = resolution_native / resolution_1024 * meters_per_native_pixel
	
	image_settings['unprocessed_raw_image'] = img_3_uint8
	image_settings['unprocessed_fjord_boundary'] = fjord_boundary
	image_settings['resolution_1024'] = resolution_1024
	image_settings['meters_per_1024_pixel'] = meters_per_1024_pixel
	image_settings['image_name_base'] = image_name_base
	image_settings['domain'] = domain
	image_settings['year'] = int(year)
	image_settings['i'] = i


def preprocess_image_production(settings, metrics):
	"""Prepares images by resizing, slicing, and normalizing images."""
	#Initalize variables
	full_size = settings['full_size']
	image_settings = settings['image_settings']
	img_3_uint8 = image_settings['unprocessed_raw_image']
	fjord_boundary = image_settings['unprocessed_fjord_boundary']
	
	#Rescale mask while perserving continuity of edges
	img_3_f64 = resize(img_3_uint8, (full_size, full_size), preserve_range=True)  #np.float64 [0.0, 65535.0]
	fjord_boundary_final_f32 = resize(fjord_boundary, (full_size, full_size), order=0, preserve_range=True) #np.float64 [0.0, 255.0]
	
	#Ensure correct scaling and no clipping of data values
	img_max = img_3_f64.max()
	img_min = img_3_f64.min()
	img_range = img_max - img_min
	if (img_max != 0.0 and img_range > 255.0):
		img_3_f64 = np.round(img_3_f64 / img_max * 255.0).astype(np.float32) #np.float32 [0, 65535.0]
	else:
		img_3_f64 = img_3_f64.astype(np.float32)
	img_final_f32 = img_3_f64 #np.float32 [0.0, 255.0]
	
	image_settings['raw_image'] = img_final_f32
	image_settings['fjord_boundary_final_f32'] = fjord_boundary_final_f32


