# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 18:06:26 2019

@author: Daniel
"""

#from skimage.io import imsave
import numpy as np
from keras import backend as K

import sys, os, cv2, gdal
sys.path.insert(1, '../training/keras-deeplab-v3-plus')
sys.path.insert(2, '../training')
from model_cfm_dual_wide_x65 import Deeplabv3
from AdamAccumulate import AdamAccumulate

os.environ["CUDA_VISIBLE_DEVICES"]="0" #Only make first GPU visible on multi-gpu setups
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

from skimage.transform import resize
from skimage.io import imread
from scipy.spatial import distance

from aug_generators_dual import create_unaugmented_data_patches_from_rgb_image


def deviation(gt, pr, smooth=1e-6, per_image=True):
	mismatch = np.sum(np.abs(gt[:,:,:,1] - pr[:,:,:,1]), axis=(1, 2)) #(B)
	length = np.sum(pr[:,:,:,0], axis=(1, 2)) #(B)
	deviation = mismatch / (length + smooth) #(B)
	mean_deviation = np.mean(deviation) * 3.0 #- (account for line thickness of 3 at 224)
	return mean_deviation


def calculate_mean_deviation(pred, mask):
	"""Calculates mean deviation between two lines represented by nonzero pixels in pred and mask"""
	#Generate Nx2 matrix of pixels that represent the front
	x1, y1 = np.nonzero(pred)
	x2, y2 = np.nonzero(mask)
	pred_coords = np.array(list(zip(x1, y1)))
	mask_coords = np.array(list(zip(x2, y2)))
	
	#Return NaN if front is not detected in either pred or mask
	if pred_coords.shape[0] == 0 or mask_coords.shape[0] == 0:
		return np.nan, np.array([])
	
	#Generate the pairwise distances between each point and the closest point in the other array
	distances1 = distance.cdist(pred_coords, mask_coords).min(axis=1)
	distances2 = distance.cdist(mask_coords, pred_coords).min(axis=1)
	distances = np.concatenate((distances1, distances2))
	
	#Calculate the average distance between each point and the closest point in the other array
	mean_deviation = np.mean(distances)
	return mean_deviation, distances

def calculate_edge_iou(gt, pr, smooth=1e-6, per_image=True):
	intersection = np.sum(gt[:,:,:] * pr[:,:,:], axis=(1, 2)) #(B)
	union = np.sum(gt[:,:,:] + pr[:,:,:] >= 1.0, axis=(1, 2)) #(B)
	iou_score = intersection / (union + smooth) #(B)
	mean_iou_score = np.mean(iou_score) #- (account for line thickness of 3 at 224)
	return mean_iou_score
	
def predict(model, img_3_uint8, mask_uint8, fjord_boundary, pred_norm_image, full_size, img_size, stride):
	"""Takes in a neural network model, input image, mask, fjord boundary mask, and windowded normalization image to create prediction output.
		Uses the full original size, image patch size, and stride legnth as additional variables."""
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
	mask_uint8 = (mask_f64 / mask_f64.max() * 255).astype(np.uint8)
			
	#Calculate edge from original resolution mask
	thickness = 3
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
	mask_edge_f64 = cv2.Canny(mask_uint8, 250, 255 * 2).astype('float64') #thresholds = Use diagonals to detect strong edges, then connect anything with at least a single edge
	mask_edge_f32 = cv2.dilate(mask_edge_f64, kernel, iterations = 1).astype(np.float32) #np.float32 [0.0, 255.0]
	mask_final_f32 = mask_edge_f32 / mask_edge_f32.max()					  
	
	#Generate image patches (test time augmentation) to ensure confident predictions
	patches = create_unaugmented_data_patches_from_rgb_image(img_final_f32, None, window_shape=(img_size, img_size, 3), stride=stride)
	
	#Predict results
	results = model.predict(patches, batch_size=16, verbose=1)
	
	#Initilize outputs
	raw_image_final_f32 = img_final_f32 / 255.0
	pred_image = np.zeros((full_size, full_size, 3))
	
	#Reassemble original resolution image by each 3x3 set of overlapping windows.
	strides = int((full_size - img_size) / stride + 1) #(256-224 / 16 + 1) = 3
	for x in range(strides):
		for y in range(strides):
			x_start = x * stride
			x_end = x_start + img_size
			y_start = y * stride
			y_end = y_start + img_size
			
			pred_patch = results[x*strides + y,:,:,0:2]
			pred_image[x_start:x_end, y_start:y_end, 0:2] += pred_patch
			
	#Normalize output by dividing by number of patches each pixel is included in
	pred_image = np.nan_to_num(pred_image)
	pred_image_final_f32 = (pred_image / pred_norm_image).astype(np.float32)
	
	return raw_image_final_f32, pred_image_final_f32, mask_final_f32, fjord_boundary_final_f32


def compile_model(img_size):
	"""Compile the CALFIN Neural Network model and loads pretrained weights."""
	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)
	img_shape = (img_size, img_size, 3)
	model = Deeplabv3(input_shape=img_shape, classes=16, OS=16, backbone='xception', weights=None)
	
	model.compile(optimizer=AdamAccumulate(lr=1e-4, accum_iters=2))
	model.summary()
	model.load_weights('../training/cfm_weights_patched_dual_wide_x65_224_e65_iou0.5136.h5')
	
	return model

def process(i, validation_files, settings, metrics):
	model = settings['model'] 
	pred_norm_image = settings['pred_norm_image'] 
	full_size = settings['full_size']
	img_size = settings['img_size']
	stride = settings['stride']
	tif_source_path = settings['tif_source_path']
	fjord_boundaries_path = settings['fjord_boundaries_path']
	image_settings = settings['image_settings']
	
	image_path = validation_files[i]
	image_dir = os.path.dirname(image_path) 
	image_name = os.path.basename(image_path)
	image_name_base = os.path.splitext(image_name)[0]
	image_name_base_parts = image_name_base.split('_')
	domain = image_name_base_parts[0]
	date = image_name_base_parts[3]
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
	
	#Retrieve pixel to meter scaling ratiio
#			if domain not in domain_scalings:
	geotiff = gdal.Open(tif_path)

	#Get bounds
	geoTransform = geotiff.GetGeoTransform()
#	xMin = geoTransform[0]
#	yMax = geoTransform[3]
#	xMax = xMin + geoTransform[1] * geotiff.RasterXSize
#	yMin = yMax + geoTransform[5] * geotiff.RasterYSize
	
	#Transform vertices
#	top_left = np.array([xMin, yMax])
#	scale = np.array([xMax - xMin, yMin - yMax])
	meters_per_native_pixel = (np.abs(geoTransform[1]) + np.abs(geoTransform[5])) / 2
	resolution_1024 = (img_3_uint8.shape[0] + img_3_uint8.shape[1])/2
	resolution_native = (geotiff.RasterXSize + geotiff.RasterYSize) / 2
	meters_per_1024_pixel = resolution_native / resolution_1024 * meters_per_native_pixel
	
	#Predict front using compiled model and retrieve results
	results = predict(model, img_3_uint8, mask_uint8, fjord_boundary, pred_norm_image, full_size, img_size, stride)
	image_settings['results'] = results
	image_settings['raw_image'] = results[0]
	image_settings['pred_image'] = results[1]
	image_settings['mask_image'] = results[2]
	image_settings['fjord_boundary_final_f32'] = results[3]
						
	image_settings['resolution_1024'] = resolution_1024
	image_settings['meters_per_1024_pixel'] = meters_per_1024_pixel
	image_settings['domain'] = domain
	image_settings['image_name_base'] = image_name_base

	image_settings['img_3_uint8'] = img_3_uint8
	image_settings['mask_uint8'] = mask_uint8 
	image_settings['fjord_boundary'] = fjord_boundary
	image_settings['meters_per_1024_pixel'] = meters_per_1024_pixel


if __name__ == '__main__':
	pass