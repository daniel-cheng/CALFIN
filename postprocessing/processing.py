# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 18:06:26 2019

@author: Daniel
"""

#from skimage.io import imsave
import numpy as np
from numpy import genfromtxt
from keras import backend as K
from pyproj import Proj, transform
from scipy.spatial import distance

import os, cv2, glob, gdal, sys
sys.path.insert(1, '../training/keras-deeplab-v3-plus')
sys.path.insert(2, '../training')
sys.path.insert(3, '../training/FrontLearning')
import unet_model
from model_cfm_dual_wide_x65 import Deeplabv3
from AdamAccumulate import AdamAccumulate
from segmentation_models.losses import bce_jaccard_loss, jaccard_loss, binary_crossentropy
from segmentation_models.metrics import iou_score, jaccard_score

os.environ["CUDA_VISIBLE_DEVICES"]="0" #Only make first GPU visible on multi-gpu setups
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

from aug_generators_dual import create_unaugmented_data_patches_from_rgb_image


def process(settings, metrics):	
	image_settings = settings['image_settings']
	image_settings['original_raw'] = image_settings['raw_image']
	image_settings['original_fjord_boundary'] = image_settings['fjord_boundary_final_f32']
	image_settings['unprocessed_original_raw'] = image_settings['unprocessed_raw_image']
	image_settings['unprocessed_original_fjord_boundary'] = image_settings['unprocessed_fjord_boundary']
	if settings['driver'] != 'production':
		image_settings['original_mask'] = image_settings['mask_image']
		image_settings['unprocessed_original_mask'] = image_settings['unprocessed_mask_image']
	predict(settings, metrics)
	

def predict(settings, metrics):	
	if settings['driver'] == 'calfin':	
		predict_calfin(settings, metrics)
	elif settings['driver'] == 'calfin_on_zhang':	
		predict_calfin(settings, metrics)
	elif settings['driver'] == 'calfin_on_baumhoer':	
		predict_calfin(settings, metrics)
	elif settings['driver'] == 'calfin_on_mohajerani':	
		predict_calfin(settings, metrics)
	elif settings['driver'] == 'mohajerani_on_calfin':
		predict_calfin(settings, metrics)
	elif settings['driver'] == 'mask_extractor':	
		process_mask(settings, metrics)
	elif settings['driver'] == 'production':	
		predict_calfin(settings, metrics)
	else:
		raise Exception('Input driver must be "calfin" or "mohajerani"')


def process_mask(settings, metrics):
	image_settings = settings['image_settings']
	img_final_f32 = image_settings['raw_image']
	mask_image = image_settings['mask_image'] 
	empty_image = settings['empty_image']
	
	#Initilize outputs
	raw_image_final_f32 = img_final_f32 / 255.0
	mask_final_f32 = np.stack((mask_image[:,:,0], mask_image[:,:,1], empty_image), axis = -1)
	
	#Save results
	image_settings['raw_image'] = raw_image_final_f32
	image_settings['pred_image'] = mask_final_f32


def predict_calfin(settings, metrics):	
	"""Takes in a neural network model, input image, mask, fjord boundary mask, and windowded normalization image to create prediction output.
	Uses the full original size, image patch size, and stride legnth as additional variables."""
	image_settings = settings['image_settings']
	img_final_f32 =	image_settings['raw_image']
	full_size = settings['full_size']
	img_size = settings['img_size']
	stride = settings['stride']
	model = settings['model']
	pred_norm_image = settings['pred_norm_image']
	
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
	
	#Save results
	image_settings['raw_image'] = raw_image_final_f32
	image_settings['pred_image'] = pred_image_final_f32


def process_mohajerani_on_calfin(settings, metrics):
	image_settings = settings['image_settings']
	full_size = settings['full_size']
	
	image_name_base = image_settings['image_name_base']
	bounding_box = image_settings['actual_bounding_box']
	unprocessed_original_raw = image_settings['unprocessed_original_raw']
	
	coordinates_path = r'D:\Daniel\Documents\Github\CALFIN Repo Intercomp\postprocessing\output_helheim_calfin'
	image_name_base_parts = image_name_base.split('_')
	domain = image_settings['domain']
	satellite = image_name_base_parts[1]
	level = image_name_base_parts[2]
	date = image_name_base_parts[3].replace('-', '')
	path_row = image_name_base_parts[4].replace('-', '')
	
	
	csv_path = glob.glob(os.path.join(coordinates_path, domain + ' ' + "_".join([satellite, level, path_row, date]) + '*'))[0]
	
	coordinates = genfromtxt(csv_path, delimiter=',')
	csv_name = os.path.basename(csv_path)
	csv_name_parts = csv_name.split()
	domain = csv_name_parts[0]
	landsat_name = csv_name_parts[1]
	landsat_name_parts = landsat_name.split('_')
	satellite = landsat_name_parts[0]
	level = landsat_name_parts[1]
	path_row = landsat_name_parts[2]
	date = landsat_name_parts[3]
	path = path_row[0:3]
	row = path_row[3:6]
	year = date[0:4]
	month = date[4:6]
	day = date[6:]
	tier = landsat_name_parts[6]
	band = landsat_name_parts[7]
	date_string = '-'.join([year, month, day])
	path_row_string = '-'.join([path, row])
	#Akullikassaap_LE07_L1TP_2000-03-17_014-009_T1_B4.png
	
#	calfin_path = r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\validation"
	tif_source_path = r"D:\Daniel\Documents\Github\CALFIN Repo\preprocessing\CalvingFronts\tif"
	
	calfin_name = '_'.join([domain, satellite, level, date_string, path_row_string, tier, band])
#	calfin_raw_path = os.path.join(calfin_path, calfin_name + '.png')
#	calfin_mask_path = os.path.join(calfin_path, calfin_name + '_mask.png')
	calfin_tif_path = os.path.join(tif_source_path, domain, year, calfin_name + '.tif')
		
	
	# Load geotiff and get domain layer/bounding box of area to mask
	geotiff = gdal.Open(calfin_tif_path)
#	print(calfin_tif_path)
	
	#Get bounds
	geoTransform = geotiff.GetGeoTransform()
	xMin = geoTransform[0]
	yMax = geoTransform[3]
#	xMax = xMin + geoTransform[1] * geotiff.RasterXSize
#	yMin = yMax + geoTransform[5] * geotiff.RasterYSize
	
#	print(calfin_raw_path, calfin_mask_path, calfin_tif_path)
	#install pyproj from pip instead of conda on windows to avoid undefined epsg
	inProj = Proj('epsg:3413')
	outProj = Proj('epsg:32624')
	reprojected_coordinates = transform(inProj,outProj,coordinates[:,0], coordinates[:,1])
	reprojected_coordinates = np.array(reprojected_coordinates).T 
	
	top_left = np.array([xMin, yMax])
	scale = np.array([1 / geoTransform[1], 1 / geoTransform[5]])
	transformed_coordinates = (reprojected_coordinates - top_left) * scale #pixel, 0, 0 is top left, 1024, 1004 is bottom right
	
	#256 by 256
	scale = np.array([full_size / unprocessed_original_raw.shape[0], full_size / unprocessed_original_raw.shape[0]]) 
	scaled_coordinates = transformed_coordinates * scale
	
	top_left = np.array([bounding_box[1], bounding_box[0]])
	scale = np.array([full_size / bounding_box[3], full_size / bounding_box[2]])
	subset_transformed_coordinates = (scaled_coordinates - top_left) * scale
#	transformed_coordinates[:,1] = unprocessed_original_raw.shape[1] - transformed_coordinates[:,1]
	
	pts = np.vstack((subset_transformed_coordinates[:,0],subset_transformed_coordinates[:,1])).astype(np.int32).T
	
	# Draw the lane onto the warped blank image
	#plt.plot(left_fitx, ploty, color='yellow')
	image = np.zeros((full_size, full_size, 3))
	cv2.polylines(image,  [pts],  False,  (1, 0, 0),  1)
	
	image_settings['polyline_image'] = image


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


def calculate_iou(gt, pr, smooth=1e-6, per_image=True):
	intersection = np.sum(gt[:,:,:] * pr[:,:,:], axis=(1, 2)) #(B)
	union = np.sum(gt[:,:,:] + pr[:,:,:] >= 1.0, axis=(1, 2)) #(B)
	iou_score = intersection / (union + smooth) #(B)
	mean_iou_score = np.mean(iou_score) #- (account for line thickness of 3 at 224)
	return mean_iou_score


SMOOTH = 1e-12
SMOOTH2 = 1e-1
def bce_ln_jaccard_loss(gt, pr, bce_weight=1.0, smooth=SMOOTH, per_image=True):
	bce = K.mean(binary_crossentropy(gt[:,:,:,0], pr[:,:,:,0]))*25/26 + K.mean(binary_crossentropy(gt[:,:,:,1], pr[:,:,:,1]))/26
	loss = bce_weight * bce - K.log(jaccard_score(gt[:,:,:,0], pr[:,:,:,0], smooth=smooth, per_image=per_image))*25/26 - K.log(jaccard_score(gt[:,:,:,1], pr[:,:,:,1], smooth=smooth, per_image=per_image))/26
	return loss


def iou_score(gt, pr, smooth=SMOOTH, per_image=True):
	edge_iou_score = jaccard_score(gt[:,:,:,0], pr[:,:,:,0], smooth=smooth, per_image=per_image)
	mask_iou_score = jaccard_score(gt[:,:,:,1], pr[:,:,:,1], smooth=smooth, per_image=per_image)
	return (edge_iou_score * 25 + mask_iou_score)/26

def edge_iou_score(gt, pr, smooth=SMOOTH, per_image=True):
	edge_iou_score = jaccard_score(gt[:,:,:,0], pr[:,:,:,0], smooth=smooth, per_image=per_image)
	return edge_iou_score


def mask_iou_score(gt, pr, smooth=SMOOTH, per_image=True):
	mask_iou_score = jaccard_score(gt[:,:,:,1], pr[:,:,:,1], smooth=smooth, per_image=per_image)
	return mask_iou_score


def deviation(gt, pr, smooth=SMOOTH2, per_image=True):
	mismatch = K.sum(K.abs(gt[:,:,:,1] - pr[:,:,:,1]), axis=[1, 2]) #(B)
	length = K.sum(gt[:,:,:,0], axis=[1, 2]) #(B)
	deviation = mismatch / (length + smooth) #(B)

	mean_deviation = K.mean(deviation) / 3.0 #- (account for line thickness of 3 at 224)
	return mean_deviation


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


def compile_unet_model(img_size):
	"""Compile the CALFIN Neural Network model and loads pretrained weights."""
	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)
	height = img_size
	width = img_size
	channels = 3
	n_init = 32
	n_layers = 4
	drop = 0.2
	model = unet_model.unet_model_double_dropout(height=height, width=width, channels=channels, n_init=n_init, n_layers=n_layers, drop=drop)
	print('Importing unet_model_double_dropout...')
	

	model.compile(optimizer=AdamAccumulate(lr=1e-4, accum_iters=2), loss=bce_ln_jaccard_loss, metrics=['binary_crossentropy', iou_score, edge_iou_score, mask_iou_score, deviation])
	model.summary()
	model.load_weights('../training/mohajerani_224_3_32_4_e59_iou0.4981.h5')
	
	return model
