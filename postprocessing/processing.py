# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 18:06:26 2019

@author: Daniel
"""

#from skimage.io import imsave
import numpy as np
from keras import backend as K

import sys, os
sys.path.insert(1, '../training/keras-deeplab-v3-plus')
sys.path.insert(2, '../training')
from model_cfm_dual_wide_x65 import Deeplabv3
from AdamAccumulate import AdamAccumulate

os.environ["CUDA_VISIBLE_DEVICES"]="0" #Only make first GPU visible on multi-gpu setups
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

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


def predict(settings, metrics):	
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


def process(settings, metrics):	
	image_settings = settings['image_settings']
	image_settings['original_raw'] = image_settings['raw_image']
	image_settings['original_mask'] = image_settings['mask_image']
	image_settings['original_fjord_boundary'] = image_settings['fjord_boundary_final_f32']
	image_settings['unprocessed_original_raw'] = image_settings['unprocessed_raw_image']
	image_settings['unprocessed_original_mask'] = image_settings['unprocessed_mask_image']
	image_settings['unprocessed_original_fjord_boundary'] = image_settings['unprocessed_fjord_boundary']
	predict(settings, metrics)



