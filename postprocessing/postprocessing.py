# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 18:06:26 2019

@author: Daniel
"""

import matplotlib.pyplot as plt
import numpy as np
import sys, cv2
from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import skeletonize
sys.path.insert(1, '../training/keras-deeplab-v3-plus')
sys.path.insert(2, '../training')

from preprocessing import preprocess_image
from processing import predict, calculate_mean_deviation, calculate_iou, process_mohajerani_on_calfin
from plotting import plot_validation_results, plot_production_results, plot_troubled_ones
from mask_to_shp import mask_to_shp
from error_analysis import extract_front_indicators
from ordered_line_from_unordered_points import is_outlier


def postprocess(settings, metrics):
	if settings['driver'] == 'production':
		postprocess_production(settings, metrics)
	else:
		postprocess_validated(settings, metrics)


def postprocess_validated(settings, metrics):	
	kernel = settings['kernel']
	image_settings = settings['image_settings']
	resolution_1024 = image_settings['resolution_1024']
	meters_per_1024_pixel = image_settings['meters_per_1024_pixel']
	image_name_base = image_settings['image_name_base']
	empty_image = settings['empty_image']
	
	raw_image = image_settings['raw_image']
	pred_image = image_settings['pred_image']
	mask_image = image_settings['mask_image']
	fjord_boundary_final_f32 = image_settings['fjord_boundary_final_f32']
	i = image_settings['i']
	
	#recalculate scaling
	resolution_256 = (raw_image.shape[0] + raw_image.shape[1])/2
	meters_per_256_pixel = resolution_1024 / resolution_256  * meters_per_1024_pixel
	
	#Perform optimiation on fjord boundaries.
	#Continuously erode fjord boundary mask until fjord edges are masked.
	#This is detected by looking for large increases in pixels being masked,
	#followed by few pixels being masked.
	results_pred = mask_polyline(pred_image[:,:,0], fjord_boundary_final_f32, settings)
	results_mask = mask_polyline(mask_image[:,:,0], fjord_boundary_final_f32, settings)
	polyline_image = np.stack((results_pred[0], empty_image, empty_image), axis=-1)
	bounding_boxes = results_pred[1]
	mask_edge_f32 = results_mask[0]
	
	#Calculate and save mean deviation, distances, and IoU metrics based on new masked polyline
	mean_deviation, distances = calculate_mean_deviation(polyline_image[:,:,0], mask_edge_f32)
	image_settings['distances'] = distances
	print("mean_deviation: {:.2f} pixels, {:.2f} meters".format(mean_deviation, mean_deviation * meters_per_256_pixel))
	
	#Dilate masks for easier visualization and IoU metric calculation.
	polyline_image_dilated = cv2.dilate(polyline_image.astype('float64'), kernel, iterations = 1).astype(np.float32) #np.float32 [0.0, 255.0]
	mask_edge_dilated_f32 = cv2.dilate(mask_edge_f32.astype('float64'), kernel, iterations = 1).astype(np.float32) #np.float32 [0.0, 255.0]
	mask_final_dilated_f32 = np.stack((mask_edge_dilated_f32, mask_image[:,:,1]), axis = -1)
	image_settings['polyline_image'] = polyline_image_dilated
	image_settings['mask_image'] = mask_final_dilated_f32
	
	#Calculate and save IoU metric
	pred_patch_4d = np.expand_dims(polyline_image_dilated[:,:,0], axis=0)
	mask_patch_4d = np.expand_dims(mask_final_dilated_f32[:,:,0], axis=0)
	edge_iou_score = calculate_iou(mask_patch_4d, pred_patch_4d)
	print("edge_iou_score:  {:.2f}".format(edge_iou_score))
	
	#For each calving front, subset the image AGAIN and predict. This helps accuracy for inputs with large scaling/downsampling ratios
	box_counter = 0	
	found_front = False
	image_settings['box_counter'] = box_counter
	image_settings['meters_per_256_pixel'] = meters_per_256_pixel
	image_settings['resolution_256'] = resolution_256
	image_settings['mean_deviation'] = mean_deviation
	image_settings['edge_iou_score'] = edge_iou_score
	image_settings['used_bounding_boxes'] = []
	
	#Try to find a front in each bounding box, if any
	if len(bounding_boxes) > 1:
		print('bounding_boxes', bounding_boxes)
		for bounding_box in bounding_boxes[1:]:
			image_settings['bounding_box'] = bounding_box
			found_front_in_box, metrics = reprocess_validated(settings, metrics)
			found_front = found_front or found_front_in_box
	
	#if no fronts are found yet, fallback to box that encloses entire image
	if not found_front:
		box_counter = 0
		image_settings['box_counter'] = box_counter
		image_settings['bounding_box'] = bounding_boxes[0]
		found_front, metrics = reprocess_validated(settings, metrics)
		#If we still haven't found a front in the default, we "skip" the image and make a note of it.
		if not found_front:
			metrics['image_skip_count'] += 1
		
	#Calculate confusion matrix (TP/TN/FP/FN)
	if not found_front: #If front is not found,
		if image_name_base in settings['negative_image_names']: #and there is no front, it's a true negative
			metrics['true_negatives'] += 1
		else:  #and there is a front, it's a false negative
			metrics['false_negatives'] += 1
	else: #If front is found,
		if image_name_base in settings['negative_image_names']: #and there is no front, it's a false positive
			metrics['false_positive'] += 1 
		else:  #and there is one, it's a true positive
			metrics['true_positives'] += 1
	
	print('Done {0}: {1}/{2} images'.format(image_name_base, i + 1, settings['total']))
	return metrics


def reprocess_validated(settings, metrics): 
	full_size = settings['full_size']
	mask_confidence_strength_threshold = settings['mask_confidence_strength_threshold']
	edge_confidence_strength_threshold = settings['edge_confidence_strength_threshold']
	domain_scalings = settings['domain_scalings']
	plotting = settings['plotting']
	empty_image = settings['empty_image']
	edge_detection_threshold = settings['edge_detection_threshold']
	edge_detection_size_threshold = settings['edge_detection_size_threshold']
	mask_detection_threshold = settings['mask_detection_threshold']
	mask_detection_ratio_threshold = settings['mask_detection_ratio_threshold']
	
	#image_settings are assigned per front in the image
	image_settings = settings['image_settings']
	domain = image_settings['domain']
	box_counter = image_settings['box_counter']
	bounding_box = image_settings['bounding_box']
	img_3_uint8 = image_settings['unprocessed_original_raw']
	mask_uint8 = image_settings['unprocessed_original_mask']
	fjord_boundary = image_settings['unprocessed_original_fjord_boundary']
	meters_per_1024_pixel = image_settings['meters_per_1024_pixel']
	resolution_256 = image_settings['resolution_256']
			
	image_settings['box_counter'] += 1
	box_counter += 1
	found_front = False
	
	#Try to get nearest square subset with padding equal to size of initial front
	fractional_bounding_box = np.array(bounding_box) / full_size
	sub_width = fractional_bounding_box[2] * img_3_uint8.shape[0]
	sub_height = fractional_bounding_box[3] * img_3_uint8.shape[1]
	sub_length = max(sub_width, sub_height)
	sub_x1 = fractional_bounding_box[0] * img_3_uint8.shape[0]
	sub_x2 = sub_x1 + sub_width
	sub_y1 = fractional_bounding_box[1] * img_3_uint8.shape[1]
	sub_y2 = sub_y1 + sub_height
	
	sub_center_x = (sub_x1 + sub_x2) / 2
	sub_center_y = (sub_y1 + sub_y2) / 2
	
	#Ensure subset will be within image bounds
	sub_padding_ratio = settings['sub_padding_ratio']
	half_sub_padding_ratio = sub_padding_ratio / 2
	sub_padding = sub_length * half_sub_padding_ratio
	
	#Ensure subset is at least of dimensions (full_size, full_size) and at most the size of the raw image
	if sub_padding < full_size / 2:
		sub_padding = full_size / 2
	elif sub_padding * 2 > img_3_uint8.shape[0] or sub_padding * 2 > img_3_uint8.shape[1]:
		sub_padding = np.floor(min(img_3_uint8.shape[0], img_3_uint8.shape[1]) / 2)
		
	if sub_center_x + sub_padding > img_3_uint8.shape[0]:
		sub_center_x = img_3_uint8.shape[0] - sub_padding
	elif sub_center_x - sub_padding < 0:
		sub_center_x = 0 + sub_padding
	if sub_center_y + sub_padding > img_3_uint8.shape[1]:
		sub_center_y = img_3_uint8.shape[1] - sub_padding
	elif sub_center_y - sub_padding < 0:
		sub_center_y = 0 + sub_padding
		
	#calculate new subset x/y bounds, which are guarenteed to be of sufficient size and to be within the bounds of the image
	sub_x1 = int(sub_center_x - sub_padding)
	sub_x2 = int(sub_center_x + sub_padding)
	sub_y1 = int(sub_center_y - sub_padding)
	sub_y2 = int(sub_center_y + sub_padding)
	
#	scaling = resolution_256 / resolution_1024
	actual_bounding_box = [sub_x1 / img_3_uint8.shape[0] * full_size, sub_y1 / img_3_uint8.shape[1] * full_size, (sub_x2 - sub_x1) / img_3_uint8.shape[0] * full_size, (sub_y2 - sub_y1) / img_3_uint8.shape[1] * full_size]
	print("bounding_box:", bounding_box, "fractional_bounding_box:", fractional_bounding_box, 'actual_bounding_box', actual_bounding_box)
	image_settings['actual_bounding_box'] = actual_bounding_box
#	print(sub_x1, sub_y1, sub_x2, sub_y2)
	
	#Perform subsetting
	resolution_subset = sub_padding * 2 #not simplified for clarity - just find average dimensions	
	meters_per_subset_pixel = meters_per_1024_pixel
	img_3_subset_uint8 = img_3_uint8[sub_x1:sub_x2, sub_y1:sub_y2, :]
	mask_subset_uint8 = mask_uint8[sub_x1:sub_x2, sub_y1:sub_y2]
	fjord_subset_boundary = fjord_boundary[sub_x1:sub_x2, sub_y1:sub_y2]
	
	#Repredict
	image_settings['unprocessed_raw_image'] = img_3_subset_uint8
	image_settings['unprocessed_mask_image'] = mask_subset_uint8
	image_settings['unprocessed_fjord_boundary'] = fjord_subset_boundary
	preprocess_image(settings, metrics)
	predict(settings, metrics)
	mask_image = image_settings['mask_image']
	fjord_boundary = image_settings['fjord_boundary_final_f32']
	pred_image = image_settings['pred_image']
	
	edge_detection_size_indices = np.nan_to_num(pred_image[:,:,0]) > edge_detection_threshold
	edge_detection_size = np.sum(edge_detection_size_indices, axis=(0, 1)) / 3 #Divide by 3 to account for 3-pixel wide edge
	mask_detection_ratio_indices = np.nan_to_num(pred_image[:,:,1]) > mask_detection_threshold
	mask_detection_size = np.sum(mask_detection_ratio_indices, axis=(0, 1))
	mask_nondetection_size = np.size(pred_image[:,:,1]) - mask_detection_size
	mask_detection_ratio = mask_detection_size / (mask_nondetection_size + 1)
	edge_confidence_strength_indices = np.nan_to_num(pred_image[:,:,0]) > 0.05
	edge_confidence_strength = np.mean(np.abs(0.5 - np.nan_to_num(pred_image[:,:,0][edge_confidence_strength_indices])))
	mask_confidence_strength_indices = np.nan_to_num(pred_image[:,:,1]) > 0.05
	mask_confidence_strength = np.mean(np.abs(0.5 - np.nan_to_num(pred_image[:,:,1][mask_confidence_strength_indices])))
	
	print("edge_detection_size: {:.3f} pixels".format(edge_detection_size))
	print("mask_detection_ratio: {:.3f}".format(mask_detection_ratio))
	print("edge_confidence_strength: {:.3f}".format(edge_confidence_strength * 2))
	print("mask_confidence_strength: {:.3f}".format(mask_confidence_strength * 2))
	edge_unconfident = edge_confidence_strength * 2 < edge_confidence_strength_threshold
	mask_unconfident = mask_confidence_strength * 2 < mask_confidence_strength_threshold
	edge_size_unconfident = edge_detection_size < edge_detection_size_threshold
	mask_ratio_unconfident = mask_detection_ratio > mask_detection_ratio_threshold
	
	if edge_unconfident or mask_unconfident or edge_size_unconfident or mask_ratio_unconfident:
		print("not confident, skipping")
		metrics['confidence_skip_count'] += 1
		return found_front, metrics
	
	#recalculate scaling
	meters_per_subset_pixel = resolution_subset / resolution_256  * meters_per_1024_pixel
	image_settings['meters_per_subset_pixel'] = meters_per_subset_pixel
	if domain not in domain_scalings:
		domain_scalings[domain] = meters_per_subset_pixel
	
	#Redo masking
	results_pred = mask_polyline(pred_image[:,:,0], fjord_boundary, settings, min_size_percentage=0.0005, use_extracted_front=False)
	results_mask = mask_polyline(mask_image[:,:,0], fjord_boundary, settings, min_size_percentage=0.0005, use_extracted_front=False)
	polyline_image = np.stack((results_pred[0], empty_image, empty_image), axis=-1)
	
	bounding_boxes_pred = results_pred[1]
	mask_edge_f32 = results_mask[0]
	bounding_boxes_mask = results_mask[1]
	
	#mask out largest front
	if len(bounding_boxes_pred) < 2 or len(bounding_boxes_mask) < 2:
		print("no front detected, skipping")
		metrics['no_detection_skip_count'] += 1
		return found_front, metrics
	polyline_image, bounding_box_pred = mask_bounding_box(bounding_boxes_pred, polyline_image, settings)
	mask_edge_f32, bounding_box_mask = mask_bounding_box(bounding_boxes_mask, mask_edge_f32, settings, bounding_box_pred) #If need stricter constraints, uncomment these lines
	mask_final_f32 = np.stack((mask_edge_f32, mask_image[:,:,1]), axis = -1)
	results_polyline = extract_front_indicators(polyline_image[:,:,0], z_score_cutoff=25.0)
	polyline_image = np.stack((results_polyline[0][:,:,0] / 255.0, empty_image, empty_image), axis=-1)
	
	image_settings['polyline_coords'] = results_polyline[1]
	image_settings['bounding_box_pred'] = bounding_box_pred
	image_settings['polyline_image'] = polyline_image
	image_settings['mask_image'] = mask_final_f32
	
	if settings['driver'] == 'mohajerani_on_calfin':
		process_mohajerani_on_calfin(settings, metrics)
	
	#Calculate validation metrics
	calculate_metrics_calfin(settings, metrics)
	
	#Plot and save results
	plot_validation_results(settings, metrics)
		
	#Generate shape file
	mask_to_shp(settings, metrics)
	
	#Notify rest of the code that a front has been found, so that we don't use fallback front
	metrics['front_count'] += 1
	found_front = True
	
	return found_front, metrics


def postprocess_production(settings, metrics):
	kernel = settings['kernel']
	image_settings = settings['image_settings']
	resolution_1024 = image_settings['resolution_1024']
	meters_per_1024_pixel = image_settings['meters_per_1024_pixel']
	image_name_base = image_settings['image_name_base']
	empty_image = settings['empty_image']
	
	raw_image = image_settings['raw_image']
	pred_image = image_settings['pred_image']
	fjord_boundary_final_f32 = image_settings['fjord_boundary_final_f32']
	i = image_settings['i']
	
	#recalculate scaling
	resolution_256 = (raw_image.shape[0] + raw_image.shape[1])/2
	meters_per_256_pixel = resolution_1024 / resolution_256  * meters_per_1024_pixel
	
	#Perform optimiation on fjord boundaries.
	#Continuously erode fjord boundary mask until fjord edges are masked.
	#This is detected by looking for large increases in pixels being masked,
	#followed by few pixels being masked.
	results_pred = mask_polyline(pred_image[:,:,0], fjord_boundary_final_f32, settings)
	polyline_image = np.stack((results_pred[0], empty_image, empty_image), axis=-1)
	bounding_boxes = results_pred[1]
	
	#Calculate and save mean deviation, distances, and IoU metrics based on new masked polyline
#	mean_deviation, distances = calculate_mean_deviation(polyline_image[:,:,0], mask_edge_f32)
#	image_settings['distances'] = distances
#	print("mean_deviation: {:.2f} pixels, {:.2f} meters".format(mean_deviation, mean_deviation * meters_per_256_pixel))
	
	#Dilate masks for easier visualization and IoU metric calculation.
	polyline_image_dilated = cv2.dilate(polyline_image.astype('float64'), kernel, iterations = 1).astype(np.float32) #np.float32 [0.0, 255.0]
#	mask_edge_dilated_f32 = cv2.dilate(mask_edge_f32.astype('float64'), kernel, iterations = 1).astype(np.float32) #np.float32 [0.0, 255.0]
#	mask_final_dilated_f32 = np.stack((mask_edge_dilated_f32, mask_image[:,:,1]), axis = -1)
	image_settings['polyline_image'] = polyline_image_dilated
#	image_settings['mask_image'] = mask_final_dilated_f32
	
	#Calculate and save IoU metric
#	pred_patch_4d = np.expand_dims(polyline_image_dilated[:,:,0], axis=0)
#	mask_patch_4d = np.expand_dims(mask_final_dilated_f32[:,:,0], axis=0)
#	edge_iou_score = calculate_iou(mask_patch_4d, pred_patch_4d)
#	print("edge_iou_score:  {:.2f}".format(edge_iou_score))
	
	#For each calving front, subset the image AGAIN and predict. This helps accuracy for inputs with large scaling/downsampling ratios
	box_counter = 0	
	found_front = False
	image_settings['box_counter'] = box_counter
	image_settings['meters_per_256_pixel'] = meters_per_256_pixel
	image_settings['resolution_256'] = resolution_256
#	image_settings['mean_deviation'] = mean_deviation
#	image_settings['edge_iou_score'] = edge_iou_score
	image_settings['used_bounding_boxes'] = []
	
	#Try to find a front in each bounding box, if any
	if len(bounding_boxes) > 1:
		print('bounding_boxes', bounding_boxes)
		for bounding_box in bounding_boxes[1:]:
			image_settings['bounding_box'] = bounding_box
			found_front_in_box, metrics = reprocess_production(settings, metrics)
			found_front = found_front or found_front_in_box
	
	#if no fronts are found yet, fallback to box that encloses entire image
	if not found_front:
		box_counter = 0
		image_settings['box_counter'] = box_counter
		image_settings['bounding_box'] = bounding_boxes[0]
		found_front, metrics = reprocess_production(settings, metrics)
		#If we still haven't found a front in the default, we "skip" the image and make a note of it.
		if not found_front:
			plot_troubled_ones(settings, metrics)
			metrics['image_skip_count'] += 1
		
	#Calculate confusion matrix (TP/TN/FP/FN)
	if not found_front: #If front is not found,
		if image_name_base in settings['negative_image_names']: #and there is no front, it's a true negative
			metrics['true_negatives'] += 1
		else:  #and there is a front, it's a false negative
			metrics['false_negatives'] += 1
	else: #If front is found,
		if image_name_base in settings['negative_image_names']: #and there is no front, it's a false positive
			metrics['false_positive'] += 1 
		else:  #and there is one, it's a true positive
			metrics['true_positives'] += 1
	
	print('Done {0}: {1}/{2} images'.format(image_name_base, i + 1, settings['total']))
	return metrics


def reprocess_production(settings, metrics): 
	full_size = settings['full_size']
	mask_confidence_strength_threshold = settings['mask_confidence_strength_threshold']
	edge_confidence_strength_threshold = settings['edge_confidence_strength_threshold']
	domain_scalings = settings['domain_scalings']
	empty_image = settings['empty_image']
	edge_detection_threshold = settings['edge_detection_threshold']
	edge_detection_size_threshold = settings['edge_detection_size_threshold']
	mask_detection_threshold = settings['mask_detection_threshold']
	mask_detection_ratio_threshold = settings['mask_detection_ratio_threshold']
	
	#image_settings are assigned per front in the image
	image_settings = settings['image_settings']
	domain = image_settings['domain']
	box_counter = image_settings['box_counter']
	bounding_box = image_settings['bounding_box']
	img_3_uint8 = image_settings['unprocessed_original_raw']
	fjord_boundary = image_settings['unprocessed_original_fjord_boundary']
	meters_per_1024_pixel = image_settings['meters_per_1024_pixel']
	resolution_256 = image_settings['resolution_256']
			
	image_settings['box_counter'] += 1
	box_counter += 1
	found_front = False
	
	#Try to get nearest square subset with padding equal to size of initial front
	fractional_bounding_box = np.array(bounding_box) / full_size
	sub_width = fractional_bounding_box[2] * img_3_uint8.shape[0]
	sub_height = fractional_bounding_box[3] * img_3_uint8.shape[1]
	sub_length = max(sub_width, sub_height)
	sub_x1 = fractional_bounding_box[0] * img_3_uint8.shape[0]
	sub_x2 = sub_x1 + sub_width
	sub_y1 = fractional_bounding_box[1] * img_3_uint8.shape[1]
	sub_y2 = sub_y1 + sub_height
	
	sub_center_x = (sub_x1 + sub_x2) / 2
	sub_center_y = (sub_y1 + sub_y2) / 2
	
	#Ensure subset will be within image bounds
	sub_padding_ratio = settings['sub_padding_ratio']
	half_sub_padding_ratio = sub_padding_ratio / 2
	sub_padding = sub_length * half_sub_padding_ratio
	
	#Ensure subset is at least of dimensions (full_size, full_size) and at most the size of the raw image
	if sub_padding < full_size / 2:
		sub_padding = full_size / 2
	elif sub_padding * 2 > img_3_uint8.shape[0] or sub_padding * 2 > img_3_uint8.shape[1]:
		sub_padding = np.floor(min(img_3_uint8.shape[0], img_3_uint8.shape[1]) / 2)
		
	if sub_center_x + sub_padding > img_3_uint8.shape[0]:
		sub_center_x = img_3_uint8.shape[0] - sub_padding
	elif sub_center_x - sub_padding < 0:
		sub_center_x = 0 + sub_padding
	if sub_center_y + sub_padding > img_3_uint8.shape[1]:
		sub_center_y = img_3_uint8.shape[1] - sub_padding
	elif sub_center_y - sub_padding < 0:
		sub_center_y = 0 + sub_padding
		
	#calculate new subset x/y bounds, which are guarenteed to be of sufficient size and to be within the bounds of the image
	sub_x1 = int(sub_center_x - sub_padding)
	sub_x2 = int(sub_center_x + sub_padding)
	sub_y1 = int(sub_center_y - sub_padding)
	sub_y2 = int(sub_center_y + sub_padding)
	
#	scaling = resolution_256 / resolution_1024
	actual_bounding_box = [sub_x1 / img_3_uint8.shape[0] * full_size, sub_y1 / img_3_uint8.shape[1] * full_size, (sub_x2 - sub_x1) / img_3_uint8.shape[0] * full_size, (sub_y2 - sub_y1) / img_3_uint8.shape[1] * full_size]
	print("bounding_box:", bounding_box, "fractional_bounding_box:", fractional_bounding_box, 'actual_bounding_box', actual_bounding_box)
	image_settings['actual_bounding_box'] = actual_bounding_box
#	print(sub_x1, sub_y1, sub_x2, sub_y2)
	
	#Perform subsetting
	resolution_subset = sub_padding * 2 #not simplified for clarity - just find average dimensions	
	meters_per_subset_pixel = meters_per_1024_pixel
	img_3_subset_uint8 = img_3_uint8[sub_x1:sub_x2, sub_y1:sub_y2, :]
	fjord_subset_boundary = fjord_boundary[sub_x1:sub_x2, sub_y1:sub_y2]
	
	#Repredict
	image_settings['unprocessed_raw_image'] = img_3_subset_uint8
	image_settings['unprocessed_fjord_boundary'] = fjord_subset_boundary
	preprocess_image(settings, metrics)
	predict(settings, metrics)
	fjord_boundary = image_settings['fjord_boundary_final_f32']
	pred_image = image_settings['pred_image']
	
	edge_detection_size_indices = np.nan_to_num(pred_image[:,:,0]) > edge_detection_threshold
	edge_detection_size = np.sum(edge_detection_size_indices, axis=(0, 1)) / 3 #Divide by 3 to account for 3-pixel wide edge
	mask_detection_ratio_indices = np.nan_to_num(pred_image[:,:,1]) > mask_detection_threshold
	mask_detection_size = np.sum(mask_detection_ratio_indices, axis=(0, 1))
	mask_nondetection_size = np.size(pred_image[:,:,1]) - mask_detection_size
	mask_detection_ratio = mask_detection_size / (mask_nondetection_size + 1)
	edge_confidence_strength_indices = np.nan_to_num(pred_image[:,:,0]) > 0.05
	edge_confidence_strength = np.mean(np.abs(0.5 - np.nan_to_num(pred_image[:,:,0][edge_confidence_strength_indices])))
	mask_confidence_strength_indices = np.nan_to_num(pred_image[:,:,1]) > 0.05
	mask_confidence_strength = np.mean(np.abs(0.5 - np.nan_to_num(pred_image[:,:,1][mask_confidence_strength_indices])))
	
	print("edge_detection_size: {:.3f} pixels".format(edge_detection_size))
	print("mask_detection_ratio: {:.3f}".format(mask_detection_ratio))
	print("edge_confidence_strength: {:.3f}".format(edge_confidence_strength * 2))
	print("mask_confidence_strength: {:.3f}".format(mask_confidence_strength * 2))
	edge_unconfident = edge_confidence_strength * 2 < edge_confidence_strength_threshold
	mask_unconfident = mask_confidence_strength * 2 < mask_confidence_strength_threshold
	edge_size_unconfident = edge_detection_size < edge_detection_size_threshold
	mask_ratio_unconfident = mask_detection_ratio > mask_detection_ratio_threshold
	
	if edge_unconfident or mask_unconfident or edge_size_unconfident or mask_ratio_unconfident:
		print("not confident, skipping")
		metrics['confidence_skip_count'] += 1
		return found_front, metrics
	
	#recalculate scaling
	meters_per_subset_pixel = resolution_subset / resolution_256  * meters_per_1024_pixel
	image_settings['meters_per_subset_pixel'] = meters_per_subset_pixel
	if domain not in domain_scalings:
		domain_scalings[domain] = meters_per_subset_pixel
	
	#Redo masking
	results_pred = mask_polyline(pred_image[:,:,0], fjord_boundary, settings, min_size_percentage=0.0005, use_extracted_front=False)
	polyline_image = np.stack((results_pred[0], empty_image, empty_image), axis=-1)
	
	bounding_boxes_pred = results_pred[1]
	
	#mask out largest front
	if len(bounding_boxes_pred) < 2:
		print("no front detected, skipping")
		metrics['no_detection_skip_count'] += 1
		return found_front, metrics
	polyline_image, bounding_box_pred = mask_bounding_box(bounding_boxes_pred, polyline_image, settings)
	results_polyline = extract_front_indicators(polyline_image[:,:,0], z_score_cutoff=25.0)
	polyline_image = np.stack((results_polyline[0][:,:,0] / 255.0, empty_image, empty_image), axis=-1)
	
	image_settings['polyline_coords'] = results_polyline[1]
	image_settings['bounding_box_pred'] = bounding_box_pred
	image_settings['polyline_image'] = polyline_image
	
	#Plot and save results
	kernel = settings['kernel']
	polyline_image_dilated = cv2.dilate(polyline_image.astype('float64'), kernel, iterations = 1).astype(np.float32) #np.float32 [0.0, 255.0]
	image_settings['polyline_image_dilated'] = polyline_image_dilated
	plot_production_results(settings, metrics)
	
	#Generate shape file
	mask_to_shp(settings, metrics)
	
	#Notify rest of the code that a front has been found, so that we don't use fallback front
	metrics['front_count'] += 1
	found_front = True
	
	return found_front, metrics


def remove_small_components(image:np.ndarray, limit=np.inf, min_size_percentage=0.00025):
	image = image.astype('uint8')
	#find all connected components (white blobs in image)
	nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
	#connectedComponentswithStats yields every seperated component with information on each of them, such as size
	sizes = stats[:,cv2.CC_STAT_AREA]
	ordering = np.argsort(-sizes)
	
	#for every component in the image, keep it only if it's above min_size
	min_size_floor = output.size * min_size_percentage
	if len(ordering) > 1:
		min_size = sizes[ordering[1]] * 0.1
	
	#Isolate large components
	largeComponents = np.zeros((output.shape))
	
	#Store the bounding boxes of components, so they can be isolated and reprocessed further. Default box is entire image.
	bounding_boxes = [[0, 0, image.shape[0], image.shape[1]]]
	#Skip first component, since it's the background color in edge masks
	#Restrict number of components returned depending on limit
	number_returned = 0
	for i in range(1, len(sizes)):
		if sizes[ordering[i]] >= min_size_floor and sizes[ordering[i]] >= min_size:
			mask_indices = output == ordering[i]
			x, y = np.nonzero(mask_indices)
			min_x = min(x)
			delta_x = max(x) - min_x
			min_y = min(y)
			delta_y = max(y) - min_y
			bounding_boxes.append([min_x, min_y, delta_x, delta_y])
			largeComponents[mask_indices] = image[mask_indices]
			number_returned += 1
			if number_returned >= limit:
				break
	
	#return large component image and bounding boxes for each componnet
	return largeComponents.astype(np.float32), bounding_boxes


def mask_polyline(pred_image, fjord_boundary_final_f32, settings, min_size_percentage=0.00025, use_extracted_front=True):
	""" Perform optimiation on fjord boundaries.
		Continuously erode fjord boundary mask until fjord edges are masked.
		This is detected by looking for large increases in pixels being masked,
		followed by few pixels being masked."""
	
	#get distance of each point from fjord boundary black pixel
	fjord_distances = distance_transform_edt(fjord_boundary_final_f32)
	results_polyline = extract_front_indicators(pred_image)
	
	#If front is detected, proceed with extraction
	if not results_polyline is None:
		use_extracted_front = use_extracted_front or settings['always_use_extracted_front']
		if use_extracted_front:
			polyline_image = results_polyline[0][:,:,0] / 255.0
			polyline_coords = results_polyline[1]
		else:
			#To handle cases where zoomed on basin with discounted boundary/front walls like 
			#Cornell_LE07_L1TP_2000-04-03_021-007_T1_B4_92-1, only use initial results_polyline to
			#detect presence of front - don't use the output image, since the fjord boundaries
			#have not been masked yet! (otherwise, extract_front_indicators will try to connect the gap)
			edge_binary = np.where(pred_image > 0.25, 1, 0)
			skeleton = skeletonize(edge_binary)
			front_pixels = np.nonzero(skeleton)
			polyline_image = skeleton
			polyline_coords = np.array([front_pixels[0], front_pixels[1]])
				
		#Find all pixels where distance to nearest fjord boundary is an outlier
		polyline_distances_image = fjord_distances * polyline_image
		polyline_coords_distances = polyline_distances_image[tuple(polyline_coords)]
		data = polyline_coords_distances
		
		#stop gap FIX
		if np.floor(data.max() / 2).astype(int) == 0:
			polyline_image = settings['empty_image']
			bounding_boxes = [[0, 0, settings['full_size'], settings['full_size']]]
			return polyline_image, bounding_boxes
		counts, bin_edges = np.histogram(data, bins = np.floor(data.max() / 2).astype(int))
		inlier_mask = np.logical_not(is_outlier(counts, z_score_cutoff=2.0))
		counts_inlier_max = counts[inlier_mask].max()
		
		#Determine threshold to cutoff
		counts_median = np.median(counts)
		print('counts_median', counts_median, 'counts_inlier_max', counts_inlier_max)
		threshold = 3
		for i in range(len(counts)):
			if counts[i] < counts_inlier_max:
				threshold = bin_edges[i]
				break
		
		print('threshold', threshold)
		thresholded_distances = polyline_distances_image * (polyline_distances_image > threshold)
				
		#readd elements that are close to 4
		inverse_distances = 255 - thresholded_distances
		
		#set all < -3 =0
		inverse_distances[inverse_distances < (255 - threshold)] = 0
		
		#remove small components
		large_inverse_distances = 255-remove_small_components(255-inverse_distances)[0]
		
		#get distance transform of inverse
		distances_to_front = distance_transform_edt(large_inverse_distances)
		
		#add any elements < 3 back to mask WHY ISNT THIS WORKING QUESTION MARK?
		isolated_fronts = polyline_distances_image * (np.logical_and(distances_to_front < threshold * 1.0, polyline_image > 0))
		
		#Perform final dummy remove_small_components to retrieve bounding boxes
		polyline_image = np.where(isolated_fronts > 0, 1.0, 0.0)
		
		polyline_image, bounding_boxes = remove_small_components(polyline_image, min_size_percentage=min_size_percentage)
	else:
		#If no front is detected, return empty image.
		polyline_image = settings['empty_image']
		bounding_boxes = [[0, 0, settings['full_size'], settings['full_size']]]
		polyline_coords = None
	
	return polyline_image, bounding_boxes, polyline_coords


def mask_bounding_box(bounding_boxes, image, settings, target_box = None):
	bounding_box = bounding_boxes[1]
	#If more than 1 bounding box, find the one closest to the target, in case multiple fronts are close
	if len(bounding_boxes) > 2:
		if target_box is None: #Default to target box if mask, closest to center that hasn't been used already for pred
			target_center = np.array([image.shape[0] / 2, image.shape[1] / 2])
			distances = [np.inf]
			for i in range(1, len(bounding_boxes)):
				bounding_box = bounding_boxes[i]
				bounding_box_center_x = bounding_box[0] + bounding_box[2] / 2 
				bounding_box_center_y = bounding_box[1] + bounding_box[3] / 2 
				bounding_box_center = np.array([bounding_box_center_x, bounding_box_center_y])
				distance = np.linalg.norm(target_center - bounding_box_center)
				distances.append(distance)
			
			#Go through bounding boxes in order of distance to target, picking first one that hasn't already been used
			#Determine closeness to already used bounding boxes by ensuring inter box distance is below some threshold
			distances_ordering = np.argsort(distances)
			image_settings = settings['image_settings']
			inter_box_distance_threshold = settings['inter_box_distance_threshold']
			used_bounding_boxes = image_settings['used_bounding_boxes']
			for i in range(len(distances_ordering)):
				index = distances_ordering[i]			
				bounding_box = bounding_boxes[index]
				bounding_box_center_x = bounding_box[0] + bounding_box[2] / 2 
				bounding_box_center_y = bounding_box[1] + bounding_box[3] / 2 
				bounding_box_center = np.array([bounding_box_center_x, bounding_box_center_y])
				
				original_bounding_box = image_settings['actual_bounding_box']
				top_left = np.array([original_bounding_box[0], original_bounding_box[1]])
				scale = np.array([original_bounding_box[2] / settings['full_size'], original_bounding_box[3] / settings['full_size']])
				original_bounding_box_center = bounding_box_center * scale + top_left
				
				usable = True
				for j in range(len(used_bounding_boxes)):
					used_bounding_box = used_bounding_boxes[j]
					used_bounding_box_center_x = used_bounding_box[0] + used_bounding_box[2] / 2 
					used_bounding_box_center_y = used_bounding_box[1] + used_bounding_box[3] / 2 
					used_bounding_box_center = np.array([used_bounding_box_center_x, used_bounding_box_center_y])
					inter_box_distance = np.linalg.norm(original_bounding_box_center - used_bounding_box_center)
					print('inter_box_distance', inter_box_distance)
					if inter_box_distance < inter_box_distance_threshold:
						usable = False
						break
				if usable:
					break
			#For now, use precense of target_box to detect if we should add bounding box to used_bounding boxes
			image_settings['used_bounding_boxes'].append(image_settings['actual_bounding_box'])
			
		else: #if target bounding box is provided, use that instead. For now, asusme this is a mask.
			target_center = np.array([target_box[0] + target_box[2] / 2, target_box[1] + target_box[3] / 2])
			closest_distance = settings['full_size']
			closest_bounding_box = bounding_boxes[1]
			for i in range(1, len(bounding_boxes)):
				bounding_box = bounding_boxes[i]
				bounding_box_center_x = bounding_box[0] + bounding_box[2] / 2 
				bounding_box_center_y = bounding_box[1] + bounding_box[3] / 2 
				bounding_box_center = np.array([bounding_box_center_x, bounding_box_center_y])
				distance = np.linalg.norm(target_center - bounding_box_center)
				if distance < closest_distance:
					closest_distance = distance
					closest_bounding_box = bounding_box
			bounding_box = closest_bounding_box
	padding = int(settings['full_size'] / 16)
#	padding =0
	sub_x1 = max(bounding_box[0] - padding, 0)
	sub_x2 = min(bounding_box[0] + bounding_box[2] + padding, image.shape[0])
	sub_y1 = max(bounding_box[1] - padding, 0)
	sub_y2 = min(bounding_box[1] + bounding_box[3] + padding, image.shape[1])
	
	mask = np.zeros((image.shape[0], image.shape[1])) 
	mask[sub_x1:sub_x2, sub_y1:sub_y2] = 1.0
	
	bounding_box = [sub_x1, sub_y1, sub_x2 - sub_x1, sub_y2 - sub_y1]
	
	masked_image = None
	if len(image.shape) > 2:
		masked_image = image * np.stack((mask, mask, mask), axis=-1)
		masked_image[:,:,0], bounding_boxes = remove_small_components(masked_image[:,:,0], limit = 1)
		masked_image[:,:,1], bounding_boxes = remove_small_components(masked_image[:,:,1], limit = 1)
		masked_image[:,:,2], bounding_boxes = remove_small_components(masked_image[:,:,2], limit = 1)
	else:
		masked_image = image * mask
		masked_image, bounding_boxes = remove_small_components(masked_image, limit = 1)
	return masked_image, bounding_box


def calculate_metrics_calfin(settings, metrics):
	kernel = settings['kernel']
	image_settings = settings['image_settings']
	pred_image = image_settings['pred_image']
	polyline_image = image_settings['polyline_image']
	mask_final_f32 = image_settings['mask_image']
	meters_per_subset_pixel = image_settings['meters_per_subset_pixel']
	meters_per_256_pixel = image_settings['meters_per_256_pixel']
	domain = image_settings['domain']
	mean_deviation = image_settings['mean_deviation']
	edge_iou_score = image_settings['edge_iou_score']
	year = image_settings['year']
	
	#Calculate and save mean deviation, distances, and IoU metrics based on new masked polyline
	mean_deviation_subset, distances = calculate_mean_deviation(polyline_image[:,:,0], mask_final_f32[:,:,0])
	image_settings['distances'] = distances
	mean_deviation_subset_meters = mean_deviation_subset * meters_per_subset_pixel
	metrics['mean_deviations_pixels'] = np.append(metrics['mean_deviations_pixels'], mean_deviation_subset)
	metrics['mean_deviations_meters'] = np.append(metrics['mean_deviations_meters'], mean_deviation_subset_meters)
	metrics['validation_distances_pixels'] = np.append(metrics['validation_distances_pixels'], distances)
	metrics['validation_distances_meters'] = np.append(metrics['validation_distances_meters'], distances * meters_per_subset_pixel)
	metrics['domain_mean_deviations_pixels'][domain] = np.append(metrics['domain_mean_deviations_pixels'][domain], mean_deviation_subset)
	metrics['domain_mean_deviations_meters'][domain] = np.append(metrics['domain_mean_deviations_meters'][domain], mean_deviation_subset_meters)
	metrics['domain_validation_distances_pixels'][domain] = np.append(metrics['domain_validation_distances_pixels'][domain], distances)
	metrics['domain_validation_distances_meters'][domain] = np.append(metrics['domain_validation_distances_meters'][domain], distances * meters_per_subset_pixel)
	metrics['resolution_deviation_array'] = np.concatenate((metrics['resolution_deviation_array'], np.array([[meters_per_subset_pixel, mean_deviation_subset_meters]])))
	print("mean_deviation_subset: {:.2f} pixels, {:.2f} meters".format(mean_deviation_subset, mean_deviation_subset * meters_per_subset_pixel))
	print("mean_deviation_difference: {:.2f} pixels, {:.2f} meters (- == good)".format(mean_deviation_subset - mean_deviation, mean_deviation_subset * meters_per_subset_pixel - mean_deviation * meters_per_256_pixel))
	
	#Dilate masks for easier visualization and IoU metric calculation.
	polyline_image_dilated = cv2.dilate(polyline_image.astype('float64'), kernel, iterations = 1).astype(np.float32) #np.float32 [0.0, 255.0]
	mask_edge_dilated_f32 = cv2.dilate(mask_final_f32[:,:,0].astype('float64'), kernel, iterations = 1).astype(np.float32) #np.float32 [0.0, 255.0]
	mask_final_dilated_f32 = np.stack((mask_edge_dilated_f32, mask_final_f32[:,:,1]), axis = -1)
	image_settings['polyline_image_dilated'] = polyline_image_dilated
	image_settings['mask_final_dilated_f32'] = mask_final_dilated_f32
	
	#Calculate and save IoU metric
	pred_edge_patch_4d = np.expand_dims(polyline_image_dilated[:,:,0], axis=0)
	mask_edge_patch_4d = np.expand_dims(mask_final_dilated_f32[:,:,0], axis=0)
	pred_mask_patch_4d = np.expand_dims(pred_image[:,:,1], axis=0)
	mask_mask_patch_4d = np.expand_dims(mask_final_dilated_f32[:,:,1], axis=0)
	edge_iou_score_subset = calculate_iou(mask_edge_patch_4d, pred_edge_patch_4d)
	mask_iou_score_subset = calculate_iou(mask_mask_patch_4d, pred_mask_patch_4d)
	image_settings['edge_iou_score_subset'] = edge_iou_score_subset
	image_settings['mask_iou_score_subset'] = mask_iou_score_subset
	metrics['validation_edge_ious'] = np.append(metrics['validation_edge_ious'], edge_iou_score_subset)
	metrics['validation_mask_ious'] = np.append(metrics['validation_mask_ious'], mask_iou_score_subset)
	metrics['domain_validation_edge_ious'][domain] = np.append(metrics['domain_validation_edge_ious'][domain], edge_iou_score_subset)
	metrics['domain_validation_mask_ious'][domain] = np.append(metrics['domain_validation_mask_ious'][domain], mask_iou_score_subset)
	metrics['resolution_iou_array'] = np.concatenate((metrics['resolution_iou_array'], np.array([[meters_per_subset_pixel, edge_iou_score_subset]])))
	print("edge_iou_score_subset: {:.2f}".format(edge_iou_score_subset))
	print("edge_iou_score change {:.2f} (+ == good):".format(edge_iou_score_subset - edge_iou_score))
	
	#Add to calandars
	metrics['domain_validation_calendar'][domain][year] += 1


