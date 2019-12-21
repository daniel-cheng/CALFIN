# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 18:06:26 2019

@author: Daniel
"""

import numpy as np
import sys, cv2
from error_analysis import extract_front_indicators
sys.path.insert(1, '../training/keras-deeplab-v3-plus')
sys.path.insert(2, '../training')

from processing import predict, calculate_mean_deviation, calculate_edge_iou
from plotting import plot_validation_results
from mask_to_shp import mask_to_shp

def remove_small_components(image:np.ndarray):
	image = image.astype('uint8')
	#find all connected components (white blobs in image)
	nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
	#connectedComponentswithStats yields every seperated component with information on each of them, such as size
	sizes = stats[:,cv2.CC_STAT_AREA]
	ordering = np.argsort(-sizes)
	
	#for every component in the image, keep it only if it's above min_size
	min_size_floor = output.size * 0.0001
	if len(ordering) > 1:
		min_size = sizes[ordering[1]] * 0.5
	
	#Isolate large components
	largeComponents = np.zeros((output.shape))
	
	#Store the bounding boxes of components, so they can be isolated and reprocessed further. Default box is entire image.
	bounding_boxes = [[0, 0, image.shape[0], image.shape[1]]]
	#Skip first component, since it's the background color in edge masks
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
	
	#return large component image and bounding boxes for each componnet
	return largeComponents.astype(np.float32), bounding_boxes


def mask_fjord_boundary(fjord_boundary_final_f32, iterations, pred_image_gray_uint8, settings, mask=True):
	""" Helper funcction for performing optimiation on fjord boundaries.
		Erodes a single fjord boundary mask."""
	kernel = settings['kernel']
	full_size = settings['full_size']
	fjord_boundary_eroded_f32 = cv2.erode(fjord_boundary_final_f32.astype('float64'), kernel, iterations = iterations).astype(np.float32) #np.float32 [0.0, 255.0]
	fjord_boundary_eroded_f32 = np.where(fjord_boundary_eroded_f32 > 0.5, 1.0, 0.0)
	
	if mask:
		masked_pred_uint8 = pred_image_gray_uint8 * fjord_boundary_eroded_f32.astype(np.uint8)
		results_polyline = extract_front_indicators(masked_pred_uint8)
		
		#If edge is detected, replace coarse edge mask in prediction image with connected polyline edge
		if not results_polyline is None:
			polyline_image = (results_polyline[0] / 255.0)
			bounding_boxes = [[0, 0, full_size, full_size]]
		else:
			polyline_image = np.zeros((full_size, full_size, 3))
			bounding_boxes = [[0, 0, full_size, full_size]]
		
		#Determine number of masked pixels
		x1, y2 = np.nonzero(polyline_image[:,:,0])
		num_masked_pixels = x1.shape[0]
	else:
		results_polyline = extract_front_indicators(pred_image_gray_uint8)
		
		#If edge is detected, replace coarse edge mask in prediction image with connected polyline edge
		if not results_polyline is None:
			polyline_image = (results_polyline[0] / 255.0)
			polyline_image[:,:,0] *= fjord_boundary_eroded_f32.astype(np.uint8)
			polyline_image[:,:,0], bounding_boxes = remove_small_components(polyline_image[:,:,0])
		else:
			polyline_image = np.zeros((full_size, full_size, 3))
			bounding_boxes = [[0, 0, full_size, full_size]]
		
		#Determine number of masked pixels
		x1, y2 = np.nonzero(polyline_image[:,:,0])
		num_masked_pixels = x1.shape[0]
	return num_masked_pixels, polyline_image, fjord_boundary_eroded_f32, bounding_boxes


def mask_polyline(raw_image, pred_image, fjord_boundary_final_f32, settings):
	""" Perform optimiation on fjord boundaries.
		Continuously erode fjord boundary mask until fjord edges are masked.
		This is detected by looking for large increases in pixels being masked,
		followed by few pixels being masked."""
	num_masked_pixels_array = np.array([])
	masked_pixels_diff = np.array([0])
	iteration_limit = 6 #limit of 6 pixels of erosion - otherwise, too much front detail is lost anyways
	raw_gray_uint8 = (raw_image[:,:,0] * 255.0).astype(np.uint8)
	pred_gray_uint8 = (pred_image * 255.0).astype(np.uint8)
	for j in range(iteration_limit):
		results_masking = mask_fjord_boundary(fjord_boundary_final_f32, j, pred_gray_uint8, settings)
		num_masked_pixels = results_masking[0]
		num_masked_pixels_array = np.append(num_masked_pixels_array, num_masked_pixels)
		if j > 0:
			masked_pixels_diff = np.append(masked_pixels_diff, num_masked_pixels_array[j] - num_masked_pixels_array[j - 1])
	
	#Find where the marginal increase in masked pixels decreases the most (i.e., find where increase j gives diminishing returns).
	#Use 1st derivative of number of masked pixels with respect to iteration j.
	#Find a minima (not even guarenteed to be local min) using argmin of (d/dj)(number of masked pixels)
	#Additionally, check ahead one iteration to ensure we get a better minima
	#Unable to use second derivative since it never crosses 0 (number of masked pixels is a monotonically decreasing function)
	maximal_erosions = np.argmin(masked_pixels_diff)
	original_maximal_erosions = maximal_erosions
	threshold = 0.4
	while maximal_erosions < iteration_limit - 1 and masked_pixels_diff[maximal_erosions + 1] < masked_pixels_diff[original_maximal_erosions] * threshold:
		maximal_erosions += 1
		threshold = threshold / 2
	
	#Redo fjord masking with known maximal erosion number
	results_masking = mask_fjord_boundary(fjord_boundary_final_f32, maximal_erosions, pred_gray_uint8, settings, mask=False)
	polyline_image = results_masking[1]
	fjord_boundary_eroded = results_masking[2]
	bounding_boxes = results_masking[3]
	
	return polyline_image, fjord_boundary_eroded, bounding_boxes


def mask_bounding_box(bounding_boxes, image):
	bounding_box = bounding_boxes[1]
	sub_x1 = max(bounding_box[0] - 8, 0)
	sub_x2 = min(bounding_box[0] + bounding_box[2] + 8, image.shape[0])
	sub_y1 = max(bounding_box[1] - 8, 0)
	sub_y2 = min(bounding_box[1] + bounding_box[3] + 8, image.shape[1])
	
	mask = np.zeros((image.shape[0], image.shape[1])) 
	mask[sub_x1:sub_x2, sub_y1:sub_y2] = 1.0
	
	masked_image = None
	if len(image.shape) > 2:
		masked_image = image * np.stack((mask, mask, mask), axis=-1)
	else:
		masked_image = image * mask
	
	return masked_image


def calculate_metrics_calfin(settings, metrics):
	kernel = settings['kernel']
	image_settings = settings['image_settings']
	polyline_image = image_settings['polyline_image']
	mask_final_f32 = image_settings['mask_image']
	meters_per_subset_pixel = image_settings['meters_per_subset_pixel']
	meters_per_256_pixel = image_settings['meters_per_256_pixel']
	domain = image_settings['domain']
	mean_deviation = image_settings['mean_deviation']
	edge_iou_score = image_settings['edge_iou_score']
	
	#Calculate and save mean deviation, distances, and IoU metrics based on new masked polyline
	mean_deviation_subset, distances = calculate_mean_deviation(polyline_image[:,:,0], mask_final_f32)
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
	mask_final_dilated_f32 = cv2.dilate(mask_final_f32.astype('float64'), kernel, iterations = 1).astype(np.float32) #np.float32 [0.0, 255.0]
	image_settings['polyline_image_dilated'] = polyline_image_dilated
	image_settings['mask_final_dilated_f32'] = mask_final_dilated_f32
	
	#Calculate and save IoU metric
	pred_patch_4d = np.expand_dims(polyline_image_dilated[:,:,0], axis=0)
	mask_patch_4d = np.expand_dims(mask_final_dilated_f32, axis=0)
	edge_iou_score_subset = calculate_edge_iou(mask_patch_4d, pred_patch_4d)
	image_settings['edge_iou_score_subset'] = edge_iou_score_subset
	metrics['validation_ious'] = np.append(metrics['validation_ious'], edge_iou_score_subset)
	metrics['domain_validation_ious'][domain] = np.append(metrics['domain_validation_ious'][domain], edge_iou_score_subset)
	metrics['resolution_iou_array'] = np.concatenate((metrics['resolution_iou_array'], np.array([[meters_per_subset_pixel, edge_iou_score_subset]])))
	print("edge_iou_score_subset: {:.2f}".format(edge_iou_score_subset))
	print("edge_iou_score change {:.2f} (+ == good):".format(edge_iou_score_subset - edge_iou_score))


def postprocess(i, validation_files, settings, metrics):
	kernel = settings['kernel']
	image_settings = settings['image_settings']
	resolution_1024 = image_settings['resolution_1024']
	meters_per_1024_pixel = image_settings['meters_per_1024_pixel']
	image_name_base = image_settings['image_name_base']
	img_3_uint8 = image_settings['img_3_uint8']
	mask_uint8 = image_settings['mask_uint8']
	fjord_boundary = image_settings['fjord_boundary']
	
	raw_image = image_settings['raw_image']
	pred_image = image_settings['pred_image']
	mask_image = image_settings['mask_image']
	fjord_boundary_final_f32 = image_settings['fjord_boundary_final_f32']
	
	#recalculate scaling
	resolution_256 = (raw_image.shape[0] + raw_image.shape[1])/2
	meters_per_256_pixel = resolution_1024 / resolution_256  * meters_per_1024_pixel
	
	#Perform optimiation on fjord boundaries.
	#Continuously erode fjord boundary mask until fjord edges are masked.
	#This is detected by looking for large increases in pixels being masked,
	#followed by few pixels being masked.
	results_pred = mask_polyline(raw_image, pred_image[:,:,0], fjord_boundary_final_f32, settings)
	results_mask = mask_polyline(raw_image, mask_image, fjord_boundary_final_f32, settings)
	polyline_image = results_pred[0]
	bounding_boxes = results_pred[2]
	mask_final_f32 = results_mask[0][:,:,0]
	
	#Calculate and save mean deviation, distances, and IoU metrics based on new masked polyline
	mean_deviation, distances = calculate_mean_deviation(polyline_image[:,:,0], mask_final_f32)
	image_settings['distances'] = distances
	print("mean_deviation: {:.2f} pixels, {:.2f} meters".format(mean_deviation, mean_deviation * meters_per_256_pixel))
	
	#Dilate masks for easier visualization and IoU metric calculation.
	polyline_image_dilated = cv2.dilate(polyline_image.astype('float64'), kernel, iterations = 1).astype(np.float32) #np.float32 [0.0, 255.0]
	mask_final_dilated_f32 = cv2.dilate(mask_final_f32.astype('float64'), kernel, iterations = 1).astype(np.float32) #np.float32 [0.0, 255.0]
	image_settings['polyline_image'] = polyline_image_dilated
	image_settings['mask_image'] = mask_final_dilated_f32
	
	#Calculate and save IoU metric
	pred_patch_4d = np.expand_dims(polyline_image_dilated[:,:,0], axis=0)
	mask_patch_4d = np.expand_dims(mask_final_dilated_f32, axis=0)
	edge_iou_score = calculate_edge_iou(mask_patch_4d, pred_patch_4d)
	print("edge_iou_score:  {:.2f}".format(edge_iou_score))
	
	#For each calving front, subset the image AGAIN and predict. This helps accuracy for inputs with large scaling/downsampling ratios
	box_counter = 0	
	found_front = False
	image_settings['i'] = i
	image_settings['box_counter'] = box_counter
	image_settings['image_name_base'] = image_name_base
	image_settings['img_3_uint8'] = img_3_uint8
	image_settings['mask_uint8'] = mask_uint8 
	image_settings['fjord_boundary'] = fjord_boundary
	image_settings['meters_per_1024_pixel'] = meters_per_1024_pixel
	image_settings['meters_per_256_pixel'] = meters_per_256_pixel
	image_settings['resolution_1024'] = resolution_1024
	image_settings['resolution_256'] = resolution_256
	image_settings['mean_deviation'] = mean_deviation
	image_settings['edge_iou_score'] = edge_iou_score
	
	#Try to find a front in each bounding box, if any
	if len(bounding_boxes) > 1:
		for bounding_box in bounding_boxes[1:]:
			image_settings['bounding_box'] = bounding_box
			found_front_in_box, metrics = reprocess(settings, image_settings, metrics)
			found_front = found_front or found_front_in_box
	
	#if no fronts are found yet, fallback to box that encloses entire image
	if not found_front:
		image_settings['bounding_box'] = bounding_boxes[0]
		found_front, metrics = reprocess(settings, image_settings, metrics)
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


def reprocess(settings, image_settings, metrics): 
	model = settings['model']
	pred_norm_image = settings['pred_norm_image']
	full_size = settings['full_size']
	img_size = settings['img_size']
	stride = settings['stride']
	mask_confidence_strength_threshold = settings['mask_confidence_strength_threshold']
	edge_confidence_strength_threshold = settings['edge_confidence_strength_threshold']
	domain_scalings = settings['domain_scalings']
	plotting = settings['plotting']
	
	#image_settings are assigned per front in the image
	domain = image_settings['domain']
	box_counter = image_settings['box_counter']
	bounding_box = image_settings['bounding_box']
	img_3_uint8 = image_settings['img_3_uint8']
	mask_uint8 = image_settings['mask_uint8']
	fjord_boundary = image_settings['fjord_boundary']
	meters_per_1024_pixel = image_settings['meters_per_1024_pixel']
	resolution_1024 = image_settings['resolution_1024']
	resolution_256 = image_settings['resolution_256']
			
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
	sub_padding_ratio = 2.5
	half_sub_padding_ratio = sub_padding_ratio / 2
	sub_padding = sub_length * half_sub_padding_ratio
#					sub_padding = sub_length / 2 + 128
	
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
		
	#calculate new subset x/y bounds, which are guarenteed to be of sufficient size and
	#to be within the bounds of the image
	sub_x1 = int(sub_center_x - sub_padding)
	sub_x2 = int(sub_center_x + sub_padding)
	sub_y1 = int(sub_center_y - sub_padding)
	sub_y2 = int(sub_center_y + sub_padding)
	
	scaling = resolution_256 / resolution_1024
	actual_bounding_box = [sub_x1 * scaling, sub_y1 * scaling, sub_padding * 2 * scaling, sub_padding * 2 * scaling]
	print("bounding_box:", bounding_box, "fractional_bounding_box:", fractional_bounding_box, 'actual_bounding_box', actual_bounding_box)
	image_settings['actual_bounding_box'] = actual_bounding_box
	
	#Perform subsetting
	resolution_subset = sub_padding * 2 #not simplified for clarity - just find average dimensions	
	meters_per_subset_pixel = meters_per_1024_pixel
	img_3_subset_uint8 = img_3_uint8[sub_x1:sub_x2, sub_y1:sub_y2, :]
	mask_subset_uint8 = mask_uint8[sub_x1:sub_x2, sub_y1:sub_y2]
	fjord_subset_boundary = fjord_boundary[sub_x1:sub_x2, sub_y1:sub_y2]
	
	#Repredict
	results = predict(model, img_3_subset_uint8, mask_subset_uint8, fjord_subset_boundary, pred_norm_image, full_size, img_size, stride)
	raw_image = results[0]
	pred_image = results[1]
	mask_image = results[2]
	fjord_boundary_final_f32 = results[3]
	image_settings['results'] = results
	image_settings['raw_image'] = results[0]
	image_settings['pred_image'] = results[1]
	image_settings['mask_image'] = results[2]
	image_settings['fjord_boundary_final_f32'] = results[3]
	
	edge_confidence_strength_indices = np.nan_to_num(pred_image[:,:,0]) > 0.05
	edge_confidence_strength = np.mean(np.abs(0.5 - np.nan_to_num(pred_image[:,:,0][edge_confidence_strength_indices])))
	mask_confidence_strength_indices = np.nan_to_num(pred_image[:,:,1]) > 0.05
	mask_confidence_strength = np.mean(np.abs(0.5 - np.nan_to_num(pred_image[:,:,1][mask_confidence_strength_indices])))
	
	print("edge_confidence_strength: {:.3f}".format(edge_confidence_strength * 2))
	print("mask_confidence_strength: {:.3f}".format(mask_confidence_strength * 2))
	if mask_confidence_strength * 2 < mask_confidence_strength_threshold or edge_confidence_strength * 2 < edge_confidence_strength_threshold:
		print("not confident, skipping")
		metrics['confidence_skip_count'] += 1
		return found_front, metrics
		#edge_confidence_strength 0.36093274 mask_confidence_strength 0.48205513
		#edge_confidence_strength 0.2558244 mask_confidence_strength 0.38819787
	
	#recalculate scaling
	meters_per_subset_pixel = resolution_subset / resolution_256  * meters_per_1024_pixel
	image_settings['meters_per_subset_pixel'] = meters_per_subset_pixel
	if domain not in domain_scalings:
		domain_scalings[domain] = meters_per_subset_pixel
	
	#Redo masking
	results_pred = mask_polyline(raw_image, pred_image[:,:,0], fjord_boundary_final_f32, settings)
	results_mask = mask_polyline(raw_image, mask_image, fjord_boundary_final_f32, settings)
	polyline_image = results_pred[0]
	bounding_boxes_pred = results_pred[2]
	mask_final_f32 = results_mask[0][:,:,0]
	fjord_boundary_eroded_f32 = results_mask[1]
	image_settings['fjord_boundary_eroded_f32'] = fjord_boundary_eroded_f32

	#mask out largest front
	if len(bounding_boxes_pred) < 2:
		print("no front detected, skipping")
		metrics['no_detection_skip_count'] += 1
		return found_front, metrics
	polyline_image = mask_bounding_box(bounding_boxes_pred, polyline_image)
	mask_final_f32 = mask_bounding_box(bounding_boxes_pred, mask_final_f32)
	image_settings['polyline_image'] = polyline_image
	image_settings['mask_image'] = mask_final_f32
	
	#Calculate validation metrics
	calculate_metrics_calfin(settings, metrics)
	
	#Plot results
	if plotting:
		plot_validation_results(settings, metrics)
		
	#Generate shape file
	mask_to_shp(settings, metrics)
	
	#Notify rest of the code that a front has been found, so that we don't use fallback front
	metrics['front_count'] += 1
	found_front = True
	
	return found_front, metrics
