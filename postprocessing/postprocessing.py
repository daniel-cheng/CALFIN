# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 18:06:26 2019

@author: Daniel
"""

import sys, cv2, random
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.signal import find_peaks
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
    """Postprocessing main entry function."""
    if 'bypass' in settings and settings['bypass'] == True:
        return
    if settings['driver'] == 'production':
        postprocess_production(settings, metrics)
    else:
        postprocess_validated(settings, metrics)


def postprocess_validated(settings, metrics):
    """Perform initial front extraction from manually masked images."""
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
    
    results_pred = mask_polyline(pred_image[:, :, 0], fjord_boundary_final_f32, settings)
    results_mask = mask_polyline(mask_image[:, :, 0], fjord_boundary_final_f32, settings)
    polyline_image = np.stack((results_pred[0], empty_image, empty_image), axis=-1)
    
    bounding_boxes = results_pred[1]
    mask_edge_f32 = results_mask[0]
    
    #Calculate and save mean deviation, distances, and IoU metrics based on new masked polyline
    mean_deviation, distances = calculate_mean_deviation(polyline_image[:, :, 0], mask_edge_f32)
    image_settings['distances'] = distances
    print('\t' + "mean_deviation: {:.2f} pixels, {:.2f} meters".format(mean_deviation, mean_deviation * meters_per_256_pixel))
    
    #Dilate masks for easier visualization and IoU metric calculation.
    polyline_image_dilated = cv2.dilate(polyline_image.astype(np.float64), kernel, iterations=1).astype(np.float32) #np.float32 [0.0, 255.0]
    mask_edge_dilated_f32 = cv2.dilate(mask_edge_f32.astype(np.float64), kernel, iterations=1).astype(np.float32) #np.float32 [0.0, 255.0]
    mask_final_dilated_f32 = np.stack((mask_edge_dilated_f32, mask_image[:, :, 1]), axis=-1)
    image_settings['polyline_image'] = polyline_image_dilated
    image_settings['mask_image'] = mask_final_dilated_f32
    
    #Calculate and save IoU metric
    pred_patch_4d = np.expand_dims(polyline_image_dilated[:, :, 0], axis=0)
    mask_patch_4d = np.expand_dims(mask_final_dilated_f32[:, :, 0], axis=0)
    edge_iou_score = calculate_iou(mask_patch_4d, pred_patch_4d)
    print('\t' + "edge_iou_score:  {:.2f}".format(edge_iou_score))
    
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
        print('\t' + 'potential fronts: ', len(bounding_boxes) - 1)
        print('\t' + 'bounding_boxes', bounding_boxes)
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
    """Process individual isolated fronts, based on initial detection."""
    domain_scalings = settings['domain_scalings']
    empty_image = settings['empty_image']
    
    #image_settings are assigned per front in the image
    image_settings = settings['image_settings']
    domain = image_settings['domain']
    box_counter = image_settings['box_counter']
    img_3_uint8 = image_settings['unprocessed_original_raw']
    mask_uint8 = image_settings['unprocessed_original_mask']
    fjord_boundary = image_settings['unprocessed_original_fjord_boundary']
    meters_per_1024_pixel = image_settings['meters_per_1024_pixel']
    resolution_256 = image_settings['resolution_256']
            
    image_settings['box_counter'] += 1
    box_counter += 1
    found_front = False
    
    #Perform subsetting
    sub_x1, sub_x2, sub_y1, sub_y2, sub_padding = resubset(settings)
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
    
    #Confidence estimation filtering
    edge_pred = pred_image[:, :, 0]
    mask_pred = pred_image[:, :, 1]
    confidences = estimate_confidence(settings, edge_pred, mask_pred, fjord_boundary)
    edge_unconfident = confidences['edge_unconfident']
    edge_size_unconfident = confidences['edge_size_unconfident']
    mask_ratio_unconfident = confidences['mask_ratio_unconfident']
    mask_edge_buffered_unconfident = confidences['mask_edge_buffered_unconfident']
    
    #mask_edge_buffered_unconfident indicates likely issues, but not definitive
    #(thus front is not discarded if no other condition fails)
    if  mask_edge_buffered_unconfident:
        print('\t' + "not confident (mask_edge_buffer unconfident), skipping")
        metrics['confidence_skip_count'] += 1
        return found_front, metrics
    if edge_unconfident or edge_size_unconfident or mask_ratio_unconfident:
        print('\t' + "not confident (edge, edge_size, or mask_ratio unconfident), skipping")
        metrics['confidence_skip_count'] += 1
        return found_front, metrics

    #recalculate scaling
    meters_per_subset_pixel = resolution_subset / resolution_256  * meters_per_1024_pixel
    image_settings['meters_per_subset_pixel'] = meters_per_subset_pixel
    if domain not in domain_scalings:
        domain_scalings[domain] = meters_per_subset_pixel
    
    #Redo masking
    results_pred = mask_polyline(pred_image[:, :, 0], fjord_boundary, settings)
    results_mask = mask_polyline(mask_image[:, :, 0], fjord_boundary, settings)
    polyline_image = np.stack((results_pred[0], empty_image, empty_image), axis=-1)
    
    bounding_boxes_pred = results_pred[1]
    polylines_coords_pred = results_pred[2]
    mask_edge_f32 = results_mask[0]
    bounding_boxes_mask = results_mask[1]
    polylines_coords_mask = results_mask[2]
    
    #mask out largest front
    if len(bounding_boxes_pred) < 2 or len(bounding_boxes_mask) < 2:
        print("no front detected, skipping")
        metrics['no_detection_skip_count'] += 1
        return found_front, metrics
    results_bounded_pred = mask_bounding_box(bounding_boxes_pred, polyline_image, settings, polylines_coords_pred, pred_image[:, :, 1], store_box=False)
    results_bounded_mask = mask_bounding_box(bounding_boxes_mask, mask_edge_f32, settings, polylines_coords_mask, mask_image[:, :, 1])
    
        #Fail if the bounded front isn't within the original detection
    if results_bounded_pred is None or results_bounded_mask is None:
        print('\t' + "not confident (results_bounded out of range), skipping")
        metrics['confidence_skip_count'] += 1
        return found_front, metrics
    
    polyline_image, bounding_box_pred, polyline_coords_pred = results_bounded_pred
    mask_edge_f32, bounding_box_mask, polyline_coords_mask = results_bounded_mask
        
    mask_final_f32 = np.stack((mask_edge_f32[:, :, 0], mask_image[:, :, 1]), axis=-1)
    
    image_settings['polyline_coords'] = polyline_coords_pred
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
    mask_to_shp(settings)
    
    #Notify rest of the code that a front has been found, so that we don't use fallback front
    metrics['front_count'] += 1
    found_front = True
    
    return found_front, metrics


def postprocess_production(settings, metrics):
    """Perform initial front detection using NN processed raw inputs."""
    print('postprocess_production')
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
    results_pred = mask_polyline(pred_image[:, :, 0], fjord_boundary_final_f32, settings)
    polyline_image = np.stack((results_pred[0], empty_image, empty_image), axis=-1)
    bounding_boxes = results_pred[1]
    
    #Dilate masks for easier visualization and IoU metric calculation.
    polyline_image_dilated = cv2.dilate(polyline_image.astype(np.float64), kernel, iterations=1).astype(np.float32) #np.float32 [0.0, 255.0]
    image_settings['polyline_image'] = polyline_image_dilated
    
    #For each calving front, subset the image AGAIN and predict. This helps accuracy for inputs with large scaling/downsampling ratios
    box_counter = 0
    found_any_front = False
    image_settings['box_counter'] = box_counter
    image_settings['meters_per_256_pixel'] = meters_per_256_pixel
    image_settings['resolution_256'] = resolution_256
    image_settings['used_bounding_boxes'] = []
    
    #Try to find a front in each bounding box, if any
    if len(bounding_boxes) > 1:
        print('\t' + 'potential fronts: ', len(bounding_boxes) - 1)
        print('\t' + 'bounding_boxes', bounding_boxes)
        for bounding_box in bounding_boxes[1:]:
            progress = str(image_settings['box_counter'] + 1) + '/' + str(len(bounding_boxes) - 1)
            print('Handling bounding_box ' + progress + ':', bounding_box)
            image_settings['bounding_box'] = bounding_box
            found_front, metrics = reprocess_production(settings, metrics)
            found_any_front = found_any_front or found_front
    
    #if no fronts are found yet, fallback to box that encloses entire image
    #Due to bounding box selection, this allows for one final detection 
    #of 1 more missing front.
    if not found_any_front:
        box_counter = 0
        print('Rehandling bounding_box:', bounding_boxes[0])
        image_settings['box_counter'] = box_counter
        image_settings['bounding_box'] = bounding_boxes[0]
        found_front, metrics = reprocess_production(settings, metrics)
        #If we still haven't found a front in the default, we "skip" the image and make a note of it.
        found_any_front = found_any_front or found_front
        if not found_any_front:
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
    """Process individual isolated fronts, based on initial detection."""
    print('' + 'reprocess_production')
    domain_scalings = settings['domain_scalings']
    empty_image = settings['empty_image']
    
    #image_settings are assigned per front in the image
    image_settings = settings['image_settings']
    domain = image_settings['domain']
    box_counter = image_settings['box_counter']
    img_3_uint8 = image_settings['unprocessed_original_raw']
    fjord_boundary = image_settings['unprocessed_original_fjord_boundary']
    meters_per_1024_pixel = image_settings['meters_per_1024_pixel']
    resolution_256 = image_settings['resolution_256']
            
    image_settings['box_counter'] += 1
    box_counter += 1
    found_front = False
    
    #Perform subsetting
    sub_x1, sub_x2, sub_y1, sub_y2, sub_padding = resubset(settings)
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
    
    #Confidence estimation filtering
    edge_pred = pred_image[:, :, 0]
    mask_pred = pred_image[:, :, 1]
    confidences = estimate_confidence(settings, edge_pred, mask_pred, fjord_boundary)
    edge_unconfident = confidences['edge_unconfident']
    edge_size_unconfident = confidences['edge_size_unconfident']
    mask_ratio_unconfident = confidences['mask_ratio_unconfident']
    mask_edge_buffered_unconfident = confidences['mask_edge_buffered_unconfident']
    
    #mask_edge_buffered_unconfident indicates likely issues, but not definitive
    #(thus front is not discarded if no other condition fails)
    if  mask_edge_buffered_unconfident:
        print('\t' + "not confident (mask_edge_buffer unconfident), skipping")
        metrics['confidence_skip_count'] += 1
        return found_front, metrics
    else:
        if edge_unconfident or edge_size_unconfident or mask_ratio_unconfident:
            print('\t' + "not confident (edge, edge_size, or mask_ratio unconfident), skipping")
            metrics['confidence_skip_count'] += 1
            return found_front, metrics
    
    #recalculate scaling
    meters_per_subset_pixel = resolution_subset / resolution_256  * meters_per_1024_pixel
    image_settings['meters_per_subset_pixel'] = meters_per_subset_pixel
    if domain not in domain_scalings:
        domain_scalings[domain] = meters_per_subset_pixel
    
    #Redo masking
    results_pred = mask_polyline(edge_pred, fjord_boundary, settings)
    polyline_image = np.stack((results_pred[0], empty_image, empty_image), axis=-1)
    bounding_boxes_pred = results_pred[1]
    polylines_coords = results_pred[2]
    
    #mask out largest front
    if len(bounding_boxes_pred) < 2:
        print('\t' + "no front detected, skipping")
        metrics['no_detection_skip_count'] += 1
        return found_front, metrics
    results_bounded = mask_bounding_box(bounding_boxes_pred, polyline_image, settings, polylines_coords, mask_pred)
    
    #Fail if the bounded front isn't within the original detection
    if results_bounded is None:
        print('\t' + "not confident (results_bounded out of range), skipping")
        metrics['confidence_skip_count'] += 1
        return found_front, metrics
    
    polyline_image, bounding_box_pred, polyline_coords = results_bounded
    image_settings['polyline_coords'] = polyline_coords
    image_settings['bounding_box_pred'] = bounding_box_pred
    image_settings['polyline_image'] = polyline_image
#    plt.figure(11500 + random.randint(1,500))
#    plt.imshow(polyline_image)
#    plt.show()
    
    #Plot and save results
    kernel = settings['kernel']
    polyline_image_dilated = cv2.dilate(polyline_image.astype(np.float64), kernel, iterations=1).astype(np.float32) #np.float32 [0.0, 255.0]
    image_settings['polyline_image_dilated'] = polyline_image_dilated
    plot_production_results(settings, metrics)
    
    #Generate shape file
    mask_to_shp(settings)
    
    #Notify rest of the code that a front has been found, so that we don't use fallback front
    metrics['front_count'] += 1
    found_front = True
    
    return found_front, metrics


def resubset(settings):
    """Resubset the raw input based on detected calving front boundary box."""
    full_size = settings['full_size']
    image_settings = settings['image_settings']
    bounding_box = image_settings['bounding_box']
    img_3_uint8 = image_settings['unprocessed_original_raw']
    
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
    actual_bounding_box = [sub_x1 / img_3_uint8.shape[0] * full_size, sub_y1 / img_3_uint8.shape[1] * full_size, (sub_x2 - sub_x1) / img_3_uint8.shape[0] * full_size, (sub_y2 - sub_y1) / img_3_uint8.shape[1] * full_size]
    image_settings['actual_bounding_box'] = actual_bounding_box
    
    #Calculate the position of the original bounding box in the new subset
    image_settings['target_center'] = np.array([bounding_box[0] + bounding_box[2] / 2, bounding_box[1] + bounding_box[3] / 2])
#    print(image_settings['target_center'])
            
    return sub_x1, sub_x2, sub_y1, sub_y2, sub_padding


def estimate_confidence(settings, edge_pred, mask_pred, fjord_boundary):
    """Calculate confidence measures that estimate the quality of the extracted calving front."""
    edge_confidence_strength_threshold = settings['edge_confidence_strength_threshold']
    edge_detection_size_threshold = settings['edge_detection_size_threshold']
    mask_edge_buffered_mean_threshold = settings['mask_detection_ratio_threshold']
    edge_detection_threshold = settings['edge_detection_threshold']
    mask_detection_threshold = settings['mask_detection_threshold']
    confidence_kernel = settings['confidence_kernel']
    line_thickness = settings['line_thickness']
    
    #Calculate edge/mask seperation from 0.5 (more speration = more confidence)
    #Remove bottom 0.05 of edge mask to only caclulate confidence of detected edge and not entire image
    #Mask confidence doesn't need it but the thresholds have already been tuned. `\_(^u^)_/`
    edge_detection_size_indices = edge_pred > edge_detection_threshold
    edge_detection_size = np.sum(edge_detection_size_indices, axis=(0, 1)) / line_thickness
    mask_detection_ratio_indices = mask_pred > mask_detection_threshold
    mask_detection_size = np.sum(mask_detection_ratio_indices, axis=(0, 1))
    mask_nondetection_size = np.size(mask_pred) - mask_detection_size
    mask_detection_ratio = mask_detection_size / (mask_nondetection_size + 1)
    edge_confidence_strength_indices = edge_pred > 0.05
    edge_confidence_strength = np.mean(np.abs(0.5 - edge_pred[edge_confidence_strength_indices]))
    mask_confidence_strength_indices = mask_pred > 0.05
    mask_confidence_strength = np.mean(np.abs(0.5 - mask_pred[mask_confidence_strength_indices]))
    
    #Mask edge buffered confidence
    edge_binary = np.zeros(edge_pred.shape).astype(np.float64)
    edge_confidence_strength_indices = edge_pred > 0.33
    edge_binary[edge_confidence_strength_indices] = 255.0
    edge_binary_dilated = cv2.dilate(edge_binary, confidence_kernel, iterations=1)
    edge_binary_dilated_indices = edge_binary_dilated.astype(np.float32) > 127.0
    mask_edge_buffered = np.zeros(mask_pred.shape) + 0.5
    mask_edge_buffered[edge_binary_dilated_indices] = mask_pred[edge_binary_dilated_indices]
    mask_pred_normalized = mask_edge_buffered - 0.5
    nonzeros_mask_edge_buffered = mask_pred_normalized[edge_binary_dilated_indices]
    mask_edge_buffered_confidence_strength_unisolated = np.mean(nonzeros_mask_edge_buffered)
    
    #Mask edge buffered confidence with fjord masking
    edge_binary = np.zeros(edge_pred.shape).astype(np.float64)
    fjord_boundary_dilated = cv2.erode(fjord_boundary.astype(np.float64), confidence_kernel, iterations=1)
    edge_pred_isolated = np.where(fjord_boundary_dilated.astype(np.float32) > 127.0, 1.0, 0.0) * edge_pred
    edge_confidence_strength_indices = edge_pred_isolated > 0.33
    edge_binary[edge_confidence_strength_indices] = 255.0
    edge_binary_dilated = cv2.dilate(edge_binary, confidence_kernel, iterations=1)
    edge_binary_dilated_indices = edge_binary_dilated.astype(np.float32) > 127.0
    mask_edge_buffered = np.zeros(mask_pred.shape) + 0.5
    mask_edge_buffered[edge_binary_dilated_indices] = mask_pred[edge_binary_dilated_indices]
    mask_pred_normalized = mask_edge_buffered - 0.5
    nonzeros_mask_edge_buffered = mask_pred_normalized[edge_binary_dilated_indices]
    mask_edge_buffered_confidence_strength_isolated = np.mean(nonzeros_mask_edge_buffered)
    
    #Take the best score of both mask edge buffered confidence measures
    mask_edge_buffered_confidence_strength = np.min([mask_edge_buffered_confidence_strength_unisolated + 0.5, mask_edge_buffered_confidence_strength_isolated + 0.5]) - 0.5
    nonzeros_mask_edge_buffered = mask_pred_normalized[edge_binary_dilated_indices]
    mask_edge_buffered_confidence_strength = np.mean(nonzeros_mask_edge_buffered)
    
    #print metrics and calculate final filtering booleans
    print('\t\t' + "edge_detection_size: {:.3f} pixels".format(edge_detection_size))
    print('\t\t' + "mask_detection_ratio: {:.3f}".format(mask_detection_ratio))
    print('\t\t' + "edge_confidence_strength: {:.3f}".format(edge_confidence_strength * 2))
    print('\t\t' + "mask_confidence_strength: {:.3f}".format(mask_confidence_strength * 2))
    print('\t\t' + "mask_edge_buffered_confidence_mean: {:.3f}".format(mask_edge_buffered_confidence_strength))
    edge_unconfident = edge_confidence_strength * 2 < edge_confidence_strength_threshold
    edge_size_unconfident = edge_detection_size < edge_detection_size_threshold
    mask_ratio_unconfident = mask_detection_ratio > mask_edge_buffered_mean_threshold
    mask_edge_buffered_unconfident = np.abs(mask_edge_buffered_confidence_strength) > settings['mask_edge_buffered_mean_threshold']
    
    #return confidence results
    confidences = dict()
    confidences['edge_unconfident'] = edge_unconfident
    confidences['edge_size_unconfident'] = edge_size_unconfident
    confidences['mask_ratio_unconfident'] = mask_ratio_unconfident
    confidences['mask_edge_buffered_unconfident'] = mask_edge_buffered_unconfident
    return confidences


def remove_small_components(image, limit=np.inf, min_size_percentage=0.0001):
    """Removes small connected regions from an image."""
    image = image.astype('uint8')
    #find all connected components (white blobs in image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    sizes = stats[:, cv2.CC_STAT_AREA]
    ordering = np.argsort(-sizes)
    
    #for every component in the image, keep it only if it's above min_size
    min_size = output.size * min_size_percentage
    if len(ordering) > 1:
        min_size = max(min_size, sizes[ordering[1]] * 0.15)
    
    #Isolate large components
    large_components = np.zeros((output.shape))
    
    #Store the bounding boxes of components, so they can be isolated and reprocessed further. Default box is entire image.
    bounding_boxes = [[0, 0, image.shape[0], image.shape[1]]]
    #Skip first component, since it's the background color in edge masks
    #Restrict number of components returned depending on limit
    number_returned = 0
    for i in range(1, len(sizes)):
        if sizes[ordering[i]] >= min_size:
            mask_indices = output == ordering[i]
            x, y = np.nonzero(mask_indices)
            min_x = min(x)
            delta_x = max(x) - min_x
            min_y = min(y)
            delta_y = max(y) - min_y
            bounding_boxes.append([min_x, min_y, delta_x, delta_y])
            large_components[mask_indices] = image[mask_indices]
            number_returned += 1
            if number_returned >= limit:
                break
    #return large component image and bounding boxes for each componnet
    return large_components.astype(np.float32), bounding_boxes


def mask_polyline(pred_image, fjord_boundary_final_f32, settings, min_size_percentage=0.00025):
    """ Extracts polyline from pixel masks.
        First, perform optimiation on fjord boundaries by isolating contiginous pixels far away from fjord boundaries.
        Then, remove pixels that are still too close to fjord boundaries.
        Finally, return ordered polyline. If no detections exist, return None/empty results."""
#    print('\t\t' + 'mask_polyline')
    #get distance of each point from fjord boundary black pixel
    fjord_distances = distance_transform_edt(fjord_boundary_final_f32)
    results_polyline = extract_front_indicators(pred_image)
    #If front is detected, proceed with extraction
    if not results_polyline is None:
        polyline_image = results_polyline[0][:, :, 0] / 255.0
        polylines_coords = [results_polyline[1]]

        #Check to see if there are any fjord boundaries to mask
        if fjord_boundary_final_f32.min() > 127:
            polyline_image, bounding_boxes = remove_small_components(polyline_image, min_size_percentage=min_size_percentage)
            return polyline_image, bounding_boxes, polylines_coords
        
        #No intersections with fjord - just treat as if no boundaries
#        if fjord_distances.min() > 4:
#            polyline_image, bounding_boxes = remove_small_components(polyline_image, min_size_percentage=min_size_percentage)
#            return polyline_image, bounding_boxes, polyline_coords
        
#        plt.figure(10000 + random.randint(1,500))
#        plt.imshow(pred_image)
#        plt.show()
#        
#        plt.figure(10500 + random.randint(1,500))
#        plt.imshow(polyline_image)
#        plt.show()
        
        #Find all pixels where distance to nearest fjord boundary is an outlier
        polyline_distances_image = fjord_distances * polyline_image
        polyline_coords_distances = polyline_distances_image[(polylines_coords[0][0], polylines_coords[0][1])]
        data = polyline_coords_distances
        
        #stop gap FIX - fix antarctic/no fjord handling
        num_bins = max(np.floor(data.max() / 2).astype(int), 12)
        if num_bins == 0:
            polyline_image = settings['empty_image']
            bounding_boxes = [[0, 0, settings['full_size'], settings['full_size']]]
            polylines_distances = [fjord_distances[(polylines_coords[0][0], polylines_coords[0][1])]]
            return polyline_image, bounding_boxes, polylines_coords, polylines_distances
        
        #find inlying binned distances from fjord boundary to front
        counts, bin_edges = np.histogram(data, bins=num_bins)
        inlier_mask = np.logical_not(is_outlier(counts, z_score_cutoff=2.5))
        counts_inlier_max = counts[inlier_mask].max()
        
        #Determine threshold to cutoff - any inlying border is likely part of the fjord boundary edge
        bin_index = 3
        for i in range(bin_index - 1, bin_index + 4):
            if counts[i] <= counts_inlier_max:
                if bin_edges[i-1] != 0:
                    bin_index = i - 1
                    break
        threshold = bin_edges[bin_index]
        print('\t\t\t' + 'fjord distance cutoff threshold (px):', np.around(threshold, 3))
        
        thresholded_distances = polyline_distances_image * (polyline_distances_image > threshold)
        
#        plt.figure(12000 + random.randint(1,500))
#        plt.imshow(thresholded_distances)
#        plt.show()
        
        #readd elements that are close to threshold
        inverse_distances = 255 - thresholded_distances
#        
#        plt.figure(12500 + random.randint(1,500))
#        plt.imshow(inverse_distances)
#        plt.show()
        
        #set all < -threshold =0
        inverse_distances[inverse_distances < (255 - threshold)] = 0
        
#        plt.figure(13000 + random.randint(1,500))
#        plt.imshow(inverse_distances)
#        plt.show()
        
        #remove small components
        large_inverse_distances = 255-remove_small_components(255-inverse_distances)[0]
        
#        plt.figure(13500 + random.randint(1,500))
#        plt.imshow(large_inverse_distances)
#        plt.show()
        
        #get distance transform of inverse
        distances_to_front = distance_transform_edt(large_inverse_distances)
        
        #add any elements < threshold back to mask
        isolated_fronts = polyline_distances_image * (np.logical_and(distances_to_front < threshold * 1.0, polyline_image > 0))
        
#        plt.figure(11500 + random.randint(1,500))
#        plt.imshow(isolated_fronts)
#        plt.show()
        
        #Perform final dummy remove_small_components to retrieve bounding boxes
        polyline_image = np.where(isolated_fronts > 0, 1.0, 0.0)
        test = np.stack((polyline_image, settings['empty_image'], 1.0-fjord_boundary_final_f32), axis = 2)
#        plt.figure(11000 + random.randint(1,500))
#        plt.imshow(test)
#        plt.show()
#        exit()
        
        polyline_image, bounding_boxes = remove_small_components(polyline_image, min_size_percentage=min_size_percentage)
        
        #Detect fronts that do not connect to fjord boundries (but do not eliminate them, in case of Ice Shelves)
        new_polylines_coords = []
        polylines_distances = []
        bounding_boxes_final = [bounding_boxes[0]]
        for bounding_box in bounding_boxes[1:]:
            polyline_coords = retrace_polyline(settings, polylines_coords, polyline_image, bounding_box)
            if polyline_coords is None or len(polyline_coords[0]) < 10:
                print('\t\t' + 'removing unshapely front bounding_box:', bounding_box)
                continue
            num_points = len(polyline_coords[0])
            polyline_distances = fjord_distances[(polyline_coords[0], polyline_coords[1])]
            middle_polyline_distances = polyline_distances[int(num_points * 0.15):int(num_points * 0.85)]
            start_min_distances = np.min(polyline_distances[:int(num_points * 0.15)])
            middle_max_distances = np.max(middle_polyline_distances)
            end_min_distances = np.min(polyline_distances[int(num_points * 0.85):])
            
            middle_start_diff = middle_max_distances - start_min_distances
            middle_end_diff = middle_max_distances - end_min_distances
            
#            plt.figure(12000 + random.randint(1,500))
#            plt.plot(polyline_distances)
#            plt.show()
#            errorr()
            
            #Use peak analysis to lessen
            peak_width = int(max(10.0, (num_points / 3)))
            peaks, properties = find_peaks(polyline_distances, distance=peak_width)
            triangular_distance_shape = np.sign(middle_start_diff) == np.sign(middle_end_diff) #check if the front is "shaped" like a normal one, heading away from fjord in the middle, and towards the fjord boundary at the ends.
            if len(peaks) != 0 and triangular_distance_shape == True and middle_max_distances > num_points / (4 * len(peaks)):
                new_polylines_coords.append(polyline_coords)
                polylines_distances.append(polyline_distances)
                bounding_boxes_final.append(bounding_box)
            else:
#                print(middle_start_diff, middle_end_diff, end_start_diff, start_min_distances, middle_max_distances,  end_min_distances)
                print('\t\t' + 'removing unshapely front bounding_box:', bounding_box)
        polylines_coords = new_polylines_coords
        if len(polylines_coords) == 0:
            polylines_coords = None
            polylines_distances = None
        bounding_boxes = bounding_boxes_final
    else:
        #If no front is detected, return empty image.
        polyline_image = settings['empty_image']
        bounding_boxes = [[0, 0, settings['full_size'], settings['full_size']]]
        polylines_coords = None
        polylines_distances = None
    
    return polyline_image, bounding_boxes, polylines_coords, polylines_distances


#What's up with the polyline? sometimes works, sometimes doesnt? only on validaion.
    #also the overlays are messed up.
def retrace_polyline(settings, polylines_coords, polyline_image, bounding_box):
    """Recover the ordered polyline from the isolated polyline mask.
        Clips the vector poygon with the pixel mask.
        This is done to prevent re-extraction errors/inefficiencies."""
#    print('\t\t\t' + 'retrace_polyline')
    polyline_coords = []
    image = polyline_image
    
    padding = int(settings['full_size'] / 16)
    padding = 1
        
    sub_x1 = max(bounding_box[0] - padding, 0)
    sub_x2 = min(bounding_box[0] + bounding_box[2] + padding, image.shape[0])
    sub_y1 = max(bounding_box[1] - padding, 0)
    sub_y2 = min(bounding_box[1] + bounding_box[3] + padding, image.shape[1])
    
    mask = np.zeros((image.shape[0], image.shape[1]))
    mask[sub_x1:sub_x2, sub_y1:sub_y2] = 1.0
               
    masked_image = None
    if len(image.shape) > 2:
        masked_image = image[:, :, 0] * mask
    else:
        masked_image = image * mask

    masked_image, new_bounding_boxes = remove_small_components(masked_image, limit=1)
    masked_image_dilated = cv2.dilate(masked_image.astype(np.float64), settings['kernel'], iterations=1).astype(np.float32) #np.float32 [0.0, 255.0]
    
    for polyline_coords in polylines_coords:
        for i in range(len(polyline_coords[0])):
            if masked_image_dilated[polyline_coords[0][i], polyline_coords[1][i]] > 0.5:
                for j in range(i, len(polyline_coords[0])):
                    if masked_image_dilated[polyline_coords[0][j], polyline_coords[1][j]] < 0.5:
                        break
                return np.array([polyline_coords[0][i:j], polyline_coords[1][i:j]])


def mask_bounding_box(bounding_boxes, image, settings, polylines_coords, mask_pred, store_box=True):
    """Selects a single bounding box/polyline from the detected fronts, based on various metrics.
        Capable of prioritizing confident, long, centralized fronts that are far away from fronts that have already been detected, and close to original detections in the first processing pass."""
#    print('\t\t' + 'mask_bounding_box')
    full_size = settings['full_size']
    image_settings = settings['image_settings']
    bounding_box = bounding_boxes[1]
    polyline_coords = polylines_coords[0]
    #If more than 1 bounding box, find the one closest to the target, in case multiple fronts are close
    if len(bounding_boxes) > 2:
        #Use fjord distances to select box ordering
        orderings = []
        weights = []
        #Use confidence to select box
#        ordering = calculate_box_confidence(bounding_boxes, image, settings, polylines_coords, mask_pred)
#        orderings.append(ordering)
#        weights.append(1.0)
            
        #Use length to select box
#        ordering = calculate_box_length(bounding_boxes, polylines_coords)
#        orderings.append(ordering)
#        weights.append(1.0)
        
        #Use distance to select box
        ordering = calculate_box_distance(bounding_boxes, settings)
        orderings.append(ordering)
        weights.append(2.5)
        
        #Determine which bounding box scores best according to up to 3 metrics
        placings = np.zeros((len(bounding_boxes) - 1))
        for i in range(len(orderings)):
            ordering = orderings[i]
            weight = weights[i]
            place = 0
            for index in ordering:
                placings[index] += place * weight
                place += 1
        ordering = np.argsort(placings)
        index = ordering[0]
        bounding_box = bounding_boxes[index + 1]
        polyline_coords = polylines_coords[index]
    
    polyline_center = np.array([np.mean(polyline_coords[0]), np.mean(polyline_coords[1])])
    actual_bounding_box = image_settings['actual_bounding_box']
    top_left = np.array([actual_bounding_box[0], actual_bounding_box[1]])
    scale = np.array([actual_bounding_box[2] / full_size, actual_bounding_box[3] / full_size])
    original_polyline_center = polyline_center * scale + top_left
    
    #Exit early if front is not within bounds of original bounding box
    original_bounding_box = image_settings['bounding_box']
    if not (original_polyline_center[0] > original_bounding_box[0] and 
        original_polyline_center[0] < original_bounding_box[0] + original_bounding_box[2] and
        original_polyline_center[1] > original_bounding_box[1] and 
        original_polyline_center[1] < original_bounding_box[1] + original_bounding_box[3]):
        return None
    #Store used bounding box to prevent overlap in reprocessed fronts
    if store_box:
        image_settings['used_bounding_boxes'].append(original_polyline_center)
    
    #Redraw the masked front
    polyline_image = np.zeros((image.shape[0], image.shape[1], 3))
    pts = np.vstack((polyline_coords[1], polyline_coords[0])).astype(np.int32).T
    cv2.polylines(polyline_image, [pts], False, (255, 0, 0), 1)
#    plt.figure(14000 + random.randint(1,500))
#    plt.imshow(polyline_image)
#    plt.show()
    return polyline_image, bounding_box, polyline_coords


def calculate_box_confidence(bounding_boxes, image, settings, polylines_coords, mask_pred):
    """Helper function for mask_bounding_box to determine masking confidence of each bounding box's polyline."""
    metric = []
    confidence_kernel = settings['confidence_kernel']
    for i in range(1, len(bounding_boxes)):
        polyline_coords = polylines_coords[i-1]
        
        edge_pred = np.zeros((image.shape[0], image.shape[1], 3))
        pts = np.vstack((polyline_coords[1], polyline_coords[0])).astype(np.int32).T
        cv2.polylines(edge_pred, [pts], False, (255, 0, 0), 3)
        edge_pred = edge_pred[:, :, 0].astype(np.float64)
        edge_binary_dilated = cv2.dilate(edge_pred, confidence_kernel, iterations=1) #np.float32 [0.0, 255.0]
        edge_binary_dilated_indices = edge_binary_dilated.astype(np.float32) > 127.0
        mask_edge_buffered = np.zeros(mask_pred.shape) + 0.5
        mask_edge_buffered[edge_binary_dilated_indices] = mask_pred[edge_binary_dilated_indices]
        mask_pred_normalized = mask_edge_buffered - 0.5
        nonzeros_mask_edge_buffered = mask_pred_normalized[edge_binary_dilated_indices]
        mask_edge_buffered_confidence_strength = np.abs(np.mean(nonzeros_mask_edge_buffered))
        metric.append(mask_edge_buffered_confidence_strength)
#        plt.figure(12500 + random.randint(1,500))
#        plt.imshow(mask_pred_normalized)
#        plt.show()
    print('\t\t\t' + 'metric confidence:', np.around(metric, 3))
    ordering = np.argsort(metric)
    return ordering


def calculate_box_length(bounding_boxes, polylines_coords):
    """Helper function for mask_bounding_box to determine length of each bounding box's polyline."""
    metric = []
    for i in range(1, len(bounding_boxes)):
        polyline_coords_length = len(polylines_coords[i-1][0])
        metric.append(polyline_coords_length)
    print('\t\t\t' + 'metric length (px):', np.around(metric, 3))
    ordering = list(reversed(np.argsort(metric)))
    return ordering


def calculate_box_distance(bounding_boxes, settings):
    """Helper function for mask_bounding_box to determine distance to center of image of each bounding box's polyline."""
    metric = []
    full_size = settings['full_size']
    image_settings = settings['image_settings']
    target_center = image_settings['target_center']
    subset_bounding_box = image_settings['actual_bounding_box']
    top_left = np.array([subset_bounding_box[0], subset_bounding_box[1]])
    scale = np.array([subset_bounding_box[2] / full_size, subset_bounding_box[3] / full_size])
    
    #Calculate location of original bounding box in original subset coordinates
    for i in range(1, len(bounding_boxes)):
        bounding_box = bounding_boxes[i]
        bounding_box_center = np.array([bounding_box[0] + bounding_box[2] / 2, bounding_box[1] + bounding_box[3] / 2])
        original_bounding_box_center = bounding_box_center * scale + top_left
        distance = np.linalg.norm(target_center - original_bounding_box_center)
        metric.append(distance)
    print('\t\t\t' + 'metric distance (px):', np.around(metric, 3))
    ordering = np.argsort(metric)
    return ordering


def calculate_metrics_calfin(settings, metrics):
    """Helper function to calculate evaluation metrics for validated calving fronts."""
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
    mean_deviation_subset, distances = calculate_mean_deviation(polyline_image[:, :, 0], mask_final_f32[:, :, 0])
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
    print('\t\t' + "mean_deviation_subset: {:.2f} pixels, {:.2f} meters".format(mean_deviation_subset, mean_deviation_subset * meters_per_subset_pixel))
    print('\t\t' + "mean_deviation_difference: {:.2f} pixels, {:.2f} meters (- == good)".format(mean_deviation_subset - mean_deviation, mean_deviation_subset * meters_per_subset_pixel - mean_deviation * meters_per_256_pixel))
    
    #Dilate masks for easier visualization and IoU metric calculation.
    polyline_image_dilated = cv2.dilate(polyline_image.astype(np.float64), kernel, iterations=1).astype(np.float32) #np.float32 [0.0, 255.0]
    mask_edge_dilated_f32 = cv2.dilate(mask_final_f32[:, :, 0].astype(np.float64), kernel, iterations= 1).astype(np.float32) #np.float32 [0.0, 255.0]
    mask_final_dilated_f32 = np.stack((mask_edge_dilated_f32, mask_final_f32[:, :, 1]), axis=-1)
    image_settings['polyline_image_dilated'] = polyline_image_dilated
    image_settings['mask_final_dilated_f32'] = mask_final_dilated_f32
    
    #Calculate and save IoU metric
    pred_edge_patch_4d = np.expand_dims(polyline_image_dilated[:, :, 0], axis=0)
    mask_edge_patch_4d = np.expand_dims(mask_final_dilated_f32[:, :, 0], axis=0)
    pred_mask_patch_4d = np.expand_dims(pred_image[:, :, 1], axis=0)
    mask_mask_patch_4d = np.expand_dims(mask_final_dilated_f32[:, :, 1], axis=0)
    edge_iou_score_subset = calculate_iou(mask_edge_patch_4d, pred_edge_patch_4d)
    mask_iou_score_subset = calculate_iou(mask_mask_patch_4d, pred_mask_patch_4d)
    image_settings['edge_iou_score_subset'] = edge_iou_score_subset
    image_settings['mask_iou_score_subset'] = mask_iou_score_subset
    metrics['validation_edge_ious'] = np.append(metrics['validation_edge_ious'], edge_iou_score_subset)
    metrics['validation_mask_ious'] = np.append(metrics['validation_mask_ious'], mask_iou_score_subset)
    metrics['domain_validation_edge_ious'][domain] = np.append(metrics['domain_validation_edge_ious'][domain], edge_iou_score_subset)
    metrics['domain_validation_mask_ious'][domain] = np.append(metrics['domain_validation_mask_ious'][domain], mask_iou_score_subset)
    metrics['resolution_iou_array'] = np.concatenate((metrics['resolution_iou_array'], np.array([[meters_per_subset_pixel, edge_iou_score_subset]])))
    print('\t\t' + "edge_iou_score_subset: {:.2f}".format(edge_iou_score_subset))
    print('\t\t' + "edge_iou_score change {:.2f} (+ == good):".format(edge_iou_score_subset - edge_iou_score))
    
    #Add to calandars
    metrics['domain_validation_calendar'][domain][year] += 1
