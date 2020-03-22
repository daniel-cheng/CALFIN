import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

import sys, os
sys.path.insert(1, '../training/keras-deeplab-v3-plus')
sys.path.insert(2, '../training')

from plotting import plot_histogram, plot_scatter


def print_calfin_domain_metrics(settings, metrics):
	"""Prints out metrics for CALFIN inputs and graphs mean deviations per domain."""
	dest_path_qa = settings['dest_path_qa']
	saving = settings['saving']
	for domain in metrics['domain_mean_deviations_pixels'].keys():
		sqrt_point_samples = len(metrics['domain_validation_distances_meters'][domain])
		sqrt_image_samples = len(metrics['domain_mean_deviations_meters'][domain])
		domain_scaling = settings['domain_scalings'][domain]
		domain_mean_deviation_points_pixels = np.nanmean(metrics['domain_validation_distances_pixels'][domain])
		domain_mean_deviation_points_meters = np.nanmean(metrics['domain_validation_distances_meters'][domain])
		domain_mean_deviation_images_pixels = np.nanmean(metrics['domain_mean_deviations_pixels'][domain])
		domain_mean_deviation_images_meters = np.nanmean(metrics['domain_mean_deviations_meters'][domain])
		domain_std_deviation_points_pixels = np.nanstd(metrics['domain_validation_distances_pixels'][domain]) / sqrt_point_samples * 1.96
		domain_std_deviation_points_meters = np.nanstd(metrics['domain_validation_distances_meters'][domain]) / sqrt_point_samples * 1.96
		domain_std_deviation_images_pixels = np.nanstd(metrics['domain_mean_deviations_pixels'][domain]) / sqrt_image_samples * 1.96
		domain_std_deviation_images_meters = np.nanstd(metrics['domain_mean_deviations_meters'][domain]) / sqrt_image_samples * 1.96
		domain_median_mean_deviation_points_pixels = np.nanmedian(metrics['domain_validation_distances_pixels'][domain])
		domain_median_mean_deviation_points_meters = np.nanmedian(metrics['domain_validation_distances_meters'][domain])
		domain_median_mean_deviation_images_pixels = np.nanmedian(metrics['domain_mean_deviations_pixels'][domain])
		domain_median_mean_deviation_images_meters = np.nanmedian(metrics['domain_mean_deviations_meters'][domain])
		domain_mean_edge_iou_score = np.nanmean(metrics['domain_validation_edge_ious'][domain])
		domain_mean_mask_iou_score = np.nanmean(metrics['domain_validation_mask_ious'][domain])
		domain_std_edge_iou_score = np.nanstd(metrics['domain_validation_edge_ious'][domain]) / sqrt_image_samples * 1.96
		domain_std_mask_iou_score = np.nanstd(metrics['domain_validation_mask_ious'][domain]) / sqrt_image_samples * 1.96
		domain_median_edge_iou_score = np.nanmedian(metrics['domain_validation_edge_ious'][domain])
		domain_median_mask_iou_score = np.nanmedian(metrics['domain_validation_mask_ious'][domain])
		
		#Print output to file and to shell
		default = sys.stdout
		log_file_name = os.path.join(settings['dest_root_path'], settings['log_file_name'])
		output_streams = [open(log_file_name, "w+", encoding='utf-8'), default]
		for i in range(len(output_streams)):
			sys.stdout = output_streams[i]
			print("{} mean distance (averaged over points): {:.2f} ± {:.2f} meters".format(domain, domain_mean_deviation_points_meters, domain_std_deviation_points_meters))
			print("{} mean distance (averaged over images): {:.2f} ± {:.2f} meters".format(domain, domain_mean_deviation_images_meters, domain_std_deviation_images_meters))
			print("{} mean distance (averaged over points): {:.2f} ± {:.2f} pixels".format(domain, domain_mean_deviation_points_pixels, domain_std_deviation_points_pixels))
			print("{} mean distance (averaged over images): {:.2f} ± {:.2f} pixels".format(domain, domain_mean_deviation_images_pixels, domain_std_deviation_images_pixels))
			print("{} mean front Jaccard index (Intersection over Union): {:.4f} ± {:.4f}".format(domain, domain_mean_edge_iou_score, domain_std_edge_iou_score))
			print("{} mean ice/ocean Jaccard index (Intersection over Union): {:.4f} ± {:.4f}".format(domain, domain_mean_mask_iou_score, domain_std_mask_iou_score))
			print("{} mean distance (median over points): {:.2f} meters".format(domain, domain_median_mean_deviation_points_meters))
			print("{} mean distance (median over images): {:.2f} meters".format(domain, domain_median_mean_deviation_images_meters))
			print("{} mean distance (median over points): {:.2f} pixels".format(domain, domain_median_mean_deviation_points_pixels))
			print("{} mean distance (median over images): {:.2f} pixels".format(domain, domain_median_mean_deviation_images_pixels))
			print("{} median front Jaccard index (Intersection over Union): {:.4f}".format(domain, domain_median_edge_iou_score))
			print("{} median ice/ocean Jaccard index (Intersection over Union): {:.4f}".format(domain, domain_median_mask_iou_score))
			if i == 0:
				output_streams[i].close()
		plot_histogram(metrics['domain_validation_distances_meters'][domain], "mean_deviations_meters_" + domain, dest_path_qa, saving, domain_scaling)


def print_calfin_all_metrics(settings, metrics):
	"""Prints out metrics for CALFIN inputs and graphs the results."""
	dest_path_qa = settings['dest_path_qa']
	scaling = settings['scaling']
	saving = settings['saving']
	plotting = settings['plotting']
	validation_files = settings['validation_files']
	sqrt_point_samples = len(metrics['validation_distances_meters'])
	sqrt_image_samples = len(metrics['mean_deviations_meters'])
	mean_deviation_points_pixels = np.nanmean(metrics['validation_distances_pixels'])
	mean_deviation_points_meters = np.nanmean(metrics['validation_distances_meters'])
	mean_deviation_images_pixels = np.nanmean(metrics['mean_deviations_pixels'])
	mean_deviation_images_meters = np.nanmean(metrics['mean_deviations_meters'])
	mean_edge_iou_score = np.nanmean(metrics['validation_edge_ious'])
	mean_mask_iou_score = np.nanmean(metrics['validation_mask_ious'])
	std_deviation_points_pixels = np.nanstd(metrics['validation_distances_pixels']) / sqrt_point_samples * 1.96
	std_deviation_points_meters = np.nanstd(metrics['validation_distances_meters']) / sqrt_point_samples * 1.96
	std_deviation_images_pixels = np.nanstd(metrics['mean_deviations_pixels']) / sqrt_image_samples * 1.96
	std_deviation_images_meters = np.nanstd(metrics['mean_deviations_meters']) / sqrt_image_samples * 1.96
	std_edge_iou_score = np.nanstd(metrics['validation_edge_ious']) / sqrt_image_samples * 1.96
	std_mask_iou_score = np.nanstd(metrics['validation_mask_ious']) / sqrt_image_samples * 1.96
	median_mean_deviation_points_pixels = np.nanmedian(metrics['validation_distances_pixels'])
	median_mean_deviation_points_meters = np.nanmedian(metrics['validation_distances_meters'])
	median_mean_deviation_images_pixels = np.nanmedian(metrics['mean_deviations_pixels'])
	median_mean_deviation_images_meters = np.nanmedian(metrics['mean_deviations_meters'])
	median_edge_iou_score = np.nanmedian(metrics['validation_edge_ious'])
	median_mask_iou_score = np.nanmedian(metrics['validation_mask_ious'])
	
	if plotting:
		#Print histogram of all distance errors
		plot_histogram(metrics['validation_distances_meters'], "all_mean_deviations_meters", dest_path_qa, saving, scaling)
		
		#Print scatterplot of resolution errors
		plot_scatter(metrics['resolution_deviation_array'], "Validation Resolution vs Deviations", dest_path_qa, saving)
		plot_scatter(metrics['resolution_iou_array'], "Validation Resolution vs IoU", dest_path_qa, saving)
		plt.show()
	
	number_no_front_images = len(settings['negative_image_names']) #13 images are clouded/undetectable in the CALFIN validation set of 152 images
	number_valid_images = len(validation_files) - number_no_front_images
	percent_images_with_fronts = (1 - metrics['image_skip_count'] / len(validation_files))* 100
	number_images_with_fronts = str(number_valid_images - metrics['image_skip_count'])
	total_images = str(number_valid_images)
	
	#Print output to file and to shell
	default = sys.stdout
	log_file_name = os.path.join(settings['dest_root_path'], settings['log_file_name'])
	output_streams = [open(log_file_name, "a", encoding='utf-8'), default]
	for i in range(len(output_streams)):
		sys.stdout = output_streams[i]
		print("mean distance (averaged over points): {:.2f} ± {:.2f} meters".format(mean_deviation_points_meters, std_deviation_points_meters))
		print("mean distance (averaged over images): {:.2f} ± {:.2f} meters".format(mean_deviation_images_meters, std_deviation_images_meters))
		print("mean distance (averaged over points): {:.2f} ± {:.2f} pixels".format(mean_deviation_points_pixels, std_deviation_points_pixels))
		print("mean distance (averaged over images): {:.2f} ± {:.2f} pixels".format(mean_deviation_images_pixels, std_deviation_images_pixels))
		print("mean distance (median over points): {:.2f} meters".format(median_mean_deviation_points_meters))
		print("mean distance (median over images): {:.2f} meters".format(median_mean_deviation_images_meters))
		print("mean distance (median over points): {:.2f} pixels".format(median_mean_deviation_points_pixels))
		print("mean distance (median over images): {:.2f} pixels".format(median_mean_deviation_images_pixels))
		print("mean front Jaccard index (Intersection over Union): {:.4f} ± {:.4f}".format(mean_edge_iou_score, std_edge_iou_score))
		print("mean ice/ocean Jaccard index (Intersection over Union): {:.4f} ± {:.4f}".format(mean_mask_iou_score, std_mask_iou_score))
		print("median front Jaccard index (Intersection over Union): {:.4f}".format(median_edge_iou_score))
		print("median ice/ocean Jaccard index (Intersection over Union): {:.4f}".format(median_mask_iou_score))
			
		#Print final statistics
		print('mask_confidence_strength_threshold', settings['mask_confidence_strength_threshold'] , 
			'edge_confidence_strength_threshold', settings['edge_confidence_strength_threshold'] )
		print('image_skip_count:', metrics['image_skip_count'], 
			'front skip count:', metrics['no_detection_skip_count'] + metrics['confidence_skip_count'], 
			'no_detection_skip_count:', metrics['no_detection_skip_count'], 
			"confidence_skip_count:", metrics['confidence_skip_count'])
		print('total fronts:', metrics['front_count'])
		
		print('True Positives: {}, False Positives: {}, False Negatives: {}, True Negatives: {}'.format(
				metrics['true_positives'], metrics['false_positive'], metrics['false_negatives'], metrics['true_negatives']))
		print('% images with fronts: {:.2f} ({}/{})'.format(percent_images_with_fronts, number_images_with_fronts, total_images))
		if i == 0:
			output_streams[i].close()


def print_mohajerani_domain_metrics(settings, metrics):
	"""Prints out metrics for CALFIN inputs and graphs mean deviations per domain."""
	dest_path_qa = settings['dest_path_qa']
	saving = settings['saving']
	for domain in metrics['domain_mean_deviations_pixels'].keys():
		sqrt_point_samples = len(metrics['domain_validation_distances_meters'][domain])
		sqrt_image_samples = len(metrics['domain_mean_deviations_meters'][domain])
		domain_scaling = settings['domain_scalings'][domain]
		domain_mean_deviation_points_pixels = np.nanmean(metrics['domain_validation_distances_pixels'][domain])
		domain_mean_deviation_points_meters = np.nanmean(metrics['domain_validation_distances_meters'][domain])
		domain_mean_deviation_images_pixels = np.nanmean(metrics['domain_mean_deviations_pixels'][domain])
		domain_mean_deviation_images_meters = np.nanmean(metrics['domain_mean_deviations_meters'][domain])
		domain_std_deviation_points_pixels = np.nanstd(metrics['domain_validation_distances_pixels'][domain]) / sqrt_point_samples * 1.96
		domain_std_deviation_points_meters = np.nanstd(metrics['domain_validation_distances_meters'][domain]) / sqrt_point_samples * 1.96
		domain_std_deviation_images_pixels = np.nanstd(metrics['domain_mean_deviations_pixels'][domain]) / sqrt_image_samples * 1.96
		domain_std_deviation_images_meters = np.nanstd(metrics['domain_mean_deviations_meters'][domain]) / sqrt_image_samples * 1.96
		domain_median_mean_deviation_points_pixels = np.nanmedian(metrics['domain_validation_distances_pixels'][domain])
		domain_median_mean_deviation_points_meters = np.nanmedian(metrics['domain_validation_distances_meters'][domain])
		domain_median_mean_deviation_images_pixels = np.nanmedian(metrics['domain_mean_deviations_pixels'][domain])
		domain_median_mean_deviation_images_meters = np.nanmedian(metrics['domain_mean_deviations_meters'][domain])
		domain_mean_edge_iou_score = np.nanmean(metrics['domain_validation_edge_ious'][domain])
		domain_std_edge_iou_score = np.nanstd(metrics['domain_validation_edge_ious'][domain]) / sqrt_image_samples * 1.96
		domain_median_edge_iou_score = np.nanmedian(metrics['domain_validation_edge_ious'][domain])
		
		#Print output to file and to shell
		default = sys.stdout
		log_file_name = os.path.join(settings['dest_root_path'], settings['log_file_name'])
		output_streams = [open(log_file_name, "w+", encoding='utf-8'), default]
		for i in range(len(output_streams)):
			sys.stdout = output_streams[i]
			print("{} mean distance (averaged over points): {:.2f} ± {:.2f} meters".format(domain, domain_mean_deviation_points_meters, domain_std_deviation_points_meters))
			print("{} mean distance (averaged over images): {:.2f} ± {:.2f} meters".format(domain, domain_mean_deviation_images_meters, domain_std_deviation_images_meters))
			print("{} mean distance (averaged over points): {:.2f} ± {:.2f} pixels".format(domain, domain_mean_deviation_points_pixels, domain_std_deviation_points_pixels))
			print("{} mean distance (averaged over images): {:.2f} ± {:.2f} pixels".format(domain, domain_mean_deviation_images_pixels, domain_std_deviation_images_pixels))
			print("{} mean front Jaccard index (Intersection over Union): {:.4f} ± {:.4f}".format(domain, domain_mean_edge_iou_score, domain_std_edge_iou_score))
			print("{} mean ice/ocean Jaccard index (Intersection over Union): {:.4f} ± {:.4f}".format(domain, np.nan, np.nan))
			print("{} mean distance (median over points): {:.2f} meters".format(domain, domain_median_mean_deviation_points_meters))
			print("{} mean distance (median over images): {:.2f} meters".format(domain, domain_median_mean_deviation_images_meters))
			print("{} mean distance (median over points): {:.2f} pixels".format(domain, domain_median_mean_deviation_points_pixels))
			print("{} mean distance (median over images): {:.2f} pixels".format(domain, domain_median_mean_deviation_images_pixels))
			print("{} median front Jaccard index (Intersection over Union): {:.4f}".format(domain, domain_median_edge_iou_score))
			print("{} median ice/ocean Jaccard index (Intersection over Union): {:.4f}".format(domain, np.nan))
			if i == 0:
				output_streams[i].close()
		plot_histogram(metrics['domain_validation_distances_meters'][domain], "mean_deviations_meters_" + domain, dest_path_qa, saving, domain_scaling)


def print_mohajerani_all_metrics(settings, metrics):
	"""Prints out metrics for CALFIN inputs and graphs the results."""
	dest_path_qa = settings['dest_path_qa']
	scaling = settings['scaling']
	saving = settings['saving']
	plotting = settings['plotting']
	validation_files = settings['validation_files']
	sqrt_point_samples = len(metrics['validation_distances_meters'])
	sqrt_image_samples = len(metrics['mean_deviations_meters'])
	mean_deviation_points_pixels = np.nanmean(metrics['validation_distances_pixels'])
	mean_deviation_points_meters = np.nanmean(metrics['validation_distances_meters'])
	mean_deviation_images_pixels = np.nanmean(metrics['mean_deviations_pixels'])
	mean_deviation_images_meters = np.nanmean(metrics['mean_deviations_meters'])
	mean_edge_iou_score = np.nanmean(metrics['validation_edge_ious'])
	mean_mask_iou_score = np.nanmean(metrics['validation_mask_ious'])
	std_deviation_points_pixels = np.nanstd(metrics['validation_distances_pixels']) / sqrt_point_samples * 1.96
	std_deviation_points_meters = np.nanstd(metrics['validation_distances_meters']) / sqrt_point_samples * 1.96
	std_deviation_images_pixels = np.nanstd(metrics['mean_deviations_pixels']) / sqrt_image_samples * 1.96
	std_deviation_images_meters = np.nanstd(metrics['mean_deviations_meters']) / sqrt_image_samples * 1.96
	std_edge_iou_score = np.nanstd(metrics['validation_edge_ious']) / sqrt_image_samples * 1.96
	std_mask_iou_score = np.nanstd(metrics['validation_mask_ious']) / sqrt_image_samples * 1.96
	median_mean_deviation_points_pixels = np.nanmedian(metrics['validation_distances_pixels'])
	median_mean_deviation_points_meters = np.nanmedian(metrics['validation_distances_meters'])
	median_mean_deviation_images_pixels = np.nanmedian(metrics['mean_deviations_pixels'])
	median_mean_deviation_images_meters = np.nanmedian(metrics['mean_deviations_meters'])
	median_edge_iou_score = np.nanmedian(metrics['validation_edge_ious'])
	median_mask_iou_score = np.nanmedian(metrics['validation_mask_ious'])
	
	if plotting:
		#Print histogram of all distance errors
		plot_histogram(metrics['validation_distances_meters'], "all_mean_deviations_meters", dest_path_qa, saving, scaling)
		
		#Print scatterplot of resolution errors
		plot_scatter(metrics['resolution_deviation_array'], "Validation Resolution vs Deviations", dest_path_qa, saving)
		plot_scatter(metrics['resolution_iou_array'], "Validation Resolution vs IoU", dest_path_qa, saving)
		plt.show()
	
	number_no_front_images = len(settings['negative_image_names']) #13 images are clouded/undetectable in the CALFIN validation set of 152 images
	number_valid_images = len(validation_files) - number_no_front_images
	percent_images_with_fronts = (1 - metrics['image_skip_count'] / len(validation_files))* 100
	number_images_with_fronts = str(number_valid_images - metrics['image_skip_count'])
	total_images = str(number_valid_images)
	
	#Print output to file and to shell
	default = sys.stdout
	log_file_name = os.path.join(settings['dest_root_path'], settings['log_file_name'])
	output_streams = [open(log_file_name, "a", encoding='utf-8'), default]
	for i in range(len(output_streams)):
		sys.stdout = output_streams[i]
		print("mean distance (averaged over points): {:.2f} ± {:.2f} meters".format(mean_deviation_points_meters, std_deviation_points_meters))
		print("mean distance (averaged over images): {:.2f} ± {:.2f} meters".format(mean_deviation_images_meters, std_deviation_images_meters))
		print("mean distance (averaged over points): {:.2f} ± {:.2f} pixels".format(mean_deviation_points_pixels, std_deviation_points_pixels))
		print("mean distance (averaged over images): {:.2f} ± {:.2f} pixels".format(mean_deviation_images_pixels, std_deviation_images_pixels))
		print("mean distance (median over points): {:.2f} meters".format(median_mean_deviation_points_meters))
		print("mean distance (median over images): {:.2f} meters".format(median_mean_deviation_images_meters))
		print("mean distance (median over points): {:.2f} pixels".format(median_mean_deviation_points_pixels))
		print("mean distance (median over images): {:.2f} pixels".format(median_mean_deviation_images_pixels))
		print("mean front Jaccard index (Intersection over Union): {:.4f} ± {:.4f}".format(mean_edge_iou_score, std_edge_iou_score))
		print("mean ice/ocean Jaccard index (Intersection over Union): {:.4f} ± {:.4f}".format(np.nan, np.nan))
		print("median front Jaccard index (Intersection over Union): {:.4f}".format(median_edge_iou_score))
		print("median ice/ocean Jaccard index (Intersection over Union): {:.4f}".format(np.nan))
			
		#Print final statistics
		print('mask_confidence_strength_threshold', settings['mask_confidence_strength_threshold'] , 
			'edge_confidence_strength_threshold', settings['edge_confidence_strength_threshold'] )
		print('image_skip_count:', metrics['image_skip_count'], 
			'front skip count:', metrics['no_detection_skip_count'] + metrics['confidence_skip_count'], 
			'no_detection_skip_count:', metrics['no_detection_skip_count'], 
			"confidence_skip_count:", metrics['confidence_skip_count'])
		print('total fronts:', metrics['front_count'])
		
		print('True Positives: {}, False Positives: {}, False Negatives: {}, True Negatives: {}'.format(
				metrics['true_positives'], metrics['false_positive'], metrics['false_negatives'], metrics['true_negatives']))
		print('% images with fronts: {:.2f} ({}/{})'.format(percent_images_with_fronts, number_images_with_fronts, total_images))
		if i == 0:
			output_streams[i].close()


def output_calendar_csv(settings, metrics):
	"""Creates a csv file with the number of images per year per domain."""
	calendar_name = settings['driver'] + '_calendar.csv'
	calendar_path = os.path.join(settings['dest_path_qa'], calendar_name)
	pd.DataFrame.from_dict(data=metrics['domain_validation_calendar'], orient='columns').to_csv(calendar_path, header=True)


