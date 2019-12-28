import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

import sys
sys.path.insert(1, '../training/keras-deeplab-v3-plus')
sys.path.insert(2, '../training')

from plotting import plot_histogram, plot_scatter


def print_calfin_domain_metrics(settings, metrics):
	dest_path_qa = settings['dest_path_qa']
	saving = settings['saving']
	for domain in metrics['domain_mean_deviations_pixels'].keys():
		domain_scaling = settings['domain_scalings'][domain]
		domain_mean_deviation_points_pixels = np.nanmean(metrics['domain_validation_distances_pixels'][domain])
		domain_mean_deviation_points_meters = np.nanmean(metrics['domain_validation_distances_meters'][domain])
		domain_mean_deviation_images_pixels = np.nanmean(metrics['domain_mean_deviations_pixels'][domain])
		domain_mean_deviation_images_meters = np.nanmean(metrics['domain_mean_deviations_meters'][domain])
		domain_median_mean_deviation_points_pixels = np.nanmedian(metrics['domain_validation_distances_pixels'][domain])
		domain_median_mean_deviation_points_meters = np.nanmedian(metrics['domain_validation_distances_meters'][domain])
		domain_median_mean_deviation_images_pixels = np.nanmedian(metrics['domain_mean_deviations_pixels'][domain])
		domain_median_mean_deviation_images_meters = np.nanmedian(metrics['domain_mean_deviations_meters'][domain])
		domain_mean_edge_iou_score = np.nanmean(metrics['domain_validation_ious'][domain])
		domain_median_edge_iou_score = np.nanmedian(metrics['domain_validation_ious'][domain])
		print("{} mean deviation (averaged over points): {:.2f} meters".format(domain, domain_mean_deviation_points_meters))
		print("{} mean deviation (averaged over images): {:.2f} meters".format(domain, domain_mean_deviation_images_meters))
		print("{} mean deviation (averaged over points): {:.2f} pixels".format(domain, domain_mean_deviation_points_pixels))
		print("{} mean deviation (averaged over images): {:.2f} pixels".format(domain, domain_mean_deviation_images_pixels))
		print("{} mean Jaccard index (Intersection over Union): {:.4f}".format(domain, domain_mean_edge_iou_score))
		print("{} mean deviation (median over points): {:.2f} meters".format(domain, domain_median_mean_deviation_points_meters))
		print("{} mean deviation (median over images): {:.2f} meters".format(domain, domain_median_mean_deviation_images_meters))
		print("{} mean deviation (median over points): {:.2f} pixels".format(domain, domain_median_mean_deviation_points_pixels))
		print("{} mean deviation (median over images): {:.2f} pixels".format(domain, domain_median_mean_deviation_images_pixels))
		print("{} median Jaccard index (Intersection over Union): {:.4f}".format(domain, domain_median_edge_iou_score))
		plot_histogram(metrics['domain_validation_distances_meters'][domain], "mean_deviations_meters_" + domain, dest_path_qa, saving, domain_scaling)


def print_calfin_all_metrics(settings, metrics):
	dest_path_qa = settings['dest_path_qa']
	scaling = settings['scaling']
	saving = settings['saving']
	validation_files = settings['validation_files']
	mean_deviation_points_pixels = np.nanmean(metrics['validation_distances_pixels'])
	mean_deviation_points_meters = np.nanmean(metrics['validation_distances_meters'])
	mean_deviation_images_pixels = np.nanmean(metrics['mean_deviations_pixels'])
	mean_deviation_images_meters = np.nanmean(metrics['mean_deviations_meters'])
	mean_edge_iou_score = np.nanmean(metrics['validation_ious'])
	median_mean_deviation_points_pixels = np.nanmedian(metrics['validation_distances_pixels'])
	median_mean_deviation_points_meters = np.nanmedian(metrics['validation_distances_meters'])
	median_mean_deviation_images_pixels = np.nanmedian(metrics['mean_deviations_pixels'])
	median_mean_deviation_images_meters = np.nanmedian(metrics['mean_deviations_meters'])
	median_edge_iou_score = np.nanmedian(metrics['validation_ious'])
	print("mean deviation (averaged over points): {:.2f} meters".format(mean_deviation_points_meters))
	print("mean deviation (averaged over images): {:.2f} meters".format(mean_deviation_images_meters))
	print("mean deviation (averaged over points): {:.2f} pixels".format(mean_deviation_points_pixels))
	print("mean deviation (averaged over images): {:.2f} pixels".format(mean_deviation_images_pixels))
	print("mean deviation (median over points): {:.2f} meters".format(median_mean_deviation_points_meters))
	print("mean deviation (median over images): {:.2f} meters".format(median_mean_deviation_images_meters))
	print("mean deviation (median over points): {:.2f} pixels".format(median_mean_deviation_points_pixels))
	print("mean deviation (median over images): {:.2f} pixels".format(median_mean_deviation_images_pixels))
	print("mean Jaccard index (Intersection over Union): {:.4f}".format(mean_edge_iou_score))
	print("median Jaccard index (Intersection over Union): {:.4f}".format(median_edge_iou_score))
	
	#Print histogram of all distance errors
	plot_histogram(metrics['validation_distances_meters'], "all_mean_deviations_meters", dest_path_qa, saving, scaling)
	
	#Print scatterplot of resolution errors
	plot_scatter(metrics['resolution_deviation_array'], "Validation Resolution vs Deviations", dest_path_qa, saving)
	plot_scatter(metrics['resolution_iou_array'], "Validation Resolution vs IoU", dest_path_qa, saving)
	plt.show()
	
	#Print final statistics
	print('mask_confidence_strength_threshold', settings['mask_confidence_strength_threshold'] , 
		'edge_confidence_strength_threshold', settings['edge_confidence_strength_threshold'] )
	print('image_skip_count:', metrics['image_skip_count'], 
		'front skip count:', metrics['no_detection_skip_count'] + metrics['confidence_skip_count'], 
		'no_detection_skip_count:', metrics['no_detection_skip_count'], 
		"confidence_skip_count:", metrics['confidence_skip_count'])
	print('total fronts:', metrics['front_count'])
	
	number_no_front_images = len(settings['negative_image_names']) #13 images are clouded/undetectable in the CALFIN validation set of 152 images
	number_valid_images = len(validation_files) - number_no_front_images
	percent_images_with_fronts = (1 - metrics['image_skip_count'] / len(validation_files))* 100
	number_images_with_fronts = str(number_valid_images - metrics['image_skip_count'])
	total_images = str(number_valid_images)
	
	print('True Positives: {}, False Positives: {}, False Negatives: {}, True Negatives: {}'.format(
			metrics['true_positives'], metrics['false_positive'], metrics['false_negatives'], metrics['true_negatives']))
	print('% images with fronts: {:.2f} ({}/{})'.format(percent_images_with_fronts, number_images_with_fronts, total_images))


