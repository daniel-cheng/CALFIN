import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from collections import defaultdict
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

import sys, cv2, glob
sys.path.insert(1, '../training/keras-deeplab-v3-plus')
sys.path.insert(2, '../training')

from preprocessing import  preprocess
from processing import compile_model, process
from postprocessing import postprocess
from plotting import plot_histogram, plot_scatter

def main(settings, metrics):
	#Begin processing validation images
	troubled_ones = [3, 14, 22, 43, 66, 83, 97, 114, 161]
	troubled_ones = [137]
#	for i in range(0, len(settings['validation_files'])):
	for i in troubled_ones:
		preprocess(i, settings, metrics)
		process(settings, metrics)
		postprocess(settings, metrics)
	
	#Print statistics
	print_calfin_domain_metrics(settings, metrics)
	print_calfin_all_metrics(settings, metrics)
	
	return settings, metrics


def initialize(model):
	#initialize settings and model if not already done	
	plotting = True
	saving = True
	
	#Initialize plots
	plt.close('all')
	font = {'family' : 'normal',
	        'size'   : 14}
	plt.rc('font', **font)
		
	validation_files = glob.glob(r"..\training\data\validation\*B[0-9].png")
	dest_path_qa = r"..\outputs\validation\quality_assurance"
	dest_root_path = r"..\outputs\validation"
	scaling = 96.3 / 1.97
	settings = dict()
	settings['driver'] = 'calfin'
	settings['validation_files'] = validation_files
	settings['model'] = model
	settings['results'] = []
	settings['plotting'] = plotting
	settings['saving'] = saving
	settings['full_size'] = full_size
	settings['img_size'] = img_size
	settings['stride'] = stride
	settings['line_thickness'] = 3
	settings['kernel'] = cv2.getStructuringElement(cv2.MORPH_RECT, (settings['line_thickness'], settings['line_thickness']))
	settings['fjord_boundaries_path'] = r"..\training\data\fjord_boundaries"
	settings['tif_source_path'] = r"..\preprocessing\CalvingFronts\tif"
	settings['dest_path_qa'] = dest_path_qa
	settings['dest_root_path'] = dest_root_path
	settings['save_path'] = r"..\processing\landsat_preds"
	settings['total'] = len(validation_files)
	settings['empty_image'] = np.zeros((settings['full_size'], settings['full_size']))
	settings['scaling'] = scaling
	settings['domain_scalings'] = dict()
	settings['mask_confidence_strength_threshold'] = 0.875
	settings['edge_confidence_strength_threshold'] = 0.575
	settings['sub_padding_ratio'] = 2.5
	settings['edge_detection_threshold'] = 0.25 #Minimum confidence threshold for a prediction to be contribute to edge size
	settings['edge_detection_size_threshold'] = full_size / 8 #32 minimum pixel length required for an edge to trigger a detection
	settings['mask_detection_threshold'] = 0.25 #Minimum confidence threshold for a prediction to be contribute to edge size
	settings['mask_detection_ratio_threshold'] = 16 #if land/ice area is 32 times bigger than ocean/mélange, classify as no front/unconfident prediction
	settings['inter_box_distance_threshold'] = full_size / 16
	settings['image_settings'] = dict()
	#To calculate confusion matrix, include images where no front can be detected.
	settings['negative_image_names'] = ['Hayes_LC08_L1TP_2016-06-07_080-237_T1_B5',
		'Helheim_LC08_L1TP_2018-06-06_232-013_T1_B5',
		'Helheim_LE07_L1TP_2006-05-12_232-013_T2_B4',
		'Helheim_LT05_L1TP_1991-07-14_232-013_T1_B4',
		'Kælvegletscher_LE07_L1TP_2012-08-25_231-012_T1_B4',
		'Nordre-Parallelgletsjer_LT05_L1TP_1997-05-22_229-012_T2_B4',
		'Rink-Isbrae_LT05_L1TP_1986-06-16_013-009_T1_B4',
		'Sermeq-Avannarleq-69_LC08_L1TP_2018-10-11_008-012_T1_B5',
		'Sermeq-Avannarleq-70_LC08_L1TP_2014-04-28_011-010_T1_B5',
		'Styrtegletsjer_LE07_L1TP_2010-04-16_229-012_T1_B4',
		'Umiammakku_LE07_L1TP_2011-06-20_014-009_T1_B4',
		'Upernavik-NE_LC08_L1TP_2013-09-28_015-008_T1_B5',
		'Upernavik-SE_LC08_L1TP_2016-05-06_016-008_T1_B5']
	
	metrics = dict()
	metrics['confidence_skip_count'] = 0
	metrics['no_detection_skip_count'] = 0
	metrics['front_count'] = 0
	metrics['image_skip_count'] = 0
	metrics['mean_deviations_pixels'] = np.array([])
	metrics['mean_deviations_meters'] = np.array([])
	metrics['validation_distances_pixels'] = np.array([])
	metrics['validation_distances_meters'] = np.array([])
	metrics['domain_mean_deviations_pixels'] = defaultdict(lambda: np.array([]))
	metrics['domain_mean_deviations_meters'] = defaultdict(lambda: np.array([]))
	metrics['domain_validation_distances_pixels'] = defaultdict(lambda: np.array([]))
	metrics['domain_validation_distances_meters'] = defaultdict(lambda: np.array([]))
	metrics['domain_validation_ious'] = defaultdict(lambda: np.array([]))
	metrics['resolution_deviation_array'] = np.zeros((0,2))
	metrics['validation_ious'] = np.array([])
	metrics['resolution_iou_array'] = np.zeros((0,2))
	metrics['true_negatives'] = 0
	metrics['false_negatives'] = 0
	metrics['false_positive'] = 0
	metrics['true_positives'] = 0
	
	#Each 256x256 image will be split into 9 overlapping 224x224 patches to reduce boundary effects
	#and ensure confident predictions. To normalize this when overlaying patches back together, 
	#generate normalization image that scales the predicted image based on number of patches each pixel is in.
	strides = int((full_size - img_size) / stride + 1) #(256-224 / 16 + 1) = 3
	pred_norm_image = np.zeros((full_size, full_size, 3))
	pred_norm_patch = np.ones((img_size, img_size, 2))
	for x in range(strides):
		for y in range(strides):
			x_start = x * stride
			x_end = x_start + img_size
			y_start = y * stride
			y_end = y_start + img_size
			pred_norm_image[x_start:x_end, y_start:y_end, 0:2] += pred_norm_patch
	settings['pred_norm_image'] = pred_norm_image
	
	return settings, metrics


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


if __name__ == '__main__':
	full_size = 256
	img_size = 224
	stride = 16
	try:
	    model
	except NameError:
	    model = compile_model(img_size)
	settings, metrics = initialize(model)
	
	main(settings, metrics)


