import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from collections import defaultdict
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

import sys, cv2, glob
sys.path.insert(1, '../training/keras-deeplab-v3-plus')
sys.path.insert(2, '../training')
from plotting import plot_histogram, plot_scatter
from processing import compile_model, process
from postprocessing import postprocess

if __name__ == '__main__':
	#initialize settings and model if not already done
	full_size = 256
	img_size = 224
	stride = 16
	try:
	    model
	except NameError:
	    model = compile_model(img_size)
	
	processing = False
	plotting = True
	saving = True
	if processing == False:
		#Initialize plots
		plt.close('all')
		font = {'family' : 'normal',
		        'size'   : 14}
		plt.rc('font', **font)
		
		#Initialize variables
		print('-'*30)
		print('Processing images...')
		print('-'*30)
		validation_files = glob.glob(r"..\training\data\validation\*B[0-9].png")
		dest_path = r"..\outputs\validation"
		scaling = 96.3 / 1.97
		settings = dict()
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
		settings['dest_path'] = dest_path
		settings['save_path'] = r"..\processing\landsat_preds"
		settings['total'] = len(validation_files)
		settings['empty_image'] = np.zeros((settings['full_size'], settings['full_size']))
		settings['scaling'] = scaling
		settings['domain_scalings'] = dict()
		settings['mask_confidence_strength_threshold'] = 0.8
		settings['edge_confidence_strength_threshold'] = 0.6
		settings['image_settings'] = dict()
		
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

		
		#Each 256x256 image will be split into 9 overlapping 224x224 patches to reduce boundary effects
		#and ensure confident predictions. To normalize this when overlaying patches back together, 
		#generate normalization image that scales the predicted image based on number of patches pixel is in.
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
		
		#Begin processing validation images
		for i in range(0, len(validation_files)):
#		for i in range(110,112):
#		for i in range(22,23):
			 process(i, validation_files, settings, metrics)
			 postprocess(i, validation_files, settings, metrics)
					
		for domain in metrics['domain_mean_deviations_pixels'].keys():
			domain_scaling = settings['domain_scalings'][domain]
			domain_mean_deviation_points_pixels = np.nanmean(metrics['domain_validation_distances_pixels'][domain])
			domain_mean_deviation_points_meters = np.nanmean(metrics['domain_validation_distances_meters'][domain])
			domain_mean_deviation_images_pixels = np.nanmean(metrics['domain_mean_deviations_pixels'][domain])
			domain_mean_deviation_images_meters = np.nanmean(metrics['domain_mean_deviations_meters'][domain])
			domain_mean_edge_iou_score = np.nanmean(metrics['domain_validation_ious'][domain])
			print("{} mean deviation (averaged over points): {:.2f} meters".format(domain, domain_mean_deviation_points_meters))
			print("{} mean deviation (averaged over images): {:.2f} meters".format(domain, domain_mean_deviation_images_meters))
			print("{} mean deviation (averaged over points): {:.2f} pixels".format(domain, domain_mean_deviation_points_pixels))
			print("{} mean deviation (averaged over images): {:.2f} pixels".format(domain, domain_mean_deviation_images_pixels))
			print("{} mean Jaccard index (Intersection over Union): {:.4f}".format(domain, domain_mean_edge_iou_score))
			plot_histogram(metrics['domain_validation_distances_meters'][domain], domain + "_mean_deviations_meters", dest_path, saving, domain_scaling)
			
		mean_deviation_points_pixels = np.nanmean(metrics['validation_distances_pixels'])
		mean_deviation_points_meters = np.nanmean(metrics['validation_distances_meters'])
		mean_deviation_images_pixels = np.nanmean(metrics['mean_deviations_pixels'])
		mean_deviation_images_meters = np.nanmean(metrics['mean_deviations_meters'])
		mean_edge_iou_score = np.nanmean(metrics['validation_ious'])
		print("mean deviation (averaged over points): {:.2f} meters".format(mean_deviation_points_meters))
		print("mean deviation (averaged over images): {:.2f} meters".format(mean_deviation_images_meters))
		print("mean deviation (averaged over points): {:.2f} pixels".format(mean_deviation_points_pixels))
		print("mean deviation (averaged over images): {:.2f} pixels".format(mean_deviation_images_pixels))
		print("mean Jaccard index (Intersection over Union): {:.4f}".format(mean_edge_iou_score))
		
		#Print histogram of all distance errors
		plot_histogram(metrics['validation_distances_meters'], "all_mean_deviations_meters", dest_path, saving, scaling)
		
		#Print scatterplot of resolution errors
		plot_scatter(metrics['resolution_deviation_array'], "Validation Resolution vs Deviations", dest_path, saving)
		plot_scatter(metrics['resolution_iou_array'], "Validation Resolution vs IoU", dest_path, saving)
		plt.show()
		
		#Print final statistics
		print('mask_confidence_strength_threshold', settings['mask_confidence_strength_threshold'] , 
			'edge_confidence_strength_threshold', settings['edge_confidence_strength_threshold'] )
		print('image_skip_count:', metrics['image_skip_count'], 
			'front skip count:', metrics['no_detection_skip_count'] + metrics['confidence_skip_count'], 
			'no_detection_skip_count:', metrics['no_detection_skip_count'], 
			"confidence_skip_count:", metrics['confidence_skip_count'])
		print('total fronts:', metrics['front_count'])
		
		number_no_front_images = 13 #13 images are clouded/undetectable in the CALFIN validation set of 152 images
		number_total_images = len(validation_files)
		number_valid_images = len(validation_files) - number_no_front_images
		percent_images_with_fronts = (1 - metrics['image_skip_count'] / len(validation_files))* 100
		number_images_with_fronts = str(number_valid_images - metrics['image_skip_count'])
		total_images = str(number_valid_images)
		
#		false_positives = number_total_images - (number_valid_images + metrics['image_skip_count'])
#		false_negatives = 
#		true_positives = 
#		true_negatives = 
		print('% images with fronts: {:.2f} ({}/{})'.format(percent_images_with_fronts, number_images_with_fronts, total_images))