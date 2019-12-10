import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from collections import defaultdict

import sys, os, cv2, glob, shutil, gdal
sys.path.insert(1, 'keras-deeplab-v3-plus')
sys.path.insert(2, '../postprocessing')
from model_cfm_dual_wide_x65 import Deeplabv3
from AdamAccumulate import AdamAccumulate

#Only make first GPU visible on multi-gpu setups
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from skimage.transform import resize
from skimage.io import imread, imsave
import error_analysis
from scipy.spatial import distance
from matplotlib.lines import Line2D

from aug_generators_dual import create_unaugmented_data_patches_from_rgb_image

full_size = 256
img_size = 224
stride = 16
K.set_image_data_format('channels_last')  # TF dimension ordering in this code


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


def plot_validation_results(image_name_base, raw_image, original_raw, pred_image, polyline_image, empty_image, distances_meters, mask_image, index, dest_path, saving, scaling, edge_iou, fjord_boundary_image):
	"""Plots a standardized set of 6 plots for validation of the neural network, and quantifies its error per image."""
	#Set figure size for 1600x900 resolution, tight layout
	plt.rcParams["figure.figsize"] = (16,9)
	
	#Initialize plots
	hist_bins = 20
	f, axarr = plt.subplots(2, 3, num=index)
	f.suptitle(image_name_base, fontsize=18, weight='bold')
	
	#Create the color key for each subplots' legends	
	preprocess_legend = [Line2D([0], [0], color='#ff0000', lw=4),
					     Line2D([0], [0], color='#00ff00', lw=4),
	                     Line2D([0], [0], color='#0000ff', lw=4)]
	nn_legend = [Line2D([0], [0], color='#00ff00', lw=4),
	             Line2D([0], [0], color='#ff0000', lw=4)]
	front_legend = [Line2D([0], [0], color='#ff0000', lw=4)]
	comparison_legend = [Line2D([0], [0], color='#ff0000', lw=4),
					     Line2D([0], [0], color='#00ff00', lw=4)]
	
	#Begin plotting the 2x3 validation results output
	original_raw_gray = np.stack((original_raw[:,:,0], original_raw[:,:,0], original_raw[:,:,0]), axis=-1)
	raw_image_gray = np.stack((raw_image[:,:,0], raw_image[:,:,0], raw_image[:,:,0]), axis=-1)
	axarr[0,0].imshow(np.clip(original_raw_gray, 0.0, 1.0))
	axarr[0,0].set_title(r'$\bf{a)}$ Raw Subset')
	
	raw_image = np.clip(raw_image, 0.0, 1.0)
	axarr[0,1].imshow(raw_image)
	axarr[0,1].set_title(r'$\bf{b)}$ Preprocessed Input')
	axarr[0,1].legend(preprocess_legend, ['Raw', 'HDR', 'S/H'], prop={'weight': 'normal'}, facecolor='#eeeeee', loc='upper center', bbox_to_anchor=(0.5, 0.0), shadow=True, ncol=3)
	axarr[0,1].axis('off')
	
	pred_image = np.clip(pred_image, 0.0, 1.0)
	axarr[0,2].imshow(pred_image)
	axarr[0,2].set_title(r'$\bf{c)}$ NN Output')
	axarr[0,2].legend(nn_legend, ['Land/Ice', 'Front'], prop={'weight': 'normal'}, facecolor='#eeeeee', loc='upper center', bbox_to_anchor=(0.5, 0.0), shadow=True, ncol=2)
	axarr[0,2].axis('off')
	
	extracted_front = np.clip(np.stack((polyline_image[:,:,0], empty_image, empty_image), axis=-1) + raw_image * 0.8, 0.0, 1.0)
	axarr[1,0].imshow(extracted_front)
	axarr[1,0].set_title(r'$\bf{d)}$ Extracted Front')
	axarr[1,0].legend(front_legend, ['Front'], prop={'weight': 'normal'}, facecolor='#eeeeee', loc='upper center', bbox_to_anchor=(0.5, 0.0), shadow=True, ncol=1)
	axarr[1,0].axis('off')
	
	overlay = np.clip(np.stack((polyline_image[:,:,0], mask_image, empty_image), axis=-1) + raw_image * 0.8, 0.0, 1.0)
	axarr[1,1].imshow(overlay)
	axarr[1,1].set_title(r'$\bf{e)}$ NN vs Ground Truth Front')
	axarr[1,1].set_xlabel('Jaccard Index: {:.4f}'.format(edge_iou))
	axarr[1,1].legend(comparison_legend, ['NN', 'GT'], prop={'weight': 'normal'}, facecolor='#eeeeee', loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)
	axarr[1,1].tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off') # labels along the bottom edge are off
	
	# which = both major and minor ticks are affected
	axarr[1,2].hist(distances_meters, bins=hist_bins, range=[0.0, 20.0 * scaling])
	axarr[1,2].set_xlabel('Distance to nearest point (mean=' + '{:.2f}m)'.format(np.mean(distances_meters)))
	axarr[1,2].set_ylabel('Number of points')
	axarr[1,2].set_title(r'$\bf{f)}$ Per-pixel Pairwise Error (meters)')
	
	
	#Refresh plot if necessary
	plt.subplots_adjust(top = 0.90, bottom = 0.075, right = 0.975, left = 0.025, hspace = 0.3, wspace = 0.2)
	f.canvas.draw()
	f.canvas.flush_events()
	
	#Save figure
	if saving:
		plt.savefig(os.path.join(dest_path, image_name_base + '_' + index + '_validation.png'))
		imsave(os.path.join(dest_path, image_name_base + '_' + index + '_original_raw.png'), original_raw_gray)
		imsave(os.path.join(dest_path, image_name_base + '_' + index + '_subset_raw.png'), raw_image)
		imsave(os.path.join(dest_path, image_name_base + '_' + index + '_pred.png'), pred_image)
		imsave(os.path.join(dest_path, image_name_base + '_' + index + '_front_only.png'), polyline_image)
		imsave(os.path.join(dest_path, image_name_base + '_' + index + '_overlay_front.png'), extracted_front)
		imsave(os.path.join(dest_path, image_name_base + '_' + index + '_overlay_comparison.png'), overlay)
		


def plot_histogram(distances, name, dest_path, saving, scaling):
	"""Plots a standardized set of 6 plots for validation of the neural network, and quantifies its error per image."""
	#Initialize plots
	hist_bins = 20
	f, axarr = plt.subplots(1, 1, num=name)
	f.suptitle('Validation Set: Per-point Distance from True Front', fontsize=16)
	
	axarr.hist(distances, bins=hist_bins, range=[0.0, 20.0 * scaling])
	axarr.set_xlabel('Distance to nearest point (meters)')
	axarr.set_ylabel('Number of points')
	plt.figtext(0.5, 0.01, r'Mean Distance = {:.2f}m'.format(np.mean(distances)), wrap=True, horizontalalignment='center', fontsize=14, weight='bold')
		
	#Set figure size for 1600x900 resolution, tight layout
	plt.rcParams["figure.figsize"] = (8,4.5)
	plt.subplots_adjust(top = 0.90, bottom = 0.15, right = 0.90, left = 0.1, hspace = 0.25, wspace = 0.25)
	
	#Refresh plot if necessary
	f.canvas.draw()
	f.canvas.flush_events()
	
	#Save figure
	if saving:
		plt.savefig(os.path.join(dest_path, name + '.png'))
		

def plot_scatter(data, name, dest_path, saving):
	"""Plots a standardized set of 6 plots for validation of the neural network, and quantifies its error per image."""
	#Initialize plots
	f, axarr = plt.subplots(1, 1, num=name)
	f.suptitle('Validation Set: Per-point Distance from True Front', fontsize=16)
	
	axarr.scatter(data[:,0], data[:,1])
	axarr.set_xlabel('Resolution (meters per pixel)')
	axarr.set_ylabel('Average Mean Distance')
		
	#Set figure size for 1600x900 resolution, tight layout
	plt.rcParams["figure.figsize"] = (8,4.5)
	plt.subplots_adjust(top = 0.90, bottom = 0.15, right = 0.90, left = 0.1, hspace = 0.25, wspace = 0.25)
	
	#Refresh plot if necessary
	f.canvas.draw()
	f.canvas.flush_events()
	
	#Save figure
	if saving:
		plt.savefig(os.path.join(dest_path, name + '.png'))


def remove_small_components(image:np.ndarray):
	image = image.astype('uint8')
	#find all your connected components (white blobs in your image)
	nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
	#connectedComponentswithStats yields every seperated component with information on each of them, such as size
	sizes = stats[:,cv2.CC_STAT_AREA]
	ordering = np.argsort(-sizes)
	#print(sizes, ordering)
	
	min_size_floor = output.size * 0.0001
	if len(ordering) > 1:
		min_size = sizes[ordering[1]] * 0.5
	# print(min_size)
	#your answer image
	largeComponents = np.zeros((output.shape))
	#for every component in the image, you keep it only if it's above min_size
	#Skip first, since it's the background color
	bounding_boxes = [[0, 0, image.shape[0], image.shape[1]]]
	for i in range(1, len(sizes)):
		# print(sizes, ordering[i])
		if sizes[ordering[i]] >= min_size_floor and sizes[ordering[i]] >= min_size:
			mask_indices = output == ordering[i]
			x, y = np.nonzero(mask_indices)
			min_x = min(x)
			delta_x = max(x) - min_x
			min_y = min(y)
			delta_y = max(y) - min_y
			bounding_boxes.append([min_x, min_y, delta_x, delta_y])
			largeComponents[mask_indices] = image[mask_indices]
#			bounding_boxes = 
	return largeComponents.astype(np.float32), bounding_boxes


def mask_fjord_boundary(fjord_boundary_final_f32, kernel, iterations, raw_image_gray_uint8, pred_image_gray_uint8, mask=True):
	""" Helper funcction for performing optimiation on fjord boundaries.
		Erodes a single fjord boundary mask."""
	fjord_boundary_eroded_f32 = cv2.erode(fjord_boundary_final_f32.astype('float64'), kernel, iterations = iterations).astype(np.float32) #np.float32 [0.0, 255.0]
	fjord_boundary_eroded_f32 = np.where(fjord_boundary_eroded_f32 > 0.5, 1.0, 0.0)
	
	if mask:
		masked_pred_uint8 = pred_image_gray_uint8 * fjord_boundary_eroded_f32.astype(np.uint8)
		results_polyline = error_analysis.extract_front_indicators(raw_image_gray_uint8, masked_pred_uint8, 0, [256, 256])
		
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
		results_polyline = error_analysis.extract_front_indicators(raw_image_gray_uint8, pred_image_gray_uint8, 0, [256, 256])
		
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

def mask_polyline(raw_image, pred_image, fjord_boundary_final_f32, kernel, recursion=True):
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
		results_masking = mask_fjord_boundary(fjord_boundary_final_f32, kernel, j, raw_gray_uint8, pred_gray_uint8)
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
	results_masking = mask_fjord_boundary(fjord_boundary_final_f32, kernel, maximal_erosions, raw_gray_uint8, pred_gray_uint8, mask=False)
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

def compile_model():
	"""Compile the CALFIN Neural Network model and loads pretrained weights."""
	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)
	img_shape = (img_size, img_size, 3)
	model = Deeplabv3(input_shape=img_shape, classes=16, OS=16, backbone='xception', weights=None)
	
	model.compile(optimizer=AdamAccumulate(lr=1e-4, accum_iters=2))
	model.summary()
	model.load_weights('cfm_weights_patched_dual_wide_x65_224_e65_iou0.5136.h5')
	
	return model


if __name__ == '__main__':
	#initialize once, run many times
	try:
	    model
	except NameError:
	    model = compile_model()
	
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
		results = []
		validation_files = glob.glob(r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\validation\*B[0-9].png")
		fjord_boundaries_path = r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\fjord_boundaries"
		tif_source_path = r"D:\Daniel\Documents\Github\CALFIN Repo\preprocessing\CalvingFronts\tif"
		dest_path = r"D:\Daniel\Documents\Github\CALFIN Repo\outputs\validation"
		save_path = r"D:\Daniel\Documents\Github\CALFIN Repo\processing\landsat_preds"
		total = len(validation_files)
		imgs = None
		imgs_mask = None
		i = 0
		return_images = True
		mean_deviations_pixels = np.array([])
		mean_deviations_meters = np.array([])
		empty_image = np.zeros((full_size, full_size))
		validation_distances_pixels = np.array([])
		validation_distances_meters = np.array([])
		validation_ious = np.array([])
		scaling = 96.3 / 1.97
		domain_scalings = dict()
		domain_mean_deviations_pixels = defaultdict(lambda: np.array([]))
		domain_mean_deviations_meters = defaultdict(lambda: np.array([]))
		domain_validation_distances_pixels = defaultdict(lambda: np.array([]))
		domain_validation_distances_meters = defaultdict(lambda: np.array([]))
		domain_validation_ious = defaultdict(lambda: np.array([]))
		domain_validation_skips = defaultdict(lambda: np.array([]))
		domain_validation_total = defaultdict(lambda: np.array([]))
		resolution_deviation_array = np.zeros((0,2))
		resolution_iou_array = np.zeros((0,2))
		no_detection_skip_count = 0
		confidence_skip_count = 0
		front_count = 0
		image_skip_count = 0
		mask_confidence_strength_threshold = 0.8
		edge_confidence_strength_threshold = 0.6
		
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
			
		#Begin processing validation images
#		for i in range(0, len(validation_files)):
#		for i in range(110,112):
		for i in range(22,23):
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
			raw_save_path = os.path.join(save_path, image_name_base + '_raw.png')
			mask_save_path = os.path.join(save_path, image_name_base + '_mask.png')
			pred_save_path = os.path.join(save_path, image_name_base + '_pred.png')
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
			xMin = geoTransform[0]
			yMax = geoTransform[3]
			xMax = xMin + geoTransform[1] * geotiff.RasterXSize
			yMin = yMax + geoTransform[5] * geotiff.RasterYSize
			
			#Transform vertices
			top_left = np.array([xMin, yMax])
			scale = np.array([xMax - xMin, yMin - yMax])
			meters_per_native_pixel = (np.abs(geoTransform[1]) + np.abs(geoTransform[5])) / 2
			resolution_1024 = (img_3_uint8.shape[0] + img_3_uint8.shape[1])/2
			resolution_native = (geotiff.RasterXSize + geotiff.RasterYSize) / 2
			meters_per_1024_pixel = resolution_native / resolution_1024 * meters_per_native_pixel
			
			#Predict front using compiled model and retrieve results
			result = predict(model, img_3_uint8, mask_uint8, fjord_boundary, pred_norm_image, full_size, img_size, stride)
			raw_image = result[0]
			pred_image = result[1]
			mask_image = result[2]
			fjord_boundary_final_f32 = result[3]
			overlay = raw_image*0.75 + pred_image *0.25
			
			#recalculate scaling
			resolution_256 = (raw_image.shape[0] + raw_image.shape[1])/2
			meters_per_256_pixel = resolution_1024 / resolution_256  * meters_per_1024_pixel
				
			#Save out images to training/landsat_preds
			shutil.copy2(image_path, raw_save_path)
			shutil.copy2(mask_path, mask_save_path)
			imsave(pred_save_path, pred_image)
			
			#Perform optimiation on fjord boundaries.
			#Continuously erode fjord boundary mask until fjord edges are masked.
			#This is detected by looking for large increases in pixels being masked,
			#followed by few pixels being masked.
			thickness = 3
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
			results_pred = mask_polyline(raw_image, pred_image[:,:,0], fjord_boundary_final_f32, kernel)
			results_mask = mask_polyline(raw_image, mask_image, fjord_boundary_final_f32, kernel)
			polyline_image = results_pred[0]
			bounding_boxes = results_pred[2]
			mask_final_f32 = results_mask[0][:,:,0]
			fjord_boundary_eroded_f32 = results_mask[1]
			
			#Calculate and save mean deviation, distances, and IoU metrics based on new masked polyline
			mean_deviation, distances = calculate_mean_deviation(polyline_image[:,:,0], mask_final_f32)
			print("mean_deviation (pixels):", mean_deviation, "mean_deviation (meters):", mean_deviation * meters_per_256_pixel)
			
			#Dilate masks for easier visualization and IoU metric calculation.
			polyline_image_dilated = cv2.dilate(polyline_image.astype('float64'), kernel, iterations = 1).astype(np.float32) #np.float32 [0.0, 255.0]
			mask_final_dilated_f32 = cv2.dilate(mask_final_f32.astype('float64'), kernel, iterations = 1).astype(np.float32) #np.float32 [0.0, 255.0]
			
			#Calculate and save IoU metric
			pred_patch_4d = np.expand_dims(polyline_image_dilated[:,:,0], axis=0)
			mask_patch_4d = np.expand_dims(mask_final_dilated_f32, axis=0)
			edge_iou_score = calculate_edge_iou(mask_patch_4d, pred_patch_4d)
			print("edge_iou_score:", edge_iou_score)
			
			original_raw = raw_image
			
			plot_validation_results(image_name_base, raw_image, original_raw, pred_image, polyline_image_dilated, empty_image, mean_deviation * meters_per_256_pixel, mask_final_dilated_f32, str(i + 1) + '-0', dest_path, saving, scaling, edge_iou_score, fjord_boundary_eroded_f32)
			
			#For each calving front, subset the image AGAIN and predict. This helps accuracy for
			#inputs with large scaling/downsampling ratios
			box_counter = 0
			
			found_front = False
			if len(bounding_boxes) > 1:
				for bounding_box in bounding_boxes[1:]:
					box_counter += 1
					fractional_bounding_box = np.array(bounding_box) / full_size
	#				if fractional_bounding_box[2] < 0.66 and fractional_bounding_box[3] < 0.66:
					#Try to get nearest square subset with padding equal to size of initial front
					print(bounding_box, fractional_bounding_box)
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
					
					#Perform subsetting
					resolution_subset = sub_padding * 2 #not simplified for clarity - just find average dimensions	
					meters_per_subset_pixel = meters_per_1024_pixel
					img_3_subset_uint8 = img_3_uint8[sub_x1:sub_x2, sub_y1:sub_y2, :]
					mask_subset_uint8 = mask_uint8[sub_x1:sub_x2, sub_y1:sub_y2]
					fjord_subset_boundary = fjord_boundary[sub_x1:sub_x2, sub_y1:sub_y2]
					
					#Repredict
					result = predict(model, img_3_subset_uint8, mask_subset_uint8, fjord_subset_boundary, pred_norm_image, full_size, img_size, stride)
					raw_image = result[0]
					pred_image = result[1]
					mask_image = result[2]
					fjord_boundary_final_f32 = result[3]
					overlay = raw_image*0.75 + pred_image *0.25
					
					edge_confidence_strength_indices = np.nan_to_num(pred_image[:,:,0]) > 0.05
					edge_confidence_strength = np.mean(np.abs(0.5 - np.nan_to_num(pred_image[:,:,0][edge_confidence_strength_indices])))
					mask_confidence_strength_indices = np.nan_to_num(pred_image[:,:,1]) > 0.05
					mask_confidence_strength = np.mean(np.abs(0.5 - np.nan_to_num(pred_image[:,:,1][mask_confidence_strength_indices])))
					
					print("edge_confidence_strength", edge_confidence_strength * 2, "mask_confidence_strength", mask_confidence_strength * 2)
					if mask_confidence_strength * 2 < mask_confidence_strength_threshold or edge_confidence_strength * 2 < edge_confidence_strength_threshold:
	#				if mask_confidence_strength * 2 < 0.7 or edge_confidence_strength * 2 < 0.6:
#					if mask_confidence_strength * 2 < 0.7 or edge_confidence_strength * 2 < 0.6:
						print("not confident, skipping")
						confidence_skip_count += 1
						continue
	#edge_confidence_strength 0.36093274 mask_confidence_strength 0.48205513
	#edge_confidence_strength 0.2558244 mask_confidence_strength 0.38819787
					#recalculate scaling
					meters_per_subset_pixel = resolution_subset / resolution_256  * meters_per_1024_pixel
					if domain not in domain_scalings:
						domain_scalings[domain] = meters_per_subset_pixel
					
					#Redo masking
					results_pred = mask_polyline(raw_image, pred_image[:,:,0], fjord_boundary_final_f32, kernel)
					results_mask = mask_polyline(raw_image, mask_image, fjord_boundary_final_f32, kernel)
					polyline_image = results_pred[0]
					bounding_boxes_pred = results_pred[2]
					mask_final_f32 = results_mask[0][:,:,0]
					bounding_boxes_mask = results_mask[2]
					fjord_boundary_eroded_f32 = results_mask[1]
	#				polyline_image[:,:,0], bounding_boxes = remove_small_components(polyline_image[:,:,0])
	#				mask_final_f32, bounding_boxes = remove_small_components(mask_final_f32)
		
					#mask out largest front
					if len(bounding_boxes_pred) < 2:
						print("no front detected, skipping")
						no_detection_skip_count += 1
						continue
					polyline_image = mask_bounding_box(bounding_boxes_pred, polyline_image)
					mask_final_f32 = mask_bounding_box(bounding_boxes_pred, mask_final_f32)
					
					#Save out images to training/landsat_preds
#					shutil.copy2(image_path, raw_save_path)
#					shutil.copy2(mask_path, mask_save_path)
#					imsave(pred_save_path, pred_image)
			
	#					#If not performing subsetting, set scaling equal to default 256 ratio
	#					if domain not in domain_scalings:
	#						domain_scalings[domain] = meters_per_256_pixel
						
					#Calculate and save mean deviation, distances, and IoU metrics based on new masked polyline
					mean_deviation_subset, distances = calculate_mean_deviation(polyline_image[:,:,0], mask_final_f32)
					mean_deviation_subset_meters = mean_deviation_subset * meters_per_subset_pixel
					mean_deviations_pixels = np.append(mean_deviations_pixels, mean_deviation_subset)
					mean_deviations_meters = np.append(mean_deviations_meters, mean_deviation_subset_meters)
					validation_distances_pixels = np.append(validation_distances_pixels, distances)
					validation_distances_meters = np.append(validation_distances_meters, distances * meters_per_subset_pixel)
					domain_mean_deviations_pixels[domain] = np.append(domain_mean_deviations_pixels[domain], mean_deviation_subset)
					domain_mean_deviations_meters[domain] = np.append(domain_mean_deviations_meters[domain], mean_deviation_subset_meters)
					domain_validation_distances_pixels[domain] = np.append(domain_validation_distances_pixels[domain], distances)
					domain_validation_distances_meters[domain] = np.append(domain_validation_distances_meters[domain], distances * meters_per_subset_pixel)
					resolution_deviation_array = np.concatenate((resolution_deviation_array, np.array([[meters_per_subset_pixel, mean_deviation_subset_meters]])))
					print("mean_deviation_subset (pixels):", mean_deviation_subset, "mean_deviation_subset (meters):", mean_deviation_subset * meters_per_subset_pixel)
					print("mean_deviation_difference (pixels) (- = good):", mean_deviation_subset - mean_deviation, "mean_deviation_difference (meters) (- = good):", mean_deviation_subset * meters_per_subset_pixel - mean_deviation * meters_per_256_pixel)
					
					#Dilate masks for easier visualization and IoU metric calculation.
					polyline_image_dilated = cv2.dilate(polyline_image.astype('float64'), kernel, iterations = 1).astype(np.float32) #np.float32 [0.0, 255.0]
					mask_final_dilated_f32 = cv2.dilate(mask_final_f32.astype('float64'), kernel, iterations = 1).astype(np.float32) #np.float32 [0.0, 255.0]
					
					#Calculate and save IoU metric
					pred_patch_4d = np.expand_dims(polyline_image_dilated[:,:,0], axis=0)
					mask_patch_4d = np.expand_dims(mask_final_dilated_f32, axis=0)
					edge_iou_score_subset = calculate_edge_iou(mask_patch_4d, pred_patch_4d)
					validation_ious = np.append(validation_ious, edge_iou_score)
					domain_validation_ious[domain] = np.append(domain_validation_ious[domain], edge_iou_score)
					resolution_iou_array = np.concatenate((resolution_iou_array, np.array([[meters_per_subset_pixel, edge_iou_score]])))
					print("edge_iou_score_subset:", edge_iou_score_subset, "edge_iou_score change (+ = good):", edge_iou_score_subset - edge_iou_score)
					
					#Plot results
					if plotting == True:
						distance_meters = distances * meters_per_subset_pixel
						plot_validation_results(image_name_base, raw_image, original_raw, pred_image, polyline_image_dilated, empty_image, distance_meters, mask_final_dilated_f32, str(i + 1) + '-'  + str(box_counter), dest_path, saving, scaling, edge_iou_score_subset, fjord_boundary_eroded_f32)
						front_count += 1
						found_front = True
			if not found_front:
				image_skip_count += 1
			print('Done {0}: {1}/{2} images'.format(image_name, i + 1, total))
#					break
					
		for domain in domain_mean_deviations_pixels.keys():
			domain_scaling = domain_scalings[domain]
			domain_mean_deviation_points_pixels = np.nanmean(domain_validation_distances_pixels[domain])
			domain_mean_deviation_points_meters = np.nanmean(domain_validation_distances_meters[domain])
			domain_mean_deviation_images_pixels = np.nanmean(domain_mean_deviations_pixels[domain])
			domain_mean_deviation_images_meters = np.nanmean(domain_mean_deviations_meters[domain])
			domain_mean_edge_iou_score = np.nanmean(domain_validation_ious[domain])
			print(domain + " mean deviation (averaged over points): " + "{:.2f}".format(domain_mean_deviation_points_meters) + ' meters')
			print(domain + " mean deviation (averaged over images): " + "{:.2f}".format(domain_mean_deviation_images_meters) + ' meters')
			print(domain + " mean deviation (averaged over points): " + "{:.2f}".format(domain_mean_deviation_points_pixels) + ' pixels')
			print(domain + " mean deviation (averaged over images): " + "{:.2f}".format(domain_mean_deviation_images_pixels) + ' pixels')
			print(domain + " mean Jaccard index (Intersection over Union): " + "{:.4f}".format(domain_mean_edge_iou_score))
			plot_histogram(domain_validation_distances_meters[domain], domain + "_mean_deviations_meters", dest_path, saving, domain_scaling)
			
		mean_deviation_points_pixels = np.nanmean(validation_distances_pixels)
		mean_deviation_points_meters = np.nanmean(validation_distances_meters)
		mean_deviation_images_pixels = np.nanmean(mean_deviations_pixels)
		mean_deviation_images_meters = np.nanmean(mean_deviations_meters)
		mean_edge_iou_score = np.nanmean(validation_ious)
		print("mean deviation (averaged over points): " + "{:.2f}".format(mean_deviation_points_meters) + ' meters')
		print("mean deviation (averaged over images): " + "{:.2f}".format(mean_deviation_images_meters) + ' meters')
		print("mean deviation (averaged over points): " + "{:.2f}".format(mean_deviation_points_pixels) + ' pixels')
		print("mean deviation (averaged over images): " + "{:.2f}".format(mean_deviation_images_pixels) + ' pixels')
		print("mean Jaccard index (Intersection over Union): " + "{:.4f}".format(mean_edge_iou_score))
		
		
		#Print histogram of all distance errors
		plot_histogram(validation_distances_meters, "all_mean_deviations_meters", dest_path, saving, scaling)
		
		#Print scatterplot of resolution errors
		plot_scatter(resolution_deviation_array, "Validation Resolution vs Deviations", dest_path, saving)
		plot_scatter(resolution_iou_array, "Validation Resolution vs IoU", dest_path, saving)
		
		plt.show()
		
		print('mask_confidence_strength_threshold', mask_confidence_strength_threshold, 'edge_confidence_strength_threshold', edge_confidence_strength_threshold)
		print('image_skip_count:', image_skip_count, 'front skip count:', no_detection_skip_count + confidence_skip_count, 'no_detection_skip_count:', no_detection_skip_count, "confidence_skip_count:", confidence_skip_count)
		print('total fronts:', front_count)
		print('% images with fronts: ' + "{:.2f}".format((1 - image_skip_count / len(validation_files))* 100), '(' + str(len(validation_files) - image_skip_count) + '/' + str(len(validation_files)) + ')')