import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
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
from pyproj import Proj, transform
		
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
	
def predict_edge(model, img_3_uint8, mask_uint8, fjord_boundary, pred_norm_image, full_size, img_size, stride):
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

def predict(model, img_3_uint8, mask_uint8, fjord_boundary, pred_norm_image, full_size, img_size, stride):
	"""Takes in a neural network model, input image, mask, fjord boundary mask, and windowded normalization image to create prediction output.
		Uses the full original size, image patch size, and stride legnth as additional variables."""
	#Rescale mask while perserving continuity of edges
	thickness = 3
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
	mask_uint8_dilated = cv2.dilate(mask_uint8.astype('float64'), kernel, iterations = 1).astype(np.uint8) #np.uint8 [0, 255]
	
	augs = aug_validation(img_size=full_size)
	dat = augs(image=img_3_uint8, mask=mask_uint8_dilated)
	img_3_aug_f32 = dat['image'].astype('float32') #np.float32 [0.0, 255.0]
	mask_aug_f32 = dat['mask'].astype('float32') #np.float32 [0.0, 255.0]
	
	#Reskeletonize mask once at desired resolution to ensure single pixel wide target
	mask_rescaled_f32 = np.where(mask_aug_f32[:,:,0] > 127.0, 1.0, 0.0)
	mask_final_f32 = skeletonize(mask_rescaled_f32) #np.float32 [0.0, 1.0]
	
	#Resize fjord boundary
	dat = augs(image=fjord_boundary)
	fjord_boundary_final_f32 = dat['image'].astype('float32') #np.float32 [0.0, 1.0]
	
	#Ensure correct scaling and no clipping of data values

	img_max = img_3_aug_f32.max()
	img_min = img_3_aug_f32.min()
	img_range = img_max - img_min
	if (img_max != 0.0 and img_range > 255.0):
		img_3_aug_f32 = np.round(img_3_aug_f32 / img_max * 255.0) #np.float32 [0, 65535.0]
	else:
		img_3_aug_f32 = img_3_aug_f32.astype(np.float32)
	img_final_f32 = img_3_aug_f32.astype(np.float32) #np.float32 [0.0, 255.0]
	
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
	hist_bins = 10
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
	
	overlay = np.clip(np.stack((polyline_image[:,:,0], mask_image, empty_image), axis=-1) + raw_image_gray * 0.8, 0.0, 1.0)
	axarr[1,1].imshow(overlay)
	axarr[1,1].set_title(r'$\bf{e)}$ NN vs Ground Truth Front')
	axarr[1,1].set_xlabel('Jaccard Index: {:.4f}'.format(edge_iou))
	axarr[1,1].legend(comparison_legend, ['NN', 'GT'], prop={'weight': 'normal'}, facecolor='#eeeeee', loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)
	axarr[1,1].tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off') # labels along the bottom edge are off
	
	# which = both major and minor ticks are affected
	axarr[1,2].hist(distances_meters, bins=hist_bins, range=[0.0, 8.0 * scaling])
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
	plt.figtext(0.5, 0.00, r'Mean Distance = {:.2f}m'.format(np.mean(distances)), wrap=True, horizontalalignment='center', fontsize=14, weight='bold')
		
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
	create_model = False
	if create_model:
		try:
		    model
		except NameError:
		    model = compile_model()
		
	#Initialize plots
	plt.close('all')
	font = {'family' : 'normal',
	        'size'   : 14}
	
	plt.rc('font', **font)
	
	#Validate Mohajerani et al. performance on selected CALFIN data
	validate = True
	if validate:
		print('-'*30)
		print('Validating intercomparison data...')
		print('-'*30)


		validation_files = []
		calfin_path = r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\validation"
		validation_files = glob.glob(r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\validation\*B[0-9].png")
		fjord_boundaries_path = r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\fjord_boundaries"
		tif_source_path = r"D:\Daniel\Documents\Github\CALFIN Repo\preprocessing\CalvingFronts\tif"
		dest_path = r"D:\Daniel\Documents\Github\CALFIN Repo\outputs\validation"
		save_path = r"D:\Daniel\Documents\Github\CALFIN Repo\processing\landsat_preds"
		
		coordinates_path = r'D:\Daniel\Documents\Github\CALFIN Repo Intercomp\postprocessing\output_helheim_calfin'
		for csv_path in glob.glob(os.path.join(coordinates_path, '*')):
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
			calfin_name = '_'.join([domain, satellite, level, date_string, path_row_string, tier, band])
			calfin_raw_path = os.path.join(calfin_path, calfin_name + '.png')
			calfin_mask_path = os.path.join(calfin_path, calfin_name + '_mask.png')
			calfin_tif_path = os.path.join(tif_source_path, domain, year, calfin_name + '.tif')
			
			print(calfin_raw_path, calfin_mask_path, calfin_tif_path)
			#install pyproj from pip instead of conda on windows to avoid undefined epsg
			inProj = Proj('epsg:3413')
			outProj = Proj('epsg:32624')
			transformed_coordinates = transform(inProj,outProj,coordinates[:,0], coordinates[:,1])
			print(csv_path)
			plt.figure()
			plt.plot(transformed_coordinates[0], transformed_coordinates[1])
			plt.show()
			
#			validation_files += raw_file_path
				