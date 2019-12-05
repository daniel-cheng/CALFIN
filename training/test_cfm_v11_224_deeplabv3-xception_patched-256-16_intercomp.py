import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

import sys, os, cv2, glob
sys.path.insert(1, 'keras-deeplab-v3-plus')
sys.path.insert(2, '../postprocessing')
from model_cfm_dual_wide_x65 import Deeplabv3
from AdamAccumulate import AdamAccumulate

#Only make first GPU visible on multi-gpu setups
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from skimage.io import imread
import error_analysis
from skimage.morphology import skeletonize
from scipy.spatial import distance
from matplotlib.lines import Line2D

from aug_generators_dual import aug_validation, create_unaugmented_data_patches_from_rgb_image

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


def plot_validation_results(image_name_base, raw_image, pred_image, polyline_image, empty_image, distances, mask_final_f32, index, dest_path, saving, scaling, edge_iou):
	"""Plots a standardized set of 6 plots for validation of the neural network, and quantifies its error per image."""
	#Set figure size for 1600x900 resolution, tight layout
	plt.rcParams["figure.figsize"] = (16,9)
	
	#Initialize plots
	hist_bins = 20
	f, axarr = plt.subplots(2, 3, num=index + 1)
	f.suptitle('Helheim ' + image_name_base, fontsize=18, weight='bold')
	raw_image_gray = np.stack((raw_image[:,:,0], raw_image[:,:,0], raw_image[:,:,0]), axis=-1)
	
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
	axarr[0,0].imshow(np.clip(raw_image_gray, 0.0, 1.0))
	axarr[0,0].set_title(r'$\bf{a)}$ Raw Subset')
	
	axarr[0,1].imshow(np.clip(raw_image, 0.0, 1.0))
	axarr[0,1].set_title(r'$\bf{b)}$ Preprocessed Input')
	axarr[0,1].legend(preprocess_legend, ['Raw', 'HDR', 'S/H'], prop={'weight': 'normal'}, facecolor='#eeeeee', loc='upper center', bbox_to_anchor=(0.5, 0.0), shadow=True, ncol=3)
	axarr[0,1].axis('off')
	
	axarr[0,2].imshow(np.clip(pred_image, 0.0, 1.0))
	axarr[0,2].set_title(r'$\bf{c)}$ NN Output')
	axarr[0,2].legend(nn_legend, ['Land/Ice', 'Front'], prop={'weight': 'normal'}, facecolor='#eeeeee', loc='upper center', bbox_to_anchor=(0.5, 0.0), shadow=True, ncol=2)
	axarr[0,2].axis('off')
	
	axarr[1,0].imshow(np.clip(np.stack((polyline_image[:,:,0], empty_image, empty_image), axis=-1) + raw_image_gray * 0.80, 0.0, 1.0))
	axarr[1,0].set_title(r'$\bf{d)}$ Extracted Front')
	axarr[1,0].legend(front_legend, ['Front'], prop={'weight': 'normal'}, facecolor='#eeeeee', loc='upper center', bbox_to_anchor=(0.5, 0.0), shadow=True, ncol=1)
	axarr[1,0].axis('off')
	
	axarr[1,1].imshow(np.clip(np.stack((polyline_image[:,:,0], mask_final_f32, empty_image), axis=-1) + raw_image_gray * 0.80, 0.0, 1.0))
	axarr[1,1].set_title(r'$\bf{e)}$ NN vs Ground Truth Front')
	axarr[1,1].set_xlabel('Jaccard Index: {:.4f}'.format(edge_iou))
	axarr[1,1].legend(comparison_legend, ['NN', 'GT'], prop={'weight': 'normal'}, facecolor='#eeeeee', loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)
	axarr[1,1].tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off') # labels along the bottom edge are off
	
	# which = both major and minor ticks are affected
	scaled_distances = distances * scaling
	axarr[1,2].hist(scaled_distances, bins=hist_bins, range=[0.0, 20.0 * scaling])
	axarr[1,2].set_xlabel('Distance to nearest point (mean=' + '{:.2f}m)'.format(mean_deviation * scaling))
	axarr[1,2].set_ylabel('Number of points')
	axarr[1,2].set_title(r'$\bf{f)}$ Per-pixel Pairwise Error (meters)')
	
	
	#Refresh plot if necessary
	plt.subplots_adjust(top = 0.90, bottom = 0.075, right = 0.975, left = 0.025, hspace = 0.3, wspace = 0.2)
	f.canvas.draw()
	f.canvas.flush_events()
	
	#Save figure
	if saving:
		plt.savefig(os.path.join(dest_path, image_name_base + '_validation.png'))


def plot_histogram(distances, index, dest_path, saving, scaling):
	"""Plots a standardized set of 6 plots for validation of the neural network, and quantifies its error per image."""
	#Initialize plots
	hist_bins = 20
	f, axarr = plt.subplots(1, 1, num=index + 1)
	f.suptitle('Helheim Inter-comparison: Per-point Distance from True Front', fontsize=16)
	
	scaled_distances = distances * scaling
	axarr.hist(scaled_distances, bins=hist_bins, range=[0.0, 20.0 * scaling])
	axarr.set_xlabel('Distance to nearest point (meters)')
	axarr.set_ylabel('Number of points')
	plt.figtext(0.5, 0.01, r'Mean Distance = {:.2f}m'.format(np.mean(scaled_distances)), wrap=True, horizontalalignment='center', fontsize=14, weight='bold')
		
	#Set figure size for 1600x900 resolution, tight layout
	plt.rcParams["figure.figsize"] = (8,4.5)
	plt.subplots_adjust(top = 0.90, bottom = 0.15, right = 0.90, left = 0.1, hspace = 0.25, wspace = 0.25)
	
	#Refresh plot if necessary
	f.canvas.draw()
	f.canvas.flush_events()
	
	#Save figure
	if saving:
		plt.savefig(os.path.join(dest_path, 'Helhim_mean_deviations_intercomp.png'))


def mask_fjord_boundary(fjord_boundary_final_f32, kernel, iterations, raw_image_gray_uint8, pred_image_gray_uint8):
	""" Helper funcction for performing optimiation on fjord boundaries.
		Erodes a single fjord boundary mask."""
	fjord_boundary_eroded_f32 = cv2.erode(fjord_boundary_final_f32.astype('float64'), kernel, iterations = iterations).astype(np.float32) #np.float32 [0.0, 255.0]
	fjord_boundary_eroded_f32 = np.where(fjord_boundary_eroded_f32 > 0.5, 1.0, 0.0)
	
	masked_pred_uint8 = pred_image_gray_uint8 * fjord_boundary_eroded_f32.astype(np.uint8)
	results_polyline = error_analysis.extract_front_indicators(raw_image_gray_uint8, masked_pred_uint8, 0, [256, 256])
	
	#If edge is detected, replace coarse edge mask in prediction image with connected polyline edge
	if not results_polyline is None:
		polyline_image = (results_polyline[0] / 255.0)
	else:
		polyline_image = np.zeros((full_size, full_size, 3))
	
	#Determine number of masked pixels
	x1, y2 = np.nonzero(polyline_image[:,:,0])
	num_masked_pixels = x1.shape[0]
	return num_masked_pixels, polyline_image, fjord_boundary_eroded_f32

def mask_polyline(raw_image, pred_image, fjord_boundary_final_f32, kernel):
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
	threshold = 0.5
	if maximal_erosions < iteration_limit - 1 and masked_pixels_diff[maximal_erosions + 1] < masked_pixels_diff[maximal_erosions] * threshold:
		maximal_erosions += 1
	
	#Redo fjord masking with known maximal erosion number
	results_masking = mask_fjord_boundary(fjord_boundary_final_f32, kernel, maximal_erosions, raw_gray_uint8, pred_gray_uint8)
	polyline_image = results_masking[1]
	
	return polyline_image


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
	initialized = True
	if initialized == False:
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
		print('Creating images...')
		print('-'*30)
		results = []
		yara_files = glob.glob(r"D:\Daniel\Documents\Github\CALFIN Repo Intercomp\training\data\validation\*Subset.png")
		validation_files = glob.glob(r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\validation\*B[0-9].png")
		fjord_boundary = imread(r"D:\Daniel\Documents\Github\CALFIN Repo Intercomp\training\data\fjord_boundaries\Helheim_fjord_boundaries.png")
		dest_path = r"D:\Daniel\Documents\Github\CALFIN Repo Intercomp\outputs\helheim_aligned"
		validation_files = yara_files
		total = len(validation_files)
		imgs = None
		imgs_mask = None
		i = 0
		return_images = True
		mean_deviations = np.array([])
		empty_image = np.zeros((full_size, full_size))
		validation_distances = np.array([])
		validation_ious = np.array([])
		scaling = 96.3 / 1.97
		
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
		for i in range(len(validation_files)):
			image_path = validation_files[i]
			image_name = image_path.split(os.path.sep)[-1]
			image_name_base = image_name.split('.')[0]
			mask_path = image_path.replace('Subset', 'Front')
			
			#Read in raw/mask image pair
			img_3_uint8 = imread(image_path) #np.uint16 [0, 65535]
			mask_uint8 = 255.0 - imread(mask_path) #np.uint16 [0, 65535]
			if img_3_uint8.shape[2] != 3:
				img_3_uint8 = np.concatenate((img_3_uint8, img_3_uint8, img_3_uint8))
			
			#Predict front using compiled model and retrieve results
			result = predict(model, img_3_uint8, mask_uint8, fjord_boundary, pred_norm_image, full_size, img_size, stride)
			raw_image = result[0]
			pred_image = result[1]
			mask_final_f32 = result[2]
			fjord_boundary_final_f32 = result[3]
			overlay = raw_image*0.75 + pred_image *0.25
			
			#Perform optimiation on fjord boundaries.
			#Continuously erode fjord boundary mask until fjord edges are masked.
			#This is detected by looking for large increases in pixels being masked,
			#followed by few pixels being masked.
			thickness = 3
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
			polyline_image = mask_polyline(raw_image, pred_image[:,:,0], fjord_boundary_final_f32, kernel)
								
			#Calculate and save mean deviation, distances, and IoU metrics based on new masked polyline
			mean_deviation, distances = calculate_mean_deviation(polyline_image[:,:,0], mask_final_f32)
			mean_deviations = np.append(mean_deviations, mean_deviation)
			validation_distances = np.append(validation_distances, distances)
			print("mean_deviation:", mean_deviation)
			
			#Dilate masks for easier visualization and IoU metric calculation.
			thickness = 3
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
			polyline_image = cv2.dilate(polyline_image.astype('float64'), kernel, iterations = 1).astype(np.float32) #np.float32 [0.0, 255.0]
			mask_final_f32 = cv2.dilate(mask_final_f32.astype('float64'), kernel, iterations = 1).astype(np.float32) #np.float32 [0.0, 255.0]
			
			#Calculate and save IoU metric
			pred_patch_4d = np.expand_dims(polyline_image[:,:,0], axis=0)
			mask_patch_4d = np.expand_dims(mask_final_f32, axis=0)
			edge_iou_score = calculate_edge_iou(mask_patch_4d, pred_patch_4d)
			validation_ious = np.append(validation_ious, edge_iou_score)
			print("edge_iou_score:", edge_iou_score)
			
			#Plot results
			if plotting == True:
				plot_validation_results(image_name_base, raw_image, pred_image, polyline_image, empty_image, distances, mask_final_f32, i, dest_path, saving, scaling, edge_iou_score)
			
			print('Done {0}: {1}/{2} images'.format(image_name, i + 1, total))
#			break
		mean_deviation_points = np.mean(validation_distances)
		mean_deviation_images = np.mean(mean_deviations)
		mean_edge_iou_score = np.mean(validation_ious)
		print("mean deviation (averaged over points): " + "{:.2f}".format(mean_deviation_points * scaling) + ' meters')
		print("mean deviation (averaged over images): " + "{:.2f}".format(mean_deviation_images * scaling) + ' meters')
		print("mean deviation (averaged over points): " + "{:.2f}".format(mean_deviation_points) + ' pixels')
		print("mean deviation (averaged over images): " + "{:.2f}".format(mean_deviation_images) + ' pixels')
		print("mean Jaccard index (Intersection over Union): " + "{:.4f}".format(mean_edge_iou_score))
		
		#Print histogram of all distance errors
#		plot_histogram(validation_distances, total, dest_path, saving, scaling)
		
		plt.show()