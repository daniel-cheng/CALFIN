from __future__ import print_function

import os, cv2, skimage, glob, shutil
import numpy as np

from skimage.morphology import skeletonize
from skimage.transform import resize
from skimage.io import imsave, imread
from random import shuffle
from aug_generators import aug_daniel, aug_pad, aug_resize, create_unagumented_data_from_image, create_unagumented_data_from_rgb_image
import sys
sys.path.insert(0, '../postprocessing')
from poisson_line import ordered_line_from_unordered_points_tree

data_path = 'data/'
temp_path = 'temp/'
img_size = 512
image_stride = img_size / 4
image_stride_range = 9

edge_filter = np.array((
	[-1, -1, -1],
	[-1, 8, -1],
	[-1, -1, -1]), dtype="int")

def extractPatches(im, window_shape=(img_size, img_size), stride=image_stride):
	#In order to extract patches, determine the resolution of the image needed to extract regular patches
	#Pad image if necessary
#	print(im.shape)
	nr, nc = im.shape

	#If image is smaller than window size, pad to fit.
	#else, pad to the least integer multiple of stride
	#Window shape is assumed to multiple of stride.
	#Find the smallest multiple of stride that is greater than image dimensions
	leastRowStrideMultiple = (np.ceil(nr / stride) * stride).astype(np.uint16)
	leastColStrideMultiple = (np.ceil(nc / stride) * stride).astype(np.uint16)
	#If image is smaller than window, pad to window shape. Else, pad to least stride multiple.
	nrPad = max(window_shape[0], leastRowStrideMultiple) - nr
	ncPad = max(window_shape[1], leastColStrideMultiple) - nc
	#Add Stride border around image, and nrPad/ncPad to image to make sure it is divisible by stride.
	stridePadding = int(stride / 2)
	paddingRow = (stridePadding, nrPad + stridePadding)
	paddingCol = (stridePadding, ncPad + stridePadding)
	padding = (paddingRow, paddingCol)
	imPadded = np.pad(im, padding, 'constant')

	patches = skimage.util.view_as_windows(imPadded, window_shape, stride)
	nR, nC, H, W = patches.shape
	nWindow = nR * nC
	patches = np.reshape(patches, (nWindow, H, W))
	return patches, nr, nc, nR, nC

def load_validation_data(img_size):
	imgs_validation = np.load('cfm_validation_imgs_padded_' + str(img_size) + '.npy').astype(np.float32)
	imgs_mask_validation = np.load('cfm_validation_masks_padded_' + str(img_size) + '.npy').astype(np.float32)
	return (imgs_validation, imgs_mask_validation)

def create_validation_data_from_directory(input_data_path, output_data_path, img_size):
	imgs, imgs_mask = create_data_from_directory(input_data_path, output_data_path, img_size)
	np.save('cfm_validation_imgs_padded_' + str(img_size) + '.npy', imgs)
	np.save('cfm_validation_masks_padded_' + str(img_size) + '.npy', imgs_mask)
	
def create_data_from_directory(input_train_data_path, output_train_data_path, img_size):
#	if os.path.exists(output_train_data_path):
#		shutil.rmtree(output_train_data_path)
#	os.mkdir(output_train_data_path)
	
	images = glob.glob(input_train_data_path + '/*[0-9].png')
	total = len(images)
	imgs = None
	imgs_mask = None
	i = 0
	augs_pad = aug_pad(img_size=img_size)
	augs_resize_img = aug_resize(img_size=img_size, interpolation=1)
	augs_resize_mask = aug_resize(img_size=img_size, interpolation=0)
	
	print('-'*30)
	print('Creating images...')
	print('-'*30)
	for image_path in images:
		image_name = image_path.split(os.path.sep)[-1]
		image_mask_name = image_name.split('.')[0] + '_mask.png'
		img_uint16 = imread(os.path.join(input_train_data_path, image_name), as_gray=True) #np.uint16 [0, 65535]
		mask_uint16 = imread(os.path.join(input_train_data_path, image_mask_name), as_gray=True) #np.uint16 [0, 65535]
		
		#Convert greyscale to RGB greyscale
		img_max = img_uint16.max()
		mask_max = mask_uint16.max()
		if img_max != 0.0:
			img_uint8 = np.round(img_uint16 / img_max * 255.0).astype(np.uint8) #np.uint8 [0, 255.0]
		else:
			img_uint8 = img_uint16.astype(np.uint8)
		if mask_max != 0.0:
			mask_uint8 = np.floor(mask_uint16 / mask_max * 255.0).astype(np.uint8) #np.uint8 [0, 255.0]
		else:
			mask_uint8 = mask_uint16.astype(np.uint8)
			
		#If downsizing image, proceed normally
		if img_uint8.shape[0] > img_size and img_uint8.shape[1] > img_size:
			img_3_uint8 = np.stack((img_uint8,)*3, axis=-1)
			mask_3_uint8 = np.stack((mask_uint8,)*3, axis=-1).astype(np.uint8)
			
			#Resize image while preserving aspect ratio
			dat = augs_resize_img(image=img_3_uint8, mask=mask_3_uint8)
			img_aug_3_uint8 = dat['image'] #np.uint8 [0, 255]
			mask_aug_uint8 = np.mean(dat['mask'], axis=2).astype(np.uint8) #np.uint8 [0, 255]
			
			#Calculate edge from original resolution mask
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
			mask_edge = cv2.Canny(mask_aug_uint8, 250, 255 * 2) #thresholds = Use diagonals to detect strong edges, then connect anything with at least a single edge
			mask_edge = cv2.dilate(mask_edge.astype('float64'), kernel, iterations = 1).astype(np.uint8)
			mask_edge_3_uint8 = np.stack((mask_edge,)*3, axis=-1)
			
			#Pad image if needed
			dat = augs_pad(image=img_aug_3_uint8, mask=mask_edge_3_uint8)
			img_final_3_uint8 = dat['image'] #np.uint8 [0, 255]
			mask_final_3_uint8 = dat['mask'] #np.uint8 [0, 255]
			mask_final_3_uint8 = np.where(mask_final_3_uint8 > 127, 1, 0).astype(np.uint8) #np.uint8 [0, 1]
#		else: 
#		#If upscaling image, resample edge
#		else:
#			#Calculate edge from original resolution mask
#			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#			mask_edge = cv2.Canny(mask_uint8, 250, 255 * 2) #thresholds = Use diagonals to detect strong edges, then connect anything with at least a single edge
#	#		mask_edge = cv2.dilate(mask_edge.astype('float64'), kernel, iterations = 1).astype(np.uint8)
#			edge_bianry = np.where(mask_edge > 127, 1, 0)
#			skeleton = skeletonize(edge_bianry) * 255
#			front_pixels = np.nonzero(skeleton)
#			try:
#				front_line = np.array(ordered_line_from_unordered_points_tree(front_pixels, img_uint8.shape, 4))
#				mask_edge_3_uint8 = np.stack((mask_edge,)*3, axis=-1)
#				img_3_uint8 = np.stack((img_uint8,)*3, axis=-1)
#				
#				#Resize image while preserving aspect ratio
#				dat_img = augs_resize_img(image=img_3_uint8)
#				img_aug_3_uint8 = dat_img['image'] #np.uint8 [0, 255]
#				
#				#Rescale points from original mask and draw scaled edge
#				mask_edge_3_uint8 = np.zeros(img_aug_3_uint8.shape)
#				points = np.array(front_line)
#				points_y = points[0,:] * img_aug_3_uint8.shape[0] / mask_uint8.shape[0]
#				points_x = points[1,:] * img_aug_3_uint8.shape[1] / mask_uint8.shape[1]
#		#		np.flip(points, axis=0)
#				scaled_tranposed_points = np.transpose(np.array([points_x, points_y])).astype(np.int32)
#				scaled_tranposed_curve = cv2.approxPolyDP(scaled_tranposed_points, 1.5, False)
#				cv2.polylines(mask_edge_3_uint8, [scaled_tranposed_curve], False, (255, 255, 255))
#				
#				mask_edge_uint8 = np.mean(mask_edge_3_uint8, axis=2)
#				mask_edge_uint8 = cv2.dilate(mask_edge_uint8.astype('float64'), kernel, iterations = 1).astype(np.uint8)
#				edge_bianry = np.where(mask_edge_uint8 > 127, 1, 0)
#				skeleton = skeletonize(edge_bianry) * 255
#				mask_edge_uint8 = cv2.dilate(skeleton.astype('float64'), kernel, iterations = 1).astype(np.uint8)
#				mask_edge_scaled_3_uint8 = np.stack((mask_edge_uint8,)*3, axis=-1)
#				
#				#Pad image if needed
#				dat = augs_pad(image=img_aug_3_uint8, mask=mask_edge_scaled_3_uint8)
#				img_final_3_uint8 = dat['image'] #np.uint8 [0, 255]
#				mask_final_3_uint8 = dat['mask'] #np.uint8 [0, 255]
#				mask_final_3_uint8 = np.where(mask_final_3_uint8 > 127, 1, 0).astype(np.uint8) #np.uint8 [0, 1]
#			except ValueError:
#				print('No front detected for:', image_name)
#				#Resize image while preserving aspect ratio
#				dat_img = augs_resize_img(image=img_3_uint8)
#				img_aug_3_uint8 = dat_img['image'] #np.uint8 [0, 255]
#				
#				#Pad image if needed
#				dat = augs_pad(image=img_aug_3_uint8)
#				img_final_3_uint8 = dat['image'] #np.uint8 [0, 255]
#				mask_final_3_uint8 = np.zeros(img_final_3_uint8.shape).astype(np.uint8)
			
			patches, maskPatches = create_unagumented_data_from_rgb_image(img_final_3_uint8, mask_final_3_uint8)
			
			imsave(os.path.join(output_train_data_path, image_name), np.round((patches[0,:,:,:] + 1) / 2 * 255).astype(np.uint8))
			imsave(os.path.join(output_train_data_path, image_name.split('.')[0] + '_mask.png'), (255 * maskPatches[0,:,:,:]).astype(np.uint8))
	#		imsave(os.path.join(output_train_data_path, image_name.split('.')[0] + '_pred.png'), mask_edge_scaled_3_uint8)
			
			if (imgs is not None):
				imgs = np.concatenate((imgs, patches))
				imgs_mask = np.concatenate((imgs_mask, maskPatches))
				if (imgs.shape[0] != imgs_mask.shape[0]):
					raise ValueError()
			else:
				imgs = patches
				imgs_mask = maskPatches

		i += 1
		print('Done: {0}/{1} images'.format(i, total))
	return imgs, imgs_mask

if __name__ == '__main__':
	
	input_data_path = 'data/validation_original'
	output_data_path = 'data/validation_padded_512'
	
	create_validation_data_from_directory(input_data_path, output_data_path, img_size)
#	input_data_path = 'data/train_original'
#	output_data_path = 'data/train_padded_512'
#	create_data_from_directory(input_data_path, output_data_path, img_size)
