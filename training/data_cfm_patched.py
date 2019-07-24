from __future__ import print_function

import os, cv2, skimage, glob, shutil
import numpy as np

from skimage.morphology import skeletonize
from skimage.transform import resize
from skimage.io import imsave, imread
from skimage.exposure import equalize_adapthist
from random import shuffle
from aug_generators import aug_daniel, aug_pad, aug_resize, create_unaugmented_data_patches_from_image

import sys
sys.path.insert(0, '../postprocessing')
from poisson_line import ordered_line_from_unordered_points_tree

data_path = 'data/'
temp_path = 'temp/'

def load_validation_data(full_size, img_size, stride, regen=False):
	#If patched validation/train images have not yet been generated, do so now
	id_str = str(full_size) + '_' + str(img_size) + '_' + str(stride)
	if not os.path.exists('data/validation_patched_' + id_str) or regen == True:
		input_data_path = 'data/validation'
		output_data_path = 'data/validation_patched_' + id_str 
		if not os.path.exists('data/validation_patched_' + id_str):
			os.mkdir(output_data_path)
		create_validation_data_from_directory(input_data_path, output_data_path, full_size, img_size, stride)
	if not os.path.exists('data/train_patched_' + id_str) or regen == True:
		input_data_path = 'data/train'
		output_data_path = 'data/train_patched_' + id_str
		if not os.path.exists('data/train_patched_' + id_str):
			os.mkdir(output_data_path)
		create_data_from_directory(input_data_path, output_data_path, full_size, img_size, stride)

	imgs_validation = np.load('cfm_validation_imgs_patched_' + id_str + '.npy').astype(np.float32)
	imgs_mask_validation = np.load('cfm_validation_masks_patched_' + id_str + '.npy').astype(np.float32)
	return imgs_validation, imgs_mask_validation

def create_validation_data_from_directory(input_data_path, output_data_path, full_size, img_size, stride):
	id_str = str(full_size) + '_' + str(img_size) + '_' + str(stride)
	imgs, imgs_mask = create_data_from_directory(input_data_path, output_data_path, full_size, img_size, stride, return_images=True)
	np.save('cfm_validation_imgs_patched_' + id_str + '.npy', imgs)
	np.save('cfm_validation_masks_patched_' + id_str + '.npy', imgs_mask)
	
def create_data_from_directory(input_path, output_path, full_size, img_size, stride, return_images=False):
	keep_aspect_ratio = False
	images = glob.glob(input_path + '/*[0-9].png')
	total = len(images)
	imgs = None
	imgs_mask = None
	i = 0
	augs_pad = aug_pad(img_size=full_size)
	augs_resize_img = aug_resize(img_size=full_size, interpolation=1)
	augs_resize_mask = aug_resize(img_size=full_size, interpolation=0)
	thickness = np.floor(3.0 / 224.0 * img_size).astype(int)
	print('Front thickness: ', thickness)
	
	print('-'*30)
	print('Creating images...')
	print('-'*30)
	for image_path in images:
		image_name = image_path.split(os.path.sep)[-1]
		image_edge_name = image_name.split('.')[0] + '_edge.png'
		image_mask_name = image_name.split('.')[0] + '_mask.png'
		img_uint16 = imread(os.path.join(input_path, image_name), as_gray=True) #np.uint16 [0, 65535]
		mask_uint16 = imread(os.path.join(input_path, image_mask_name), as_gray=True) #np.uint16 [0, 65535]
		img_f64 = resize(img_uint16, (full_size, full_size), preserve_range=True)  #np.float64 [0.0, 65535.0]
		mask_f64 = resize(mask_uint16, (full_size, full_size), order=0, preserve_range=True) #np.float64 [0.0, 65535.0]
		
		#Convert greyscale to RGB greyscale
		img_max = img_f64.max()
		mask_max = mask_f64.max()
		if img_max != 0.0:
			img_uint8 = np.round(img_f64 / img_max * 255.0).astype(np.uint8) #np.uint8 [0, 255.0]
		else:
			img_uint8 = img_f64.astype(np.uint8)
		if mask_max != 0.0:
			mask_uint8 = np.floor(mask_f64 / mask_max * 255.0).astype(np.uint8) #np.uint8 [0, 255.0]
		else:
			mask_uint8 = mask_f64.astype(np.uint8)
		# Adaptive Equalization
		img_uint8 = (equalize_adapthist(img_uint8, clip_limit=0.01) * 255.0).astype(np.uint8)
		
		#Ensure squishing is not too extreme
		aspect_ratio = 1
		if keep_aspect_ratio == False:
			if img_uint8.shape[0] > img_uint8.shape[1]:
				aspect_ratio = img_uint8.shape[0] / img_uint8.shape[1]
			else:
				aspect_ratio = img_uint8.shape[1] / img_uint8.shape[0]
		
		#If image is within bounds, save it.
		if img_uint8.shape[0] >= full_size and img_uint8.shape[1] >= full_size and aspect_ratio >= 0.825:
			img_3_uint8 = np.stack((img_uint8,)*3, axis=-1)
			mask_3_uint8 = np.stack((mask_uint8,)*3, axis=-1).astype(np.uint8)
			
			#Resize image while preserving aspect ratio (Useless right now)
			dat = augs_resize_img(image=img_3_uint8, mask=mask_3_uint8)
			img_aug_3_uint8 = dat['image'] #np.uint8 [0, 255]
			mask_aug_uint8 = np.mean(dat['mask'], axis=2).astype(np.uint8) #np.uint8 [0, 255]
			
			#Calculate edge from original resolution mask
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
			mask_edge = cv2.Canny(mask_aug_uint8, 250, 255 * 2) #thresholds = Use diagonals to detect strong edges, then connect anything with at least a single edge
			mask_edge = cv2.dilate(mask_edge.astype('float64'), kernel, iterations = 1).astype(np.uint8)
			mask_edge_3_uint8 = np.stack((mask_edge,)*3, axis=-1)
			
			#Pad image if needed (Useless right now)
			dat = augs_pad(image=img_aug_3_uint8, mask=mask_edge_3_uint8)
			img_final_f32 = np.mean(dat['image'], axis=2).astype('float32') #np.float32 [0.0, 255.0]
			mask_final_f32 = np.mean(dat['mask'], axis=2).astype('float32') #np.float32 [0.0, 255.0]
			mask_final_f32 = np.where(mask_final_f32 > 127.0, 1.0, 0.0) #np.float32 [0.0, 1.0]
			
			patches, maskPatches = create_unaugmented_data_patches_from_image(img_final_f32, mask_final_f32, window_shape=(img_size, img_size), stride=stride)
			
#			imsave(os.path.join(output_path, image_name), np.round((patches[0,:,:,0] + 1) / 2 * 255).astype(np.uint8))
#			imsave(os.path.join(output_path, image_name.split('.')[0] + '_mask.png'), (255 * maskPatches[0,:,:,0]).astype(np.uint8))
			
#			imsave(os.path.join(output_path, image_name), img_final_f32.astype(np.uint8))
#			imsave(os.path.join(output_path, image_edge_name), (mask_final_f32 * 255).astype(np.uint8))
#			imsave(os.path.join(output_path, image_mask_name), (mask_uint8).astype(np.uint8))
			
			imsave(os.path.join(output_path, image_name), img_final_f32.astype(np.uint8))
			imsave(os.path.join(output_path, image_mask_name), (mask_final_f32 * 255).astype(np.uint8))
#			imsave(os.path.join(output_path, image_mask_name), (mask_uint8).astype(np.uint8))
			
			if return_images:
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
	full_size = 512
	img_size = 448
	stride = 32
	load_validation_data(full_size, img_size, stride, regen=True)