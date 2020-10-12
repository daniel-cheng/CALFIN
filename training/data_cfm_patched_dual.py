from __future__ import print_function

import os, cv2, skimage, glob, shutil
import numpy as np

from skimage.morphology import skeletonize
from skimage.transform import resize
from skimage.io import imsave, imread
from skimage.exposure import equalize_adapthist
from random import shuffle
from aug_generators_dual import aug_daniel, aug_pad, aug_resize, create_unaugmented_data_patches_from_rgb_image
#from iphigen_hdr import iphigen_hdr
import matplotlib.pyplot as plt
import numpngw
from dateutil.parser import parse

data_path = 'data/'
temp_path = 'temp/'

def load_validation_data(full_size, img_size, stride, regen=False):
	#If patched validation/train images have not yet been generated, do so now
	id_str = str(full_size) + '_' + str(img_size) + '_' + str(stride)
	domains = ['Upernavik', 'Jakobshavn', 'Kong-Oscar', 'Kangiata-Nunaata', 'Hayes', 'Rink-Isbrae', 'Kangerlussuaq', 'Helheim']
	if not os.path.exists('cfm_validation_imgs_patched_dual_' + id_str + '.npy') or regen == True:
		input_data_path = 'data/validation'
		output_data_path = 'data/validation_patched_dual_' + id_str 
		if not os.path.exists('data/validation_patched_dual_' + id_str):
			os.mkdir(output_data_path)
		create_validation_data_from_directory(input_data_path, output_data_path, full_size, img_size, stride, '')
		for domain in domains:
			create_validation_data_from_directory(input_data_path, output_data_path, full_size, img_size, stride, domain)

	input_data_path = 'data/train'
	output_data_path = 'data/train_patched_dual_' + id_str 
	if not os.path.exists(output_data_path) or regen == True:
		if not os.path.exists('data/train_patched_dual_' + id_str):
			os.mkdir(output_data_path)
		create_data_from_directory(input_data_path, output_data_path, full_size, img_size, stride)

	imgs_validation = [np.load('cfm_validation_imgs_patched_dual_' + id_str + '.npy').astype(np.float32)]
	imgs_mask_validation = [np.load('cfm_validation_masks_patched_dual_' + id_str + '.npy').astype(np.float32)]
	for domain in domains:
		domain_id_str = id_str + '_' + domain
		imgs_validation.append(np.load('cfm_validation_imgs_patched_dual_' + domain_id_str + '.npy').astype(np.float32))
		imgs_mask_validation.append(np.load('cfm_validation_masks_patched_dual_' + domain_id_str + '.npy').astype(np.float32))
	return imgs_validation, imgs_mask_validation

def create_validation_data_from_directory(input_data_path, output_data_path, full_size, img_size, stride, prefix):
	if prefix == '':
		id_str = str(full_size) + '_' + str(img_size) + '_' + str(stride)
	else:
		id_str = str(full_size) + '_' + str(img_size) + '_' + str(stride) + '_' + prefix
	imgs, imgs_mask = create_data_from_directory(input_data_path, output_data_path, full_size, img_size, stride, prefix=prefix + '*', return_images=True)
	np.save('cfm_validation_imgs_patched_dual_' + id_str + '.npy', imgs)
	np.save('cfm_validation_masks_patched_dual_' + id_str + '.npy', imgs_mask)
	
def create_data_from_directory(input_path, output_path, full_size, img_size, stride, prefix='*', return_images=False):
	keep_aspect_ratio = False
	images = glob.glob(input_path + '/' + prefix + '*.png')
	images = list(filter(lambda x: '_mask' not in x, images))
	total = len(images)
	imgs = None
	imgs_mask = None
	i = 0
	#thickness = np.floor(3.0 / 224.0 * img_size).astype(int)
	thickness = 3 
	print('Front thickness: ', thickness)
	
	print('-'*30)
	print('Creating images...')
	print('-'*30)
	for image_path in images:
		image_name = image_path.split(os.path.sep)[-1]
		image_name_base = image_name.split('.')[0]
		image_mask_name = image_name_base + '_mask.png'
				
		img_3_uint16 = imread(os.path.join(input_path, image_name)) #np.uint16 [0, 65535]
		mask_uint16 = imread(os.path.join(input_path, image_mask_name), as_gray=True) #np.uint16 [0, 65535]
		img_3_f64 = resize(img_3_uint16, (full_size, full_size), preserve_range=True)  #np.float64 [0.0, 65535.0]
		mask_f64 = resize(mask_uint16, (full_size, full_size), order=0, preserve_range=True) #np.float64 [0.0, 65535.0]
		
		#Convert greyscale to RGB greyscale
		img_max = img_3_f64.max()
		img_min = img_3_f64.min()
		img_range = img_max - img_min
		mask_max = mask_f64.max()
		if (img_max != 0.0 and img_range > 255.0):
			img_3_uint8 = np.round(img_3_f64 / img_max * 255.0).astype(np.uint8) #np.float32 [0, 65535.0]
		else:
			img_3_uint8 = img_3_f64.astype(np.uint8)
		if (mask_max != 0.0):
			mask_uint8 = np.floor(mask_f64 / mask_max * 255.0).astype(np.uint8) #np.uint8 [0, 255]
		else:
			mask_uint8 = mask_f64.astype(np.uint8)
		mask_3_uint8 = np.stack((mask_uint8,)*3, axis=-1)
		
#		# Adaptive Equalization Image
#		img_adapteq_uint8 = (equalize_adapthist(img_uint8, clip_limit=0.01) * 255.0).astype(np.uint8)
#		
#		#HDR Image
#		img_3_uint8 = np.stack((img_uint8,)*3, axis=-1)
#		img_hdr_uint8 = np.mean(iphigen_hdr(img_3_uint8), axis=2).astype(np.uint8)
		
		#Ensure squishing is not too extreme
		aspect_ratio = 1
		if keep_aspect_ratio == False:
			if img_3_uint16.shape[0] > img_3_uint16.shape[1]:
				aspect_ratio = img_3_uint16.shape[0] / img_3_uint16.shape[1]
			else:
				aspect_ratio = img_3_uint16.shape[1] / img_3_uint16.shape[0]
		
		#If image is within bounds, save it.
		if img_3_uint16.shape[0] >= full_size and img_3_uint16.shape[1] >= full_size and aspect_ratio >= 0.825:
			#Calculate edge from original resolution mask
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
			mask_edge = cv2.Canny(mask_3_uint8, 250, 255 * 2) #thresholds = Use diagonals to detect strong edges, then connect anything with at least a single edge
			mask_edge_f32 = cv2.dilate(mask_edge.astype('float64'), kernel, iterations = 1).astype(np.float32) #np.float32 [0.0, 255.0]
			
			#Pad image if needed (Useless right now)
			img_final_f32 = img_3_uint8.astype(np.float32) #np.float32 [0.0, 255.0]
			mask_r_f32 = np.where(mask_edge_f32 > 127.0, 1.0, 0.0)
			mask_g_f32 = mask_uint8.astype(np.float32) / 255.0
			mask_b_f32 = mask_r_f32
			mask_final_f32 = np.stack((mask_r_f32, mask_g_f32, mask_b_f32), axis=-1) #np.float32 [0.0, 1.0]
			
			patches, maskPatches = create_unaugmented_data_patches_from_rgb_image(img_final_f32, mask_final_f32, window_shape=(img_size, img_size, 3), stride=stride, mask_channels=2)
			
#			imsave(os.path.join(output_path, image_name), np.round((patches[0,:,:,0] + 1) / 2 * 255).astype(np.uint8))
#			imsave(os.path.join(output_path, image_name.split('.')[0] + '_mask.png'), (255 * maskPatches[0,:,:,0]).astype(np.uint8))
			
#			imsave(os.path.join(output_path, image_name), img_final_f32.astype(np.uint8))
#			imsave(os.path.join(output_path, image_edge_name), (mask_final_f32 * 255).astype(np.uint8))
#			imsave(os.path.join(output_path, image_mask_name), (mask_uint8).astype(np.uint8))
			
			img_final_uint16 = (img_final_f32 * 257.0).astype(np.uint16)
			
			numpngw.write_png(os.path.join(output_path, image_name), img_final_uint16)
			#imsave(os.path.join(output_path, image_name_r), img_final_uint8[:,:,0])
			#imsave(os.path.join(output_path, image_name_g), img_final_uint8[:,:,1])
			#imsave(os.path.join(output_path, image_name_b), img_final_uint8[:,:,2])
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
		print('Done {0}: {1}/{2} images'.format(image_name, i, total))
	return imgs, imgs_mask

if __name__ == '__main__':
	full_size = 256
	img_size = 224
	stride = 16
	id_str = str(full_size) + '_' + str(img_size) + '_' + str(stride)
	#load_validation_data(full_size, img_size, stride, regen=True)

	input_data_path = 'data/train_petermann'
	output_data_path = 'data/train_patched_dual_' + id_str 
	if not os.path.exists('data/train_patched_dual_' + id_str):
		os.mkdir(output_data_path)
	create_data_from_directory(input_data_path, output_data_path, full_size, img_size, stride)
