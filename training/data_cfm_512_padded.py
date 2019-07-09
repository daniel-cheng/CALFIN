from __future__ import print_function

import os, cv2, skimage, glob
import numpy as np

from skimage.transform import resize
from skimage.io import imsave, imread
from random import shuffle
from aug_generators import aug_daniel, create_unagumented_data_from_image

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

def create_validation_data_from_directory(img_size):
	validation_data_path = 'data/validation_padded_512'
	temp_path = 'temp/validation_padded_512'
	images = glob.glob(validation_data_path + '/*[0-9].png')
	shuffle(images)
	total = len(images)
	imgs = None
	imgs_mask = None
	i = 0
	
	print('-'*30)
	print('Creating validation images...')
	print('-'*30)
	for image_path in images:
		image_name = image_path.split(os.path.sep)[-1]
		image_mask_name = image_name.split('.')[0] + '_mask.png'
		img_uint16 = imread(os.path.join(validation_data_path, image_name), as_gray=True) #np.uint16 [0, 65535]
		mask_uint16 = imread(os.path.join(validation_data_path, image_mask_name), as_gray=True) #np.uint16 [0, 65535]
		img_f64 = resize(img_uint16, (img_size, img_size), preserve_range=True)  #np.float64 [0.0, 65535.0]
		mask_f64 = resize(mask_uint16, (img_size, img_size), preserve_range=True) #np.float64 [0.0, 65535.0]
		
		#Convert greyscale to RGB greyscale, preserving as max range as possible in uint8 
		img_max = img_f64.max()
		mask_max = mask_f64.max()
		if (img_max != 0.0):
			img_uint8 = np.round(img_f64 / img_max * 255.0).astype(np.uint8) #np.uint8 [0, 255]
		if (mask_max != 0.0):
			mask_uint8 = np.floor(mask_f64 / mask_max * 255.0).astype(np.uint8) #np.uint8 [0, 255]
		mask_uint8 = np.where(mask_uint8 > 127, 1.0, 0.0).astype('float32') #np.float32 [0.0, 1.0]
		patches, maskPatches = create_unagumented_data_from_image(img_uint8, mask_uint8)
		
		imsave(os.path.join(temp_path, image_name), np.round((patches[0,:,:,0] + 1) / 2 * 255).astype(np.uint8))
		imsave(os.path.join(temp_path, image_mask_name), (255 * maskPatches[0,:,:,0]).astype(np.uint8))
	
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

	np.save('cfm_validation_imgs_padded_' + str(img_size) + '.npy', imgs)
	np.save('cfm_validation_masks_padded_' + str(img_size) + '.npy', imgs_mask)
		
if __name__ == '__main__':
	create_validation_data_from_directory(img_size)
