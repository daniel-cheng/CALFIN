from __future__ import print_function

import os, cv2, skimage, random, sys
import numpy as np

from skimage.io import imsave, imread
from skimage.transform import resize, rotate, rescale
from skimage.util import invert
from scipy.ndimage.filters import median_filter
from scipy import ndimage
from keras.applications import imagenet_utils

from aug_generators import aug_resize, aug_pad

data_path = 'landsat_raw_boundaries/'
temp_path = 'landsat_temp_boundaries/'

img_size = 256
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

def preprocess_input(x):
	"""Preprocesses a numpy array encoding a batch of images.
	# Arguments
		x: a 4D numpy array consists of RGB values within [0, 255].
	# Returns
		Input array scaled to [-1.,1.]
	"""
	return imagenet_utils.preprocess_input(x, mode='tf')

def create_unagumented_data_from_image(img, mask):	
	#Normalize inputs.
	img_pre = img.astype('float32')
	img_pre = preprocess_input(img_pre)
	
	img_resize = img_pre[np.newaxis,:,:,np.newaxis]
	if mask is None:
		mask_resize = None
	else:
		mask_resize = mask[np.newaxis,:,:,np.newaxis]
	
	return img_resize, mask_resize

def load_validation_data(img_size):
	imgs_validation = np.load('landsat_imgs_validation_boundaries_' + str(img_size) + '.npy').astype(np.float32)
	imgs_mask_validation = np.load('landsat_imgs_mask_validation_boundaries_' + str(img_size) + '.npy').astype(np.float32)
	return (imgs_validation, imgs_mask_validation)

def create_validation_data_from_image(img, mask, image_name, image_mask_name):
	#Important: rotating images in this case is important for training - otherwise, degenerates and picks false optimum	
	
#	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#	img = clahe.apply(img)
	
#	#Sharpen image.
#	blurred_img = ndimage.gaussian_filter(img, 3)
#	filter_blurred_img = ndimage.gaussian_filter(blurred_img, 1)
#	alpha = 40
#	sharpened = blurred_img + alpha * (blurred_img - filter_blurred_img)
	

	#Normalize inputs.
	img = (img / img.max() * 255).astype(np.uint8)
	imsave(os.path.join(temp_path, 'validation_full', image_name), img)
	imsave(os.path.join(temp_path, 'validation_full', image_mask_name), (mask * 255).astype(np.uint8))
	
	#Create img_size x img_size strided patches from image
	patches, maskPatches = create_unagumented_data_from_image(img, mask)

	return patches, maskPatches
	
def create_validation_data_from_directory(img_size):
	data_path = 'landsat_raw_boundaries'
	validation_data_path = os.path.join(data_path, 'validation_full')
	images = os.listdir(validation_data_path)
	augmentations = 1
	total = len(images) // 2 * augmentations
	imgs = None
	imgs_mask = None
	i = 0
	augs_resize = aug_resize(img_size=img_size)
	augs_pad = aug_pad(img_size=img_size)

	print('-'*30)
	print('Creating validation images...')
	print('-'*30)

	for image_name in images:
		if '_mask.png' in image_name or '_bqa.png' in image_name  or '_mtl.txt' in image_name or not os.path.isfile(os.path.join(validation_data_path, image_name)):
			continue
		image_mask_name = image_name.split('.')[0] + '_mask.png'
		image_pred_name = image_name.split('.')[0] + '_pred.png'
		img = imread(os.path.join(validation_data_path, image_name), as_gray=True)
		mask = imread(os.path.join(validation_data_path, image_mask_name), as_gray=True)
		imsave(os.path.join(temp_path, 'validation_full', image_mask_name), mask)
		img_resized = resize(img, (img_size, img_size), preserve_range=True)  #np.float32 [0.0, 65535.0]
		mask_resized = resize(mask, (img_size, img_size), preserve_range=True) #np.float32 [0.0, 255.0]
		
		#Convert greyscale to RGB greyscale, preserving as max range as possible in uint8 
		#(since it will be normalized again for imagenet means, it's ok if it's not divided by actual max of uint16)
		img = (img * (255.0 / img.max())).astype(np.uint8)
		
		#Resize image to max img_size
		#dat_resize = augs_resize(image=img, mask=mask)
		#img_rgb_resized = dat_resize['image']
		#mask_rgb_resized = dat_resize['mask']
		
		#Calculate edge from mask and dilate.
		mask_edges = cv2.Canny(mask_resized.astype(np.uint8), 100, 200)	
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		mask_edges = cv2.dilate(mask_edges.astype('float64'), kernel, iterations = 1)
		mask_edges = np.where(mask_edges > np.mean(mask_edges), 1.0, 0.0).astype('float32')

		#Pad image to img_size
		#dat_padded = augs_pad(image=img_rgb_resized, mask=mask_edges)
		#img_rgb_padded = dat_padded['image']
		#mask_rgb_padded = dat_padded['mask']
		
		patches, patches_mask = create_validation_data_from_image(img_resized, mask_edges, image_name, image_pred_name)
		
		if (imgs is not None):
			imgs = np.concatenate((imgs, patches))
			imgs_mask = np.concatenate((imgs_mask, patches_mask))
			if (imgs.shape[0] != imgs_mask.shape[0]):
				raise ValueError()
		else:
			imgs = patches
			imgs_mask = patches_mask

		i += 1
		print('Done: {0}/{1} images'.format(i, total))

	np.save('landsat_imgs_validation_boundaries_' + str(img_size) + '.npy', imgs)
	np.save('landsat_imgs_mask_validation_boundaries_' + str(img_size) + '.npy', imgs_mask)
		
if __name__ == '__main__':
#	create_validation_data_from_directory(sys.argv[1])
	create_validation_data_from_directory(256)
