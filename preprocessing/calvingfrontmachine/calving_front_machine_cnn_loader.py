from __future__ import print_function

import os
import numpy as np
import skimage

from skimage.io import imsave, imread
from skimage.transform import resize, rotate, rescale
import random

data_path = 'landsat_raw/'
temp_path = 'landsat_temp/'

image_rows = 256
image_cols = 256
image_stride = 128
image_stride_range = 9
img_rows = 256
img_cols = 256

def extractPatches(im, window_shape=(96, 96), stride=96):
	#In order to extract patches, determine the resolution of the image needed to extract regular patches
	#Pad image if necessary
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
	imPadded = np.pad(im, padding, 'reflect')

	patches = skimage.util.view_as_windows(imPadded, window_shape, stride)
	nR, nC, H, W = patches.shape
	nWindow = nR * nC
	patches = np.reshape(patches, (nWindow, H, W))
	return patches, nr, nc, nR, nC

def preprocess(imgs):
	imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=imgs.dtype)
	for i in range(imgs.shape[0]):
		imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)
	imgs_p = imgs_p[..., np.newaxis]
	return imgs_p

def create_train_data():
	train_data_path = os.path.join(data_path, 'train')
	images = os.listdir(train_data_path)
	total = len(images) // 2

	imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
	imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

	i = 0
	print('-'*30)
	print('Creating training images...')
	print('-'*30)
	for image_name in images:
		if 'mask' in image_name or not os.path.isfile(os.path.join(train_data_path, image_name)):
			continue
		image_mask_name = image_name.split('.')[0] + '_mask.png'

		img = imread(os.path.join(train_data_path, image_name), as_gray=True)
		img_mask = imread(os.path.join(train_data_path, image_mask_name), as_gray=True)

		img = np.array([img])
		img_mask = np.array([img_mask])

		imgs[i] = img
		imgs_mask[i] = img_mask

		print('Done: {0}/{1} images'.format(i, total))
		i += 1
	print('Loading done.')

	np.save('landsat_imgs_train_original.npy', imgs)
	np.save('landsat_imgs_mask_train_original.npy', imgs_mask)
	print('Saving to .npy files done.')

def create_train_data_from_image(img, mask, angle, scale):
	#Important: rotating images in this case is important for training - otherwise, degenerates and picks false optimum

	img = rescale(img, scale, mode='reflect')
	mask = rescale(mask, scale, mode='reflect')
	img = rotate(img, angle, mode='reflect')
	mask = rotate(mask, angle, mode='reflect')


	mean = np.mean(img)  # mean for data centering
	std = np.std(img)  # std for data normalization
	img = img.astype('float32')
	img -= mean
	img /= std

	image_stride_random = image_stride + random.randint(-image_stride_range, image_stride_range)
	patches, nr, nc, nR, nC = extractPatches(img, (image_rows, image_cols), image_stride_random)
	maskPatches, nr, nc, nR, nC = extractPatches(mask, (image_rows, image_cols), image_stride_random)
	patches = preprocess(patches)
	maskPatches = preprocess(maskPatches)

	return patches, maskPatches

def create_train_data_from_directory():
	train_data_path = os.path.join(data_path, 'train_full')
	images = os.listdir(train_data_path)
	augmentations = 6
	total = len(images) // 3 * augmentations
	imgs = None
	imgs_mask = None
	i = 0

	print('-'*30)
	print('Creating training images...')
	print('-'*30)

	for augmentation in range(augmentations):
		for image_name in images:
			if '_mask.png' in image_name or '_bqa.png' in image_name  or '_mtl.txt' in image_name or not os.path.isfile(os.path.join(train_data_path, image_name)):
				continue
			angle = random.randint(1, 359)
			scale = random.uniform(0.25, 1.0)
			image_mask_name = image_name.split('.')[0] + '_mask.png'
			img = imread(os.path.join(train_data_path, image_name), as_gray=True)
			img_mask = imread(os.path.join(train_data_path, image_mask_name), as_gray=True)
			img_mask = np.where(img_mask > np.mean(img_mask), 1, 0).astype('float32')
			patches, patches_mask = create_train_data_from_image(img, img_mask, angle, scale)

			if (imgs is not None):
				imgs = np.concatenate((imgs, patches))
				imgs_mask = np.concatenate((imgs_mask, patches_mask))
				if (imgs.shape[0] != imgs_mask.shape[0]):
					raise ValueError()
				print(imgs.shape, imgs_mask.shape, image_name)
			else:
				imgs = patches
				imgs_mask = patches_mask

			i += 1
			print('Done: {0}/{1} images'.format(i, total))

#	for i in range(len(imgs)):
#		imsave(os.path.join(temp_path, str(i) + '_raw.png'), (imgs[i,:,:,0] *255).astype(np.uint8))
#		imsave(os.path.join(temp_path, str(i) + '_pred.png'), (imgs_mask[i,:,:,0] *255).astype(np.uint8))

	np.save('landsat_imgs_train.npy', imgs)
	np.save('landsat_imgs_mask_train.npy', imgs_mask)
	print('Saving to .npy files done.')

def load_train_data():
	imgs_train = np.load('landsat_imgs_train.npy')
	imgs_mask_train = np.load('landsat_imgs_mask_train.npy')
	return imgs_train, imgs_mask_train

def create_test_data():
	train_data_path = os.path.join(data_path, 'test')
	images = os.listdir(train_data_path)
	total = len(images)

	imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
	imgs_id = np.ndarray((total, ), dtype=np.int32)

	i = 0
	print('-'*30)
	print('Creating test images...')
	print('-'*30)
	for image_name in images:
		img_id = int(image_name.split('.')[0])
		img = imread(os.path.join(train_data_path, image_name), as_gray=True)

		img = np.array([img])

		imgs[i] = img
		imgs_id[i] = img_id

		i += 1
		if i % 100 == 0:
			print('Done: {0}/{1} images'.format(i, total))

	np.save('landsat_imgs_test.npy', imgs)
	np.save('landsat_imgs_id_test.npy', imgs_id)
	print('Saving to .npy files done.')

def load_test_data():
	imgs_test = np.load('landsat_imgs_test.npy')
	imgs_id = np.load('landsat_imgs_id_test.npy')
	return imgs_test, imgs_id

def load_test_data_from_image(img, stride):
	mean = np.mean(img)  # mean for data centering
	std = np.std(img)  # std for data normalization
	img = img.astype('float32')
	img -= mean
	img /= std

	patches, nr, nc, nR, nC = extractPatches(img, (image_rows, image_cols), stride)
	patches = preprocess(patches)

#	print('-'*30)
#	print('Loading done.')
#	print('-'*30)

	return patches, nr, nc, nR, nC

if __name__ == '__main__':
	create_train_data_from_directory()
#	create_train_data()
#	create_test_data()