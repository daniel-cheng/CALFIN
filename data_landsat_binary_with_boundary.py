from __future__ import print_function

import os, cv2, skimage, random
import numpy as np

from skimage.io import imsave, imread
from skimage.transform import resize, rotate, rescale
from skimage.util import invert
from scipy.ndimage.filters import median_filter
from scipy import ndimage


data_path = 'landsat_raw_boundaries/'
temp_path = 'landsat_temp_boundaries/'

image_rows = 256
image_cols = 256
image_stride = 96
image_stride_range = 9
img_rows = 256
img_cols = 256

edge_filter = np.array((
	[-1, -1, -1],
	[-1, 8, -1],
	[-1, -1, -1]), dtype="int")

def extractPatches(im, window_shape=(256, 256), stride=96):
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

def noisy(noise_typ,image):
	if noise_typ == "gauss":
		row,col= image.shape
		mean = 0
		var = 0.1
		sigma = var**0.5
		gauss = np.random.normal(mean,sigma,(row,col))
		gauss = gauss.reshape(row,col)
		noisy = image + gauss * 0.1
		return noisy
	elif noise_typ == "s&p":
		row,col = image.shape
		s_vs_p = 0.5
		amount = 0.05
		out = np.copy(image)
		# Salt mode
		num_salt = np.ceil(amount * image.size * s_vs_p)
		coords = [np.random.randint(0, i - 1, int(num_salt))
				  for i in image.shape]
		out[coords] = 1

		# Pepper mode
		num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
		coords = [np.random.randint(0, i - 1, int(num_pepper))
				  for i in image.shape]
		out[coords] = 0
		return out
	elif noise_typ == "poisson":
		vals = len(np.unique(image))
		vals = 2 ** np.ceil(np.log2(vals)) * 0.01
		noisy = np.random.poisson(image * vals) / float(vals)
		return noisy
	elif noise_typ =="speckle":
		row,col = image.shape
		gauss = np.random.randn(row,col)
		gauss = gauss.reshape(row,col)		  
		noisy = image + image * gauss * 0.1
		return noisy
	else:
		return image

def preprocess(imgs):
	imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=imgs.dtype)
	for i in range(imgs.shape[0]):
		imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)
	imgs_p = imgs_p[..., np.newaxis]
	return imgs_p

def create_unagumented_train_data_from_image(img, mask):

	#Normalize inputs.
	img = img.astype('float32')
	img -= 145.68411499 # Calulated from 33 sample images
	img /= 33.8518720956

	#Generate image patches from preprocessed data.
	image_stride_random = image_stride + random.randint(-image_stride_range, image_stride_range)
	patches, nr, nc, nR, nC = extractPatches(img, (image_rows, image_cols), image_stride_random)
	maskPatches, nr, nc, nR, nC = extractPatches(mask, (image_rows, image_cols), image_stride_random)
	patches = preprocess(patches)
	maskPatches = preprocess(maskPatches)

	return patches, maskPatches

def create_train_data_from_image(img, mask, angle, scale, invert_img, mirror_img, noise_type, saving, image_name, image_mask_name):
	#Important: rotating images in this case is important for training - otherwise, degenerates and picks false optimum	
	
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	img = clahe.apply(img)

	#Randomly flip/invert image.
	if invert_img:
		img = np.flipud(img)
		mask = np.flipud(mask)
	if mirror_img:
		img = np.fliplr(img)
		mask = np.fliplr(mask)
	
	#Randomly rotate image.
	img = img.astype('float64')
	mask = mask.astype('float64')
	img = rescale(img, scale, mode='reflect', anti_aliasing=True, preserve_range=True)
	mask = rescale(mask, scale, mode='reflect', anti_aliasing=True, preserve_range=True)
	img = rotate(img, angle, mode='edge', preserve_range=True)
	mask = rotate(mask, angle, mode='edge', preserve_range=True)
	
	#Sharpen image.
	blurred_img = ndimage.gaussian_filter(img, 3)
	filter_blurred_img = ndimage.gaussian_filter(blurred_img, 1)
	alpha = 40
	sharpened = blurred_img + alpha * (blurred_img - filter_blurred_img)
	
	#Add random noise to image.
	img = noisy(noise_type, (sharpened / sharpened.max()))
	img *= 1.0/img.max() #Rescale to fit within -1, 1

	#Calculate edge from mask and dilate.
	mask = cv2.Canny(mask.astype(np.uint8), 100, 200)	
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	mask = cv2.dilate(mask.astype('float64'), kernel, iterations = 1)
	mask = np.where(mask > np.mean(mask), 1.0, 0.0).astype('float32')

	#Save images for verification.
	if saving:
		imsave(os.path.join(temp_path, 'train_full', image_name), img)
		imsave(os.path.join(temp_path, 'train_full', image_mask_name), (mask * 255).astype(np.uint8))

	#Normalize inputs.
	mean = np.mean(img)  # mean for data centering
	std = np.std(img)  # std for data normalization
	img = img.astype('float32')
	img -= mean
	img /= std
#	nanImg = np.isnan(img).any()
#	nanMask = np.isnan(img).any()
#	if nanImg or nanMask:
#		if nanImg:
#			print ('Nan in:', image_name)
#		if nanMask:
#			print ('Nan in mask:', image_mask_name)
#		return None, None

	#Generate image patches from preprocessed data.
	image_stride_random = image_stride + random.randint(-image_stride_range, image_stride_range)
	patches, nr, nc, nR, nC = extractPatches(img, (image_rows, image_cols), image_stride_random)
	maskPatches, nr, nc, nR, nC = extractPatches(mask, (image_rows, image_cols), image_stride_random)
	patches = preprocess(patches)
	maskPatches = preprocess(maskPatches)

	return patches, maskPatches

def create_train_data_from_directory(saveInsteadOfReturn):
	train_data_path = os.path.join(data_path, 'train_full')
	images = os.listdir(train_data_path)
	augmentations = 8
	total = len(images) // 2 * augmentations
	imgs = None
	imgs_mask = None
	i = 0
	noise_type = ['none', 's&p', 'gauss', 'speckle']

	if saveInsteadOfReturn:
		print('-'*30)
		print('Creating training images...')
		print('-'*30)

	for image_name in images:
		if '_mask.png' in image_name or '_bqa.png' in image_name  or '_mtl.txt' in image_name or not os.path.isfile(os.path.join(train_data_path, image_name)):
			continue
		image_mask_name = image_name.split('.')[0] + '_mask.png'
		img = imread(os.path.join(train_data_path, image_name), as_gray=True)
		img_mask = imread(os.path.join(train_data_path, image_mask_name), as_gray=True)
		for augmentation in range(augmentations):
			angle = random.randint(-10, 10)
			scale = random.uniform(0.9, 1.1)
			invert_img = bool(random.getrandbits(1))
			mirror_img = bool(random.getrandbits(1))
			noise = random.choice(noise_type)
			patches, patches_mask = create_train_data_from_image(img, img_mask, angle, scale, invert_img, mirror_img, noise, saveInsteadOfReturn, image_name, image_mask_name)
	
			if (imgs is not None):
				if (imgs.shape[0] != imgs_mask.shape[0]):
					print(image_name, ' mask mismatch')
				else:
					imgs = np.concatenate((imgs, patches))
					imgs_mask = np.concatenate((imgs_mask, patches_mask))
				if saveInsteadOfReturn:
					print(imgs.shape, imgs_mask.shape, image_name)
			else:
				imgs = patches
				imgs_mask = patches_mask
	
			i += 1
			if saveInsteadOfReturn:
				print('Done: {0}/{1} images'.format(i, total))
			if 'ud4_' in image_name:
				i += augmentations - 1
				break

	if saveInsteadOfReturn:
		np.save('landsat_imgs_train_boundaries.npy', imgs)
		np.save('landsat_imgs_mask_train_boundaries.npy', imgs_mask)
		print('Saving to .npy files done.')
	else:
		return imgs, imgs_mask

def load_train_data():
	imgs_train = np.load('landsat_imgs_train_boundaries.npy')
	imgs_mask_train = np.load('landsat_imgs_mask_train_boundaries.npy')
	return imgs_train, imgs_mask_train

def load_validation_data():
	imgs_validation = np.load('landsat_imgs_validation_boundaries.npy')
	imgs_mask_validation = np.load('landsat_imgs_mask_validation_boundaries.npy')
	return (imgs_validation, imgs_mask_validation)

def create_validation_data_from_image(img, mask, image_name, image_mask_name):
	#Important: rotating images in this case is important for training - otherwise, degenerates and picks false optimum	
	
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	img = clahe.apply(img)
	
	#Sharpen image.
	blurred_img = ndimage.gaussian_filter(img, 3)
	filter_blurred_img = ndimage.gaussian_filter(blurred_img, 1)
	alpha = 40
	sharpened = blurred_img + alpha * (blurred_img - filter_blurred_img)

	#Calculate edge from mask and dilate.
	mask = cv2.Canny(mask.astype(np.uint8), 100, 200)	
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	mask = cv2.dilate(mask.astype('float64'), kernel, iterations = 1)
	mask = np.where(mask > np.mean(mask), 1.0, 0.0).astype('float32')

	imsave(os.path.join(temp_path, 'validation_full', image_name), img)
	imsave(os.path.join(temp_path, 'validation_full', image_mask_name), (mask * 255).astype(np.uint8))

	#Normalize inputs.
	mean = np.mean(img)  # mean for data centering
	std = np.std(img)  # std for data normalization
	img = img.astype('float32')
	img -= mean
	img /= std

	#Generate image patches from preprocessed data.
	image_stride_random = 96
	patches, nr, nc, nR, nC = extractPatches(img, (image_rows, image_cols), image_stride_random)
	maskPatches, nr, nc, nR, nC = extractPatches(mask, (image_rows, image_cols), image_stride_random)
	patches = preprocess(patches)
	maskPatches = preprocess(maskPatches)

	return patches, maskPatches
	
def create_validation_data_from_directory():
	train_data_path = os.path.join(data_path, 'validation_full')
	images = os.listdir(train_data_path)
	augmentations = 1
	total = len(images) // 2 * augmentations
	imgs = None
	imgs_mask = None
	i = 0

	print('-'*30)
	print('Creating validation images...')
	print('-'*30)

	for image_name in images:
		if '_mask.png' in image_name or '_bqa.png' in image_name  or '_mtl.txt' in image_name or not os.path.isfile(os.path.join(train_data_path, image_name)):
			continue
		image_mask_name = image_name.split('.')[0] + '_mask.png'
		img = imread(os.path.join(train_data_path, image_name), as_gray=True)
		img_mask = imread(os.path.join(train_data_path, image_mask_name), as_gray=True)
		patches, patches_mask = create_validation_data_from_image(img, img_mask, image_name, image_mask_name)
		
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

	np.save('landsat_imgs_validation_boundaries.npy', imgs)
	np.save('landsat_imgs_mask_validation_boundaries.npy', imgs_mask)

def load_test_data_from_image(img, stride):
	#Adapative histogram equalization/contrast enhancement
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	img = clahe.apply(img)
	
	#Sharpen imaes
	blurred_img = ndimage.gaussian_filter(img, 3)
	filter_blurred_img = ndimage.gaussian_filter(blurred_img, 1)
	alpha = 40
	sharpened = blurred_img + alpha * (blurred_img - filter_blurred_img)
	
	mean = np.mean(img)  # mean for data centering
	std = np.std(img)  # std for data normalization
	img = img.astype('float32')
	img -= mean
	img /= std

	patches, nr, nc, nR, nC = extractPatches(img, (image_rows, image_cols), stride)
	patches = preprocess(patches)

	return patches, nr, nc, nR, nC
		
if __name__ == '__main__':
	#create_validation_data_from_directory()
	create_train_data_from_directory(True)
