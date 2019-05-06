from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import os, cv2, sys, glob
from skimage.io import imsave, imread
from skimage.transform import resize, rotate, rescale
from random import shuffle

from data_landsat_binary_with_boundary import load_train_data, create_unagumented_train_data_from_image, load_test_data_from_image, create_train_data_from_directory, load_validation_data
from scipy.ndimage.filters import median_filter
from transforms import aug_victor

img_rows = 256
img_cols = 256
stride = int((img_rows + img_cols) / 2 / 2) #1/2 of img_window square
smooth = 1.
data_path = 'landsat_raw_boundaries/'
pred_path = 'landsat_preds_boundaries/'
temp_path = 'landsat_temp_boundaries/'

'''
https://github.com/pietz/unet-keras/blob/master/unet.py
U-Net: Convolutional Networks for Biomedical Image Segmentation
(https://arxiv.org/abs/1505.04597)
---
Default settings allow for training within GTX1060 6GB GPU.
img_shape: (height, width, channels)
out_ch: number of output channels
start_ch: number of channels of the first conv
depth: zero indexed depth of the U-structure
inc_rate: rate at which the conv channels will increase
activation: activation function after convolutions
dropout: amount of dropout in the contracting part
batchnorm: adds Batch Normalization if true
maxpool: use strided conv instead of maxpooling if false
upconv: use transposed conv instead of upsamping + conv if false
residual: add residual connections around each conv block if true
'''

def imgaug_generator(batch_size = 4):
	data_path = 'landsat_raw_boundaries'
	temp_path = 'landsat_temp_boundaries/train_full'
	train_data_path = os.path.join(data_path, 'train_raw')
	train_mask_data_path = os.path.join(data_path, 'train_masks')
	images = os.listdir(train_data_path)
	source_counter = 0
	source_limit = len(images)

	augs = aug_victor()
	while True:
		returnCount = 0
		batch_img_patches = None
		batch_mask_patches = None
		batch_img = None
		batch_mask = None

		#Process up to batch_size numbeer of images
		for i in range(batch_size):
			#Load images, resetting source "iterator" when reaching the end
			if source_counter == source_limit:
				shuffle(images)
				source_counter = 0
			image_name = images[source_counter]
			image_mask_name = image_name.split('.')[0] + '_mask.png'
			img = imread(os.path.join(train_data_path, image_name), as_gray=True)
			mask = imread(os.path.join(train_mask_data_path, image_mask_name), as_gray=True)
			source_counter += 1

			#Convert greyscale to RGB greyscale
			img = (img * (255.0 / img.max())).astype(np.uint8)
			img = np.stack((img,)*3, axis=-1)
			mask = np.stack((mask,)*3, axis=-1)

			#Augment image.
			dat = augs(image=img, mask=mask)
			image = dat['image'][:,:,0]
			mask = dat['mask'][:,:,0]
			image = (image / image.max() * 255).astype(np.uint8)

			#Calculate edge from mask and dilate.
			mask = cv2.Canny(mask.astype(np.uint8), 100, 200)	
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
			mask = cv2.dilate(mask.astype('float64'), kernel, iterations = 1)
			mask = np.where(mask > np.mean(mask), 1.0, 0.0).astype('float32')

			imsave(os.path.join(temp_path, image_name[0:-4] + "_" + str(i) + '.png'), image)
			imsave(os.path.join(temp_path, image_mask_name[0:-4] + "_" + str(i) + '.png'), (255 * mask).astype(np.uint8))

			#Create 256xx256 strided patches from image
			patches, maskPatches = create_unagumented_train_data_from_image(image, mask)
			#Add to batches
			if batch_img is not None:
				batch_img = np.concatenate((batch_img, patches))
				batch_mask = np.concatenate((batch_mask, maskPatches))
			else:
				batch_img = patches
				batch_mask = maskPatches

		#Now, return up batch_size number of patches, or generate new ones if exhausting curent patches
		#Shuffle
		idx = np.random.permutation(len(batch_img))
		batch_img = batch_img[idx]
		batch_mask = batch_mask[idx]
		totalPatches = len(batch_img)
		while returnCount + batch_size < totalPatches:
			batch_image_return = batch_img[returnCount:returnCount+batch_size,:,:,:]
			batch_mask_return = batch_mask[returnCount:returnCount+batch_size,:,:,:]
			returnCount += batch_size
			yield (batch_image_return, batch_mask_return)


if __name__ == '__main__':
    train_generator = imgaug_generator(8)
    for i in range(200):
        next(train_generator)
