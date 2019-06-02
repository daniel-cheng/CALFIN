from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model, multi_gpu_model
from keras.models import Model, Input, load_model
from keras.layers import Concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, UpSampling2D, BatchNormalization, RepeatVector, Reshape
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l1, l2
from keras.activations import relu
from keras import backend as K
from tensorflow.python.client import device_lib

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import os, cv2, sys, glob
from skimage.io import imsave, imread
from skimage.transform import resize, rotate, rescale
from random import shuffle

from data_landsat_binary_with_boundary import load_train_data, create_unagumented_train_data_from_image, load_test_data_from_image, create_train_data_from_directory, load_validation_data
from scipy.ndimage.filters import median_filter
from transforms import aug_victor

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

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

def dice_coef(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
	return -dice_coef(y_true, y_pred)

import keras.metrics
keras.metrics.dice_coef = dice_coef

import keras.losses
keras.losses.dice_coef_loss = dice_coef_loss

def conv_block(m, dim, acti, bn, res, do=0):
	n = Conv2D(dim, 3, activation=acti, dilation_rate=1, padding='same', kernel_regularizer=l2(0.000005))(m)
	n = BatchNormalization()(n) if bn else n
	n = Dropout(do)(n) if do else n
	n = Conv2D(dim, 3, activation=acti, dilation_rate=1, padding='same', kernel_regularizer=l2(0.000005))(n)
	n = BatchNormalization()(n) if bn else n
	return Concatenate()([m, n]) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
	if depth > 0:
		#n = conv_block(m, dim, acti, bn, res, do)
		n = conv_block(m, dim, acti, bn, res)
		m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
		m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
		if up:
			m = UpSampling2D()(m)
			m = Conv2D(dim, 2, activation=acti, padding='same', kernel_regularizer=l2(0.000005))(m)
		else:
			m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
		n = Concatenate()([n, m])
		m = conv_block(n, dim, acti, bn, res)
		#m = conv_block(n, dim, acti, bn, res, do)
	else:
		m = conv_block(m, dim, acti, bn, res)
		#m = conv_block(m, dim, acti, bn, res, do)
	return m

def UNet(img_shape, out_ch=1, start_ch=48, depth=4, inc_rate=2., activation='relu',
		 dropout=0.0, batchnorm=True, maxpool=True, upconv=True, residual=True):
	i = Input(shape=img_shape)
	o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
	o = Conv2D(out_ch, 1, activation='sigmoid')(o)
	return Model(inputs=i, outputs=o)

def predict(model, image):
	imgs_test, nr, nc, nR, nC = load_test_data_from_image(image, stride)

	imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)

	image_cols = []
	stride2 = int(stride/2)

	for i in range(0, nR):
		image_row = []
		for j in range(0, nC):
			index = i * nC + j
			image = (imgs_mask_test[index, :, :, 0] * 255.).astype(np.uint8)
			image = image[stride2:img_rows - stride2 - 1, stride2:img_cols - stride2 - 1]
			image_row.append(image)
		image_cols.append(np.hstack(image_row))
	full_image = np.vstack(image_cols)
	full_image = full_image[0:nr, 0:nc].astype(np.uint8)
	return full_image

def resolve(name, basepath=None):
	if not basepath:
		basepath = os.path.dirname(os.path.realpath(__file__))
	return os.path.join(basepath, name)

def mix_data(patches, mask_patches, patches2, mask_patches2):
	"""Randomly mix two pairs of arrays together in same way while keeping same size as patches"""
	patches_all = np.concatenate((patches, patches2))
	mask_patches_all = np.concatenate((mask_patches, mask_patches2))

	idx = np.random.choice(np.arange(len(patches_all)), len(patches), replace=False)
	return patches_all[idx], mask_patches_all[idx]

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
		if (len(batch_img) is not len(batch_mask)):
			#print('batch img/mask mismatch!')
			continue
		batch_img = batch_img[idx]
		batch_mask = batch_mask[idx]
		totalPatches = len(batch_img)
		while returnCount + batch_size < totalPatches:
			batch_image_return = batch_img[returnCount:returnCount+batch_size,:,:,:]
			batch_mask_return = batch_mask[returnCount:returnCount+batch_size,:,:,:]
			returnCount += batch_size
			yield (batch_image_return, batch_mask_return)


if __name__ == '__main__':
	print('-'*30)
	print('Loading validation data...')
	print('-'*30)
	validation_data = load_validation_data() 
	
	hyperparameters = [32, 4, 2]
	hyperparameters_string = '-'.join(str(x) for x in hyperparameters)
	model_checkpoint = ModelCheckpoint('landsat_weights_with_boundary_256_' + hyperparameters_string + '_{val_dice_coef:.5f}.h5', monitor='val_dice_coef_loss', save_best_only=False)
	callbacks_list = [
		#EarlyStopping(patience=6, verbose=1, restore_best_weights=True),
		model_checkpoint
	]
	try:
		raise Error()
		print('-'*30)
		print('Loading model...')
		print('-'*30)
		model = load_model('landsat_weights_with_boundary_256_' + hyperparameters_string + '.h5')
	#	model = multi_gpu_model(model)
	#	print('-'*30)
	#	print('Loading saved weights...')
	#	print('-'*30)
		model.load_weights('landsat_weights_with_boundary_256_32-4-2_0.38461.h5')
		#origin_model = model.layers[-2] 
		#origin_model.save_weights('landsat_weights_with_boundary_singleGPU_256_16-3-1.5_-0.63244.h5')
	except:
		print('-'*30)
		print('Creating and compiling model...')
		print('-'*30)
		model = UNet((img_rows,img_cols,1), start_ch=hyperparameters[0], depth=hyperparameters[1], inc_rate=hyperparameters[2], activation='elu')
		model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=['binary_crossentropy', dice_coef])
		model.load_weights('landsat_weights_with_boundary_256_32-4-2_0.38022.h5')

	print('-'*30)
	print('Fitting model...')
	print('-'*30)
	#train_generator = imgaug_generator(4)
	#h = model.fit_generator(train_generator,
	#			steps_per_epoch=1500,
	#			epochs=50,
	#			validation_data=validation_data,
	#			verbose=1,
	#			use_multiprocessing=True,
	#			callbacks=callbacks_list)

	model.save('landsat_weights_with_boundary_256_' + hyperparameters_string + '.h5')
	model.summary()
	print('layer count:', len(model.layers))

	print('-'*30)
	print('Predicting masks on test data...')
	print('-'*30)
	validation_path = os.path.join(data_path, 'test_full')
	for name in os.listdir(validation_path):
		if '_mask' not in name and '_pred' not in name and '_bqa' not in name and '_mtl' not in name:
			path = os.path.join(validation_path, name)
			if os.path.isfile(path):
				image = imread(path, as_gray = True)
				#image_mask_name = name.split('.')[0] + '_mask.png'
				#mask = imread(os.path.join(validation_path, image_mask_name), as_gray=True)
				image = (image * (255.0 / image.max())).astype(np.uint8)
				sum_result = None
				for angle in [0, 90, 180, 270]:
					rot_img = rotate(image, angle, mode='reflect', preserve_range=True).astype(np.uint8)
					result = predict(model, rot_img).astype('float32')
					result = rotate(result, -angle, mode='reflect', preserve_range=True)
					if sum_result is None:
						sum_result = result / 4.0
					else:
						sum_result += result / 4.0

				imsave(os.path.join(pred_path, name[0:-4] + '_raw.png'), image)
				#imsave(os.path.join(pred_path, name[0:-4] + '_mask.png'), mask)
				imsave(os.path.join(pred_path, name[0:-4] + '_pred.png'), sum_result.astype(np.uint8))
