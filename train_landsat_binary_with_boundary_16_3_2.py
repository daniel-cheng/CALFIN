from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
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
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import os, cv2, sys, glob
from skimage.io import imsave, imread

from data_landsat_binary_with_boundary_224 import load_train_data, load_test_data_from_image, create_train_data_from_directory, load_validation_data
from scipy.ndimage.filters import median_filter

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 224
img_cols = 224
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
	n = Conv2D(dim, 3, activation=acti, dilation_rate=1, padding='same', kernel_regularizer=l2(0.0001))(m)
	n = BatchNormalization()(n) if bn else n
	n = Dropout(do)(n) if do else n
	n = Conv2D(dim, 3, activation=acti, dilation_rate=1, padding='same', kernel_regularizer=l2(0.0001))(n)
	n = BatchNormalization()(n) if bn else n
	return Concatenate()([m, n]) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
	if depth > 0:
		n = conv_block(m, dim, acti, bn, res, 0.0)
		m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
		m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
		if up:
			m = UpSampling2D()(m)
			m = Conv2D(dim, 2, activation=acti, padding='same', kernel_regularizer=l2(0.0001))(m)
		else:
			m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
		n = Concatenate()([n, m])
		m = conv_block(n, dim, acti, bn, res, 0.2)
	else:
		m = conv_block(m, dim, acti, bn, res, do)
	return m

def UNet(img_shape, out_ch=1, start_ch=48, depth=4, inc_rate=2., activation='relu',
		 dropout=0.5, batchnorm=True, maxpool=True, upconv=True, residual=True):
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
#	full_image = median_filter(full_image, 7)
#	full_image = np.where(full_image > 127, 255, 0)
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

if __name__ == '__main__':
	print('-'*30)
	print('Loading validation data...')
	print('-'*30)
	validation_data = load_validation_data() 
	
	hyperparameters = [16, 3, 2]
	hyperparameters_string = '-'.join(str(x) for x in hyperparameters)
	model_checkpoint = ModelCheckpoint('landsat_weights_with_boundary_224_' + hyperparameters_string + '_{val_dice_coef:.5f}.h5', monitor='val_binary_crossentropy', save_best_only=False)
	#try:
		#print('-'*30)
		#print('Loading model...')
		#print('-'*30)
		#model = load_model('landsat_weights_with_boundary_256_' + hyperparameters_string + '.h5')
	#	model = multi_gpu_model(model)
	#	print('-'*30)
	#	print('Loading saved weights...')
	#	print('-'*30):
	#	model.load_weights('landsat_weights_with_boundary_multiGPU_256_16-4-2_-0.32962.h5')
		#origin_model = model.layers[-2] 
		#origin_model.save_weights('landsat_weights_with_boundary_singleGPU_256_16-3-1.5_-0.63244.h5')
	#except:
	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)
	model = UNet((img_rows,img_cols,1), start_ch=hyperparameters[0], depth=hyperparameters[1], inc_rate=hyperparameters[2], activation='elu')
	model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=['binary_crossentropy', dice_coef])

	print('-'*30)
	print('Loading saved weights...')
	print('-'*30)
	#model.load_weights('landsat_weights_with_boundary_224_32-3-2_0.76759.h5')

	print('-'*30)
	print('Fitting model...')
	print('-'*30)
	callbacks_list = [
		EarlyStopping(patience=4, verbose=1, restore_best_weights=True),
		#ReduceLROnPlateau(patience=5, verbose=1),
		model_checkpoint
	]

#	print('-'*30)
#	print('Predicting masks on test data...')
#	print('-'*30)
#	for name in os.listdir(os.path.join(data_path, 'test_full')):
#		if '_mask' not in name and '_pred' not in name and '_bqa' not in name and '_mtl' not in name:
#			path = os.path.join(data_path, 'test_full', name)
#			if os.path.isfile(path):
#				image = imread(path, as_gray = True)
#				result = predict(model, image)
#				imsave(os.path.join(pred_path, name[0:-4] + '_raw.png'), image)
	#			imsave(os.path.join(pred_path, name[0:-4] + '_pred.png'), result)

	imgs_train, imgs_mask_train = load_train_data()
	for i in range(20):
		imgs_train_1, imgs_mask_train_1 = create_train_data_from_directory(False)
		imgs_train, imgs_mask_train = mix_data(imgs_train, imgs_mask_train, imgs_train_1, imgs_mask_train_1)
		history = model.fit(imgs_train, imgs_mask_train,
					batch_size=2,
					epochs=15,
					verbose=1,
					shuffle=True,
					validation_split=0.00,
					validation_data=validation_data,
					callbacks=callbacks_list)
		np.save('landsat_imgs_train_boundaries_224.npy', imgs_train)
		np.save('landsat_imgs_mask_train_boundaries_224.npy', imgs_mask_train)
		model.save('landsat_weights_with_boundary_224_' + hyperparameters_string + '.h5')
#	# Plot training & validation accuracy values
#	plt.plot(history.history['dice_coef'])
#	plt.plot(history.history['val_dice_coef'])
#	plt.title('Model accuracy')
#	plt.ylabel('Accuracy')
#	plt.xlabel('Epoch')
#	plt.legend(['Train', 'Test'], loc='upper left')
#	plt.show()
#
#	# Plot training & validation loss values
#	plt.plot(history.history['loss'])
#	plt.plot(history.history['val_loss'])
#	plt.title('Model loss')
#	plt.ylabel('Loss')
#	plt.xlabel('Epoch')
#	plt.legend(['Train', 'Test'], loc='upper left')
#	plt.show()
#
#	plot_model(model, to_file='model_256_' + hyperparameters_string + '.png')

	model.summary()
	#print('-'*30)
	#print('Predicting masks on test data...')
	#print('-'*30)
	#for name in os.listdir(os.path.join(data_path, 'test_full')):
	#	if '_mask' not in name and '_pred' not in name and '_bqa' not in name and '_mtl' not in name:
	#		path = os.path.join(data_path, 'test_full', name)
	#		if os.path.isfile(path):
	#			image = imread(path, as_gray = True)
	#			result = predict(model, image)
	#			imsave(os.path.join(pred_path, name[0:-4] + '_raw.png'), image)
	#			imsave(os.path.join(pred_path, name[0:-4] + '_pred.png'), result)
