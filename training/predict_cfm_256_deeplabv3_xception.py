from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model, multi_gpu_model
from keras.models import Model, Input, load_model
from keras.layers import Concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, UpSampling2D, BatchNormalization, RepeatVector, Reshape, Permute, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l1, l2, l1_l2
from keras.activations import relu, sigmoid
from keras.layers import Activation
from keras import backend as K
from tensorflow.python.client import device_lib
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from keras.applications import imagenet_utils
from segmentation_models.losses import bce_jaccard_loss, jaccard_loss, binary_crossentropy
from segmentation_models.metrics import iou_score
import sys
sys.path.insert(0, 'keras-deeplab-v3-plus-master')
from model import Deeplabv3
from clr_callback import CyclicLR

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import cv2, glob
from skimage.io import imsave, imread
from skimage.transform import resize, rotate, rescale
from random import shuffle

from data_cfm_256 import create_unagumented_data_from_image, load_validation_data
from scipy.ndimage.filters import median_filter
from transforms import aug_victor
from composition import Compose, OneOf
from transforms import *
#Make your own data augmentation function!
def aug_daniel(prob=0.8):
	return Compose([
		RandomRotate90(prob=0.5),
		Transpose(prob=0.5),
		Flip(prob=0.5),
		OneOf([
			OneOf([
				IAAAdditiveGaussianNoise(),
				GaussNoise(),
				], prob=0.3),
			OneOf([
				MotionBlur(prob=.2),
				MedianBlur(blur_limit=3, prob=.1),
				Blur(blur_limit=3, prob=.1),
			], prob=0.4),
			OneOf([
				CLAHE(clipLimit=2),
				IAASharpen(),
				IAAEmboss(),
				OneOf([
					RandomContrast(),
					RandomBrightness(),
				])
			], prob=0.6),
			OneOf([
				Distort1(prob=0.3),
				Distort2(prob=0.3),
				IAAPiecewiseAffine(prob=0.4),
				# ElasticTransform(prob=0.3),
			], prob=0.4)
		], prob=0.9),
		HueSaturationValue(prob=0.5)
		], prob=prob)
	
def aug_daniel(prob=0.8):
	return Compose([
		RandomRotate90(prob=0.5),
		Transpose(prob=0.5),
		Flip(prob=0.5),
		OneOf([
			IAAAdditiveGaussianNoise(),
			GaussNoise(),
			Blur(),
		], prob=0.3),
		OneOf([
			CLAHE(clipLimit=2),
			IAASharpen(),
			IAAEmboss(),
			OneOf([
				RandomContrast(),
				RandomBrightness(),
			]),
		], prob=0.6),
		OneOf([
			Distort1(prob=0.3),
			Distort2(prob=0.3),
			IAAPiecewiseAffine(prob=0.4),
			# ElasticTransform(prob=0.3),
		], prob=0.2),
		HueSaturationValue(prob=0.4)
		], prob=prob)
	
def aug_daniel(prob=0.8):
	return Compose([
		RandomRotate90(prob=0.5),
		Transpose(prob=0.5),
		Flip(prob=0.5),
		OneOf([
			IAAAdditiveGaussianNoise(),
			GaussNoise(),
			Blur(),
		], prob=0.3),
		OneOf([
			CLAHE(clipLimit=2),
			IAASharpen(),
			IAAEmboss(),
			OneOf([
				RandomContrast(),
				RandomBrightness(),
			]),
		], prob=0.7),
		HueSaturationValue(prob=0.4)
		], prob=prob)
	
def aug_daniel(prob=0.8):
	return Compose([
		RandomRotate90(prob=0.5),
		Transpose(prob=0.5),
		Flip(prob=0.5),
		OneOf([
			IAAAdditiveGaussianNoise(),
			GaussNoise(),
			Blur(),
		], prob=0.3),
		OneOf([
			CLAHE(clipLimit=2),
			IAASharpen(),
			IAAEmboss(),
			OneOf([
				RandomContrast(),
				RandomBrightness(),
			]),
			Blur(),
			GaussNoise()
		], prob=0.5),
		HueSaturationValue(prob=0.5)
		], prob=prob)

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
	n = Conv2D(dim, 3, activation=acti, dilation_rate=1, padding='same', kernel_regularizer=l1_l2(0.0001, 0.0001))(m)
	n = BatchNormalization()(n) if bn else n
	n = Dropout(do)(n) if do else n
	n = Conv2D(dim, 3, activation=acti, dilation_rate=1, padding='same', kernel_regularizer=l1_l2(0.0001, 0.0001))(n)
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
			m = Conv2D(dim, 2, activation=acti, padding='same', kernel_regularizer=l1_l2(0.0001, 0.0001))(m)
		else:
			m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
		n = Concatenate()([n, m])
		m = conv_block(n, dim, acti, bn, res)
		#m = conv_block(n, dim, acti, bn, res, do)
	else:
		#m = conv_block(m, dim, acti, bn, res)
		m = conv_block(m, dim, acti, bn, res, do)
	return m

def UNetModel(img_shape, out_ch=1, start_ch=48, depth=4, inc_rate=2., activation='relu',
		 dropout=0.4, batchnorm=True, maxpool=True, upconv=True, residual=True):
	i = Input(shape=img_shape)
	o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
	o = Conv2D(out_ch, 1, activation='sigmoid')(o)
	return Model(inputs=i, outputs=o)

def predict(model, image):
	#Load img in as ?X? gray, cast from uint16 to uint8 (0-255)
	patches, _ = create_unagumented_data_from_image(image, None) #np.float32 [-1.0, 1.0] (mean ~0.45), shape = (1, 256, 256, 1) #np.float32 [0.0, 1.0] (mean<0.5), shape = (1, 256, 256, 1)
	#Cast img to float32, perform imagenet preprocesssing (~0.45 means, -[1,1])
	#Resize to 1x256x256x1
	imgs_mask_test = model.predict(patches, batch_size=1, verbose=1)
	full_image = imgs_mask_test[0,:,:,0]
	
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
def preprocess_input(x):
	"""Preprocesses a numpy array encoding a batch of images.
	# Arguments
		x: a 4D numpy array consists of RGB values within [0, 255].
	# Returns
		Input array scaled to [-1.,1.]
	"""
	return imagenet_utils.preprocess_input(x, mode='tf')

			
def imgaug_generator(batch_size = 4):
	train_data_path = 'landsat_raw_boundaries/train_full'
	temp_path = 'landsat_temp_boundaries/train_full'
	images = glob.glob(train_data_path + '/*[0-9].png')
	shuffle(images)
	source_counter = 0
	source_limit = len(images)
	images_per_metabatch = 16
	augs_per_image = 8

	augs = aug_daniel()

	while True:
		returnCount = 0
		batch_img_patches = None
		batch_mask_patches = None
		batch_img = None
		batch_mask = None

		#Process up to 16 images in one batch to maitain randomness
		for i in range(images_per_metabatch):
			#Load images, resetting source "iterator" when reaching the end
			if source_counter == source_limit:
				shuffle(images)
				source_counter = 0
			image_name = images[source_counter].split(os.path.sep)[-1]
			image_mask_name = image_name.split('.')[0] + '_mask.png'
			img = imread(os.path.join(train_data_path, image_name), as_gray=True) #np.uint16 [0, 65535]
			mask = imread(os.path.join(train_data_path, image_mask_name), as_gray=True) #np.uint8 [0, 255]
			img = resize(img, (256, 256), preserve_range=True)  #np.float32 [0.0, 65535.0]
			mask = resize(mask, (256, 256), preserve_range=True) #np.float32 [0.0, 255.0]
	
			source_counter += 1

			#Convert greyscale to RGB greyscale
			img = (img * (255.0 / img.max())).astype(np.uint8)
			img_3 = np.stack((img,)*3, axis=-1)
			mask_3 = np.stack((mask,)*3, axis=-1)
		
			#Run each image through 8 random augmentations per image
			for j in range(augs_per_image):
				#Augment image.
				dat = augs(image=img_3, mask=mask_3)
				img_aug = np.mean(dat['image'], axis=2)
				mask_aug = np.mean(dat['mask'], axis=2)
				img_aug = (img_aug / img_aug.max() * 255).astype(np.uint8)  #np.uint8 [0, 255]
				
				#Calculate edge from mask and dilate.
				mask_edge = cv2.Canny(mask_aug.astype(np.uint8), 100, 200)	
				kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
				mask_edge = cv2.dilate(mask_edge.astype('float64'), kernel, iterations = 1)
				mask_edge = np.where(mask_edge > np.mean(mask_edge), 1.0, 0.0).astype('float32') #np.float32 [0.0, 1.0]
				
				patches, maskPatches = create_unagumented_data_from_image(img_aug, mask_edge)
				
#				imsave(os.path.join(temp_path, image_name.split('.')[0] + "_" + str(j) + '.png'), np.round((patches[0,:,:,0]+1)/2*255).astype(np.uint8))
#				imsave(os.path.join(temp_path, image_name.split('.')[0] + "_" + str(j) + '_mask.png'), (255 * maskPatches[0,:,:,0]).astype(np.uint8))
		
				#Add to batches
				if batch_img is not None:
					batch_img = np.concatenate((batch_img, patches)) #np.float32 [-1.0, 1.0], imagenet mean (~0.45)
					batch_mask = np.concatenate((batch_mask, maskPatches))  #np.float32 [0.0, 1.0]
				else:
					batch_img = patches
					batch_mask = maskPatches

		#Should have total of augs_per_image * images_per_metabatch to randomly choose from
		totalPatches = len(batch_img)
		#Now, return up batch_size number of patches, or generate new ones if exhausting curent patches
		#Shuffle
		idx = np.random.permutation(len(batch_img))
		if (len(batch_img) is not len(batch_mask)):
			#print('batch img/mask mismatch!')
			continue
		batch_img = batch_img[idx]
		batch_mask = batch_mask[idx]
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
	
	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)
	img_shape = (img_rows,img_cols,1)
	flatten_shape = (img_rows*img_cols,)
	target_shape = (img_rows,img_cols,3)
	inputs = Input(shape=img_shape)
	r1 = Reshape(flatten_shape)(inputs)
	r2 = RepeatVector(3)(r1)
	r3 = Reshape(target_shape)(r2)
	base_model = Deeplabv3(input_shape=(256,256,3), classes=1, backbone='xception')
	last_linear = base_model(r3)
	out = Activation('sigmoid')(last_linear)
	model = Model(inputs, out)
	SMOOTH = 1e-12
	def bce_jaccard_loss(gt, pr, bce_weight=1.0, smooth=SMOOTH, per_image=True):
		bce = K.mean(binary_crossentropy(gt, pr))
		loss = bce_weight * bce + jaccard_loss(gt, pr, smooth=smooth, per_image=per_image)
		return loss
	
	model.compile(optimizer=Adam(lr=1e-5), loss=bce_jaccard_loss, metrics=['binary_crossentropy', iou_score, 'accuracy'])
	model.summary()
#		model.load_weights('landsat_weights_with_boundary_256_deeplabv3_xception_e17_iou0.9695.h5')
#		model.load_weights('landsat_weights_with_boundary_256_deeplabv3_xception_e20_iou0.8794.h5')
			
	print('-'*30)
	print('Predicting masks on test data...')
	print('-'*30)
	pred_path = 'landsat_preds_boundaries/test_full_256'
	test_path = 'landsat_raw_boundaries_all/test_full/'

	for path in glob.glob(test_path + '/*[0-9].png'):
		domain = path.split(os.path.sep)[-1].split('_')[0]
		source_domain_path = os.path.join(test_path)
		output_domain_path = os.path.join(pred_path, domain)
		if not os.path.exists(output_domain_path):
			os.mkdir(output_domain_path)
		name = path.split(os.path.sep)[-1][0:-4]
		img = imread(path, as_gray = True)
		img = resize(img, (256, 256), preserve_range=True)  #np.float32 [0.0, 65535.0]
		
		#Convert greyscale to RGB greyscale
		img = (img * (255.0 / img.max())).astype(np.uint8) #np.uint8 [0, 255]
		
		sum_result = None
		for angle in [0, 90, 180, 270]:
#			for angle in [0]:
			rot_img = rotate(img, angle, mode='reflect', preserve_range=True).astype(np.uint8)
			result = predict(model, rot_img)
			result = rotate(result, -angle, mode='reflect', preserve_range=True)
			
			if sum_result is None:
				sum_result = result / 4.0
			else:
				sum_result += result / 4.0
				
		imsave(os.path.join(output_domain_path, name + '_raw.png'), img)
		imsave(os.path.join(output_domain_path, name + '_pred.png'), (sum_result * 255.0).astype(np.uint8))
