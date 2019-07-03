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
data_path = 'data/'
pred_path = 'preds/'
temp_path = 'temp/'

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
