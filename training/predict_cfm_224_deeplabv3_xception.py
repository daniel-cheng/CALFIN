from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.models import Model, Input, load_model
from keras.layers import Concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, UpSampling2D, BatchNormalization, RepeatVector, Reshape, Permute, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l1, l2, l1_l2
from keras.activations import relu, sigmoid
from keras.layers import Activation
from keras import backend as K
from tensorflow.python.client import device_lib
from keras.applications import imagenet_utils
from segmentation_models.losses import bce_jaccard_loss, jaccard_loss, binary_crossentropy
from segmentation_models.metrics import iou_score

import sys
sys.path.insert(0, 'keras-deeplab-v3-plus')
from model import Deeplabv3
from clr_callback import CyclicLR
from AdamAccumulate import AdamAccumulate

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import cv2, glob
from skimage.io import imsave, imread
from skimage.transform import resize, rotate, rescale
from random import shuffle

from data_cfm_224 import load_validation_data
from albumentations import *
from aug_generators import aug_daniel, imgaug_generator, create_unagumented_data_from_image

img_size = 224
data_path = 'data/'
pred_path = 'preds/'
temp_path = 'temp/'
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def predict(model, image):
	#Load img in as ?X? gray, cast from uint16 to uint8 (0-255)
	#Cast img to float32, perform imagenet preprocesssing (~0.45 means, -[1,1])
	patches, _ = create_unagumented_data_from_image(image, None) #np.float32 [-1.0, 1.0] (mean ~0.45), shape = (1, 224, 224, 1) #np.float32 [0.0, 1.0] (mean<0.5), shape = (1, 224, 224, 1)
	

	imgs_mask_test = model.predict(patches, batch_size=1, verbose=1)
	full_image = imgs_mask_test[0,:,:,0]
	
	return full_image

if __name__ == '__main__':
	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)
	img_shape = (img_size, img_size, 1)
	flatten_shape = (img_size * img_size,)
	target_shape = (img_size, img_size, 3)
	inputs = Input(shape=img_shape)
	r1 = Reshape(flatten_shape)(inputs)
	r2 = RepeatVector(3)(r1)
	r3 = Reshape(target_shape)(r2)
	base_model = Deeplabv3(input_shape=(img_size, img_size,3), classes=1, OS=16, backbone='xception')
	last_linear = base_model(r3)
	out = Activation('sigmoid')(last_linear)
	
	model = Model(inputs, out)
	model.compile(optimizer=AdamAccumulate(lr=1e-4, accum_iters=4), loss=bce_jaccard_loss, metrics=['binary_crossentropy', iou_score, 'accuracy'])
	model.summary()
	model.load_weights('cfm_weights_224_e20_iou0.5660.h5')

	print('-'*30)
	print('Predicting masks on test data...')
	print('-'*30)
	pred_path = r'D:\Daniel\Documents\Github\CALFIN Repo\processing\landsat_preds_core'
	test_path = r'D:\Daniel\Documents\Github\CALFIN Repo\processing\landsat_raw_core'

	for domain in os.listdir(test_path):
		source_domain_path = os.path.join(test_path, domain)
		output_domain_path = os.path.join(pred_path, domain)
		if not os.path.exists(output_domain_path):
			os.mkdir(output_domain_path)
		for path in glob.glob(source_domain_path + '\\*'):
			name = path.split(os.path.sep)[-1][0:-4]
			img = imread(path, as_gray = True)
			img = resize(img, (img_size, img_size), preserve_range=True)  #np.float64 [0.0, 65535.0]
			
			#Convert greyscale to RGB greyscale
			img = (img * (255.0 / img.max())).astype(np.uint8) #np.uint8 [0, 255]
			img_max = img.max()
			if (img_max != 0.0):
				img = np.round(img / img_max * 255.0).astype(np.uint8) #np.uint8 [0, 255]
			
			sum_result = None
			for angle in [0, 90, 180, 270]:
				rot_img = rotate(img, angle, mode='reflect', preserve_range=True)
				result = predict(model, rot_img)
				result = rotate(result, -angle, mode='reflect', preserve_range=True)
				
				if sum_result is None:
					sum_result = result / 4.0
				else:
					sum_result += result / 4.0
					
			imsave(os.path.join(output_domain_path, name + '_raw.png'), img)
			imsave(os.path.join(output_domain_path, name + '_pred.png'), (sum_result * 255.0).astype(np.uint8))