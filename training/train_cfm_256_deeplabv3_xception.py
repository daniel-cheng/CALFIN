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
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import cv2, glob
from skimage.io import imsave, imread
from skimage.transform import resize, rotate, rescale
from random import shuffle

from data_cfm_256 import load_validation_data
from albumentations import *
from aug_generators import aug_daniel

img_size = 256
data_path = 'data/'
pred_path = 'preds/'
temp_path = 'temp/'
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

if __name__ == '__main__':
	print('-'*30)
	print('Loading validation data...')
	print('-'*30)
	validation_data = load_validation_data(img_size) 
	
	model_checkpoint = ModelCheckpoint('cfm_weights_' + str(img_size) + '_e{epoch:02d}_iou{val_iou_score:.4f}.h5', monitor='val_iou_score', save_best_only=False)
	clr_triangular = CyclicLR(mode='triangular2', step_size=4000, base_lr=6e-4, max_lr=6e-5)
	callbacks_list = [
		EarlyStopping(patience=6, verbose=1, restore_best_weights=True),
#		clr_triangular,
		model_checkpoint
	]
	
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
	base_model = Deeplabv3(input_shape=(img_size, img_size,3), classes=1, backbone='xception')
	last_linear = base_model(r3)
	out = Activation('sigmoid')(last_linear)
	
	model = Model(inputs, out)
	model.compile(optimizer=AdamAccumulate(lr=1e-4, accum_iters=16), loss=bce_jaccard_loss, metrics=['binary_crossentropy', iou_score, 'accuracy'])
	model.summary()
	#model.load_weights('cfm_weights_e10_iou0.8776.h5')
	
	print('-'*30)
	print('Fitting model...')
	print('-'*30)
	train_generator = imgaug_generator(1, img_size)
	history = model.fit_generator(train_generator,
				steps_per_epoch=16000,
				epochs=40,
				validation_data=validation_data,
				verbose=1,
				max_queue_size=64,
				use_multiprocessing=True,
				workers=2,
				callbacks=callbacks_list)
	print(history.history)
