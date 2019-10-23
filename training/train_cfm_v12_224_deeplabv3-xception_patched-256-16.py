import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Input
from keras.layers import Concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, UpSampling2D, BatchNormalization, RepeatVector, Reshape, Permute, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.activations import relu, sigmoid
from keras.regularizers import l1, l2, l1_l2
from keras.layers import Activation
from keras import backend as K
from segmentation_models.losses import bce_jaccard_loss, jaccard_loss, binary_crossentropy
from segmentation_models.metrics import iou_score, jaccard_score

import sys
sys.path.insert(0, 'keras-deeplab-v3-plus')
from model_cfm_dual_wide_deep_x65 import Deeplabv3, _xception_block
from clr_callback import CyclicLR
from AdamAccumulate import AdamAccumulate
from AdditionalValidationSets import AdditionalValidationSets

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import cv2, glob
from skimage.io import imsave, imread
from skimage.transform import resize, rotate, rescale
from random import shuffle

from data_cfm_patched_dual import load_validation_data
from albumentations import *
from aug_generators_dual import aug_daniel, imgaug_generator_patched

full_size = 256
img_size = 224
stride = 16
data_path = 'data/'
pred_path = 'preds/'
temp_path = 'temp/'
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

if __name__ == '__main__':
	print('-'*30)
	print('Loading validation data...')
	print('-'*30)
	validation_data, validation_targets = load_validation_data(full_size, img_size, stride) 
	
	model_checkpoint = ModelCheckpoint('cfm_weights_patched_dual_wide_x65_' + str(img_size) + '_e{epoch:02d}_iou{val_iou_score:.4f}.h5', monitor='val_iou_score', save_best_only=False)
	clr_triangular = CyclicLR(mode='triangular2', step_size=12000, base_lr=6e-5, max_lr=6e-4)
	additional = AdditionalValidationSets([(validation_data[1], validation_targets[1], 'Upernavik'), (validation_data[2], validation_targets[2], 'Jakobshavn'), (validation_data[3], validation_targets[3], 'Kong-Oscar'), (validation_data[4], validation_targets[4], 'Kangiata-Nunaata'), (validation_data[5], validation_targets[5], 'Hayes'), (validation_data[6], validation_targets[6], 'Rink-Isbrae'), (validation_data[7], validation_targets[7], 'Kangerlussuaq'), (validation_data[8], validation_targets[8], 'Helheim')])
	callbacks_list = [
		#EarlyStopping(patience=6, verbose=1, restore_best_weights=False),
		#clr_triangular,
		additional,
		model_checkpoint
	]
	
	SMOOTH = 1e-12
	SMOOTH2 = 1e-1
	def bce_ln_jaccard_loss(gt, pr, bce_weight=1.0, smooth=SMOOTH, per_image=True):
		bce = K.mean(binary_crossentropy(gt[:,:,:,0], pr[:,:,:,0]))*25/26 + K.mean(binary_crossentropy(gt[:,:,:,1], pr[:,:,:,1]))/26
		loss = bce_weight * bce - K.log(jaccard_score(gt[:,:,:,0], pr[:,:,:,0], smooth=smooth, per_image=per_image))*25/26 - K.log(jaccard_score(gt[:,:,:,1], pr[:,:,:,1], smooth=smooth, per_image=per_image))/26
		return loss
	
	def iou_score(gt, pr, smooth=SMOOTH, per_image=True):
		edge_iou_score = jaccard_score(gt[:,:,:,0], pr[:,:,:,0], smooth=smooth, per_image=per_image)
		mask_iou_score = jaccard_score(gt[:,:,:,1], pr[:,:,:,1], smooth=smooth, per_image=per_image)
		return (edge_iou_score * 25 + mask_iou_score)/26

	def edge_iou_score(gt, pr, smooth=SMOOTH, per_image=True):
		edge_iou_score = jaccard_score(gt[:,:,:,0], pr[:,:,:,0], smooth=smooth, per_image=per_image)
		return edge_iou_score

	def mask_iou_score(gt, pr, smooth=SMOOTH, per_image=True):
		mask_iou_score = jaccard_score(gt[:,:,:,1], pr[:,:,:,1], smooth=smooth, per_image=per_image)
		return mask_iou_score

	def deviation(gt, pr, smooth=SMOOTH2, per_image=True):
		mismatch = K.sum(K.abs(gt[:,:,:,1] - pr[:,:,:,1]), axis=[1, 2]) #(B)
		length = K.sum(gt[:,:,:,0], axis=[1, 2]) #(B)
		deviation = mismatch / (length + smooth) #(B)
		mean_deviation = K.mean(deviation) / 3.0 #- (account for line thickness of 3 at 224)
		return mean_deviation

	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)
	img_shape = (img_size, img_size, 3)
	inputs = Input(shape=img_shape)
	model = Deeplabv3(input_shape=(img_size, img_size,3), classes=16, OS=16, backbone='xception', weights=None)
	
	model.compile(optimizer=AdamAccumulate(lr=1e-4, accum_iters=4), loss=bce_ln_jaccard_loss, metrics=['binary_crossentropy', iou_score, edge_iou_score, mask_iou_score, deviation])
	model.summary()
	#model.load_weights('cfm_weights_patched_dual_wide_x71224_e02_iou0.3631.h5')
	
	print('-'*30)
	print('Fitting model...')
	print('-'*30)
	steps_per_epoch = 8000
	train_generator = imgaug_generator_patched(4, img_size=full_size, patch_size=img_size, patch_stride=stride, steps_per_epoch=steps_per_epoch)
	history = model.fit_generator(train_generator,
				steps_per_epoch=steps_per_epoch,
				epochs=80,
				validation_data=(validation_data[0], validation_targets[0]),
				verbose=1,
#				max_queue_size=64,
#				use_multiprocessing=True,
#				workers=2,
				callbacks=callbacks_list)
	print(history.history)
