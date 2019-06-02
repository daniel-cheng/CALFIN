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
from keras.applications import imagenet_utils
from segmentation_models.losses import bce_jaccard_loss, jaccard_loss, binary_crossentropy
from segmentation_models.metrics import iou_score

import sys
sys.path.insert(0, 'keras-deeplab-v3-plus-master')
from model import Deeplabv3, BilinearUpsampling
from clr_callback import CyclicLR
from AdamAccumulate import AdamAccumulate

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from skimage.io import imsave, imread
from skimage.transform import resize, rotate, rescale

from aug_generators import imgaug_generator
from data_landsat_binary_with_boundary_512 import create_unagumented_data_from_image, load_validation_data

img_size = 512
img_rows = img_size
img_cols = img_size
stride = int((img_rows + img_cols) / 2 / 2) #1/2 of img_window square
smooth = 1.
data_path = 'landsat_raw_boundaries/'
pred_path = 'landsat_preds_boundaries/'
temp_path = 'landsat_temp_boundaries/'
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def conv_block(m, dim=3, acti='relu', bn=True, res=True, do=0):
	n = Conv2D(dim, 3, activation=acti, dilation_rate=1, padding='same', kernel_regularizer=l2(0.000001))(m)
	n = BatchNormalization()(n) if bn else n
	n = Dropout(do)(n) if do else n
	n = Conv2D(dim, 3, activation=acti, dilation_rate=1, padding='same', kernel_regularizer=l2(0.000001))(n)
	n = BatchNormalization()(n) if bn else n
	return Concatenate()([m, n]) if res else n

if __name__ == '__main__':
	print('-'*30)
	print('Loading validation data...')
	print('-'*30)
	validation_data = load_validation_data() 
	
	hyperparameters = [32, 3, 2]
	hyperparameters_string = '-'.join(str(x) for x in hyperparameters)
	model_checkpoint = ModelCheckpoint('landsat_weights_with_boundary_' + str(img_size) + '_deeplabv3_xception_aug_' + str(int(img_size/2)) + '_block_e{epoch:02d}_iou{val_iou_score:.4f}.h5', monitor='val_iou_score', save_best_only=False)
	
	clr_triangular = CyclicLR(mode='triangular', step_size=16000, base_lr=6e-5, max_lr=4e-4)
	callbacks_list = [
		EarlyStopping(patience=8, verbose=1, restore_best_weights=True),
		clr_triangular,
		model_checkpoint
	]
	
	#make a 1xX encoder and a Xx1 decoder with maxpool
	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)
	img_shape = (img_rows, img_cols, 1)
	img_input = Input(shape=img_shape)
	
	#Input to deeplabv3+ at 256x256x3
	img_input_3 = Conv2D(3, (3, 3), strides=(2, 2),
                   name='upper_entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
	img_input_3 = BatchNormalization(name='upper_entry_flow_conv1_1_BN')(img_input_3)
	img_input_3 = Activation('relu')(img_input_3)
	
	#Input to skip connection/upscaling at 512x512x32
	img_input_32 = Conv2D(32, (3, 3), strides=(1, 1),
                   name='upper_entry_flow_conv2_1', use_bias=False, padding='same')(img_input)
	img_input_32 = BatchNormalization(name='upper_entry_flow_conv2_1_BN')(img_input_32)
	img_input_32 = Activation('relu')(img_input_32)
	
	base_img_shape = (int(img_rows/2), int(img_cols/2), 3)
	base_model_half_res = Deeplabv3(input_shape=base_img_shape, classes=1, backbone='xception')
#	base_model_half_res.load_weights('landsat_weights_with_boundary_256_deeplabv3_xception_aug_e39_iou0.8986.h5')
	segmentation_half_res = base_model_half_res(img_input_3)
	segmentation_full_res = BilinearUpsampling(output_size=(img_shape[0], img_shape[1]))(segmentation_half_res)
	output = Concatenate()([img_input_32, segmentation_full_res])
	output = Conv2D(1, 1, activation='sigmoid')(output)
#
	model = Model(img_input, output)
	model.compile(optimizer=AdamAccumulate(lr=1e-4, accum_iters=6), loss=bce_jaccard_loss, metrics=['binary_crossentropy', iou_score, 'accuracy'])
	model.load_weights('landsat_weights_with_boundary_512_deeplabv3_xception_aug_512_block_e20_iou0.6861.h5')
	model.summary()

	

	print('-'*30)
	print('Fitting model...')
	print('-'*30)
	train_generator = imgaug_generator(2, img_size)
	next(train_generator)
	history = model.fit_generator(train_generator,
				steps_per_epoch=4000,
				epochs=40,
				validation_data=validation_data,
				verbose=1,
				max_queue_size=32,
	#			use_multiprocessing=True,
	#			workers=4,
				callbacks=callbacks_list)
	print(history.history)

	#plt.figure(1)
	#plt.plot(history.history['acc'])
	#plt.plot(history.history['val_acc'])
	#plt.title('model accuracy')
	#plt.ylabel('accuracy')
	#plt.xlabel('epoch')
	#plt.legend(['train', 'test'], loc='upper left')
	#plt.show()

	## summarize history for loss
	#plt.figure(2)
	#plt.plot(history.history['loss'])
	#plt.plot(history.history['val_loss'])
	#plt.title('model loss')
	#plt.ylabel('loss')
	#plt.xlabel('epoch')
	#plt.legend(['train', 'test'], loc='upper left')
	#plt.show()


	## summarize history for loss
	#plt.figure(3)
	#plt.plot(history.history['iou_score'])
	#plt.plot(history.history['val_iou_score'])
	#plt.title('model iou_score')
	#plt.ylabel('iou_score')
	#plt.xlabel('epoch')
	#plt.legend(['train', 'test'], loc='upper left')
	#plt.show()
