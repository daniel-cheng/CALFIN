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
from model import Deeplabv3
from clr_callback import CyclicLR

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from skimage.io import imsave, imread
from skimage.transform import resize, rotate, rescale

from aug_generators import imgaug_generator
from data_landsat_binary_with_boundary_384 import create_unagumented_data_from_image, load_validation_data

img_size = 384
img_rows = img_size
img_cols = img_size
stride = int((img_rows + img_cols) / 2 / 2) #1/2 of img_window square
smooth = 1.
data_path = 'landsat_raw_boundaries/'
pred_path = 'landsat_preds_boundaries/'
temp_path = 'landsat_temp_boundaries/'
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

if __name__ == '__main__':
	print('-'*30)
	print('Loading validation data...')
	print('-'*30)
	validation_data = load_validation_data() 
	
	model_checkpoint = ModelCheckpoint('landsat_weights_with_boundary_' + str(img_size) + '_deeplabv3_xception_aug_e{epoch:02d}_iou{val_iou_score:.4f}.h5', monitor='val_iou_score', save_best_only=False)
#	
#	clr_triangular = CyclicLR(mode='triangular2', step_size=16000, base_lr=6e-5, max_lr=1e-5)
	callbacks_list = [
		#EarlyStopping(patience=4, verbose=1, restore_best_weights=True),
#		clr_triangular,
		model_checkpoint
	]
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
	base_model = Deeplabv3(input_shape=(img_rows,img_cols,3), classes=1, backbone='xception')
	last_linear = base_model(r3)
	out = Activation('sigmoid')(last_linear)

	model = Model(inputs, out)
	model.compile(optimizer=Adam(lr=1e-4), loss=bce_jaccard_loss, metrics=['binary_crossentropy', iou_score, 'accuracy'])
	model.summary()
	model.load_weights('landsat_weights_with_boundary_256_deeplabv3_xception_aug_e22_iou0.8794.h5')
	

	print('-'*30)
	print('Fitting model...')
	print('-'*30)
	train_generator = imgaug_generator(2, img_size)
	next(train_generator)
	history = model.fit_generator(train_generator,
				steps_per_epoch=8000,
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
