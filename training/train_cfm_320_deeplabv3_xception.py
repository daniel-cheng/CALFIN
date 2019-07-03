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

from data_cfm_320 import create_unagumented_data_from_image, load_validation_data
from albumentations import *

img_size = 320
data_path = 'data/'
pred_path = 'preds/'
temp_path = 'temp/'

def aug_daniel(prob=0.8):
	return Compose([
		RandomRotate90(p=0.5),
		Transpose(p=0.5),
		Flip(p=0.5),
		OneOf([
			IAAAdditiveGaussianNoise(),
			GaussNoise(),
			#Blur(),
		], p=0.3),
		OneOf([
			CLAHE(clip_limit=2),
			IAASharpen(),
			IAAEmboss(),
			OneOf([
				RandomContrast(),
				RandomBrightness(),
			]),
			#Blur(),
			#GaussNoise()
		], p=0.5),
		HueSaturationValue(p=0.5)
		], p=prob)
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
			
def imgaug_generator(batch_size = 4, img_size=320):
	train_data_path = 'data/train'
	temp_path = 'temp/train'
	images = glob.glob(train_data_path + '/*[0-9].png')
	shuffle(images)
	source_counter = 0
	source_limit = len(images)
	images_per_metabatch = 16
	augs_per_image = 4
	gray_lower = 255 * 0.33
	gray_upper = 255 * 0.66

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
				images = glob.glob(train_data_path + '/*[0-9].png')
				shuffle(images)
				source_counter = 0
				source_limit = len(images)
			image_name = images[source_counter].split(os.path.sep)[-1]
			image_mask_name = image_name.split('.')[0] + '_mask.png'
			img = imread(os.path.join(train_data_path, image_name), as_gray=True) #np.uint16 [0, 65535]
			mask = imread(os.path.join(train_data_path, image_mask_name), as_gray=True) #np.uint8 [0, 255]
			img = resize(img, (img_size, img_size), preserve_range=True)  #np.float32 [0.0, 65535.0]
			mask = resize(mask, (img_size, img_size), preserve_range=True) #np.float32 [0.0, 255.0]
	
			source_counter += 1

			#Convert greyscale to RGB greyscale
			img = img.astype(np.uint8)
			img_3 = np.stack((img,)*3, axis=-1)
			mask_3 = np.stack((mask,)*3, axis=-1)

			#Run each image through 8 random augmentations per image
			for j in range(augs_per_image):
				#Augment image.
				dat = augs(image=img_3, mask=mask_3)
				img_aug = np.mean(dat['image'], axis=2)
				mask_aug = np.mean(dat['mask'], axis=2)
				
				#Gray values represent inderterminate boundaries, and do not create edges along their boundaries
				white_pixels = mask_aug >= gray_upper
				black_pixels = mask_aug <= gray_lower
				gray_pixels = np.logical_not(np.logical_or(white_pixels, black_pixels))
				mask_aug[white_pixels]= 255
				mask_aug[black_pixels]= 0
				mask_aug[gray_pixels]= 127

				#Calculate edge from mask and dilate.
				mask_edge = cv2.Canny(mask_aug.astype(np.uint8), 255*3, 255*4, L2gradient=True)
				kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
				mask_edge = cv2.dilate(mask_edge.astype('float64'), kernel, iterations = 1)
				mask_edge = np.where(mask_edge > 127, 1.0, 0.0).astype('float32') #np.float32 [0.0, 1.0]
				
				patches, maskPatches = create_unagumented_data_from_image(img_aug, mask_edge)
				
#			   imsave(os.path.join(temp_path, image_name.split('.')[0] + "_" + str(j) + '.png'), np.round((patches[0,:,:,0]+1)/2*255).astype(np.uint8))
#			   imsave(os.path.join(temp_path, image_name.split('.')[0] + "_" + str(j) + '_mask.png'), (255 * maskPatches[0,:,:,0]).astype(np.uint8))
		
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
	validation_data = load_validation_data(img_size) 
	
	hyperparameters = [32, 3, 2]
	hyperparameters_string = '-'.join(str(x) for x in hyperparameters)
	model_checkpoint = ModelCheckpoint('cfm_weights_' + str(img_size) + '_e{epoch:02d}_iou{val_iou_score:.4f}.h5', monitor='val_iou_score', save_best_only=False)
	
	clr_triangular = CyclicLR(mode='triangular2', step_size=16000, base_lr=6e-4, max_lr=6e-5)
	callbacks_list = [
		EarlyStopping(patience=6, verbose=1, restore_best_weights=True),
		clr_triangular,
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
	model.compile(optimizer=AdamAccumulate(lr=1e-4, accum_iters=12), loss=bce_jaccard_loss, metrics=['binary_crossentropy', iou_score, 'accuracy'])
	model.summary()
	#model.load_weights('cfm_weights_e10_iou0.8776.h5')
	

	print('-'*30)
	print('Fitting model...')
	print('-'*30)
	train_generator = imgaug_generator(1, img_size)
	history = model.fit_generator(train_generator,
				steps_per_epoch=8000,
				epochs=20,
				validation_data=validation_data,
				verbose=1,
				max_queue_size=64,
				use_multiprocessing=True,
				workers=2,
				callbacks=callbacks_list)
	print(history.history)
