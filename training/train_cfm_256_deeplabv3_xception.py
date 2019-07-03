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

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import cv2, glob
from skimage.io import imsave, imread
from skimage.transform import resize, rotate, rescale
from random import shuffle

from data_cfm_256 import create_unagumented_data_from_image, load_validation_data
from albumentations import *

img_size = 256
data_path = 'data/'
pred_path = 'preds/'
temp_path = 'temp/'

def aug_daniel_part1(prob=1.0):
	return Compose([
        OneOf([
            LongestMaxSize(max_size=img_size),
            Resize(img_size, img_size)
        ], p=1.0),
		OneOf([
			CLAHE(clip_limit=2, p=.6),
			IAASharpen(p=.2),
			IAAEmboss(p=.2)
		], p=.8),
		OneOf([
			IAAAdditiveGaussianNoise(p=.3),
			GaussNoise(p=.7),
		], p=.6),
		RandomRotate90(p=0.5),
		Flip(p=0.5),
		Transpose(p=0.5),
		OneOf([
			MotionBlur(p=.2),
			MedianBlur(blur_limit=3, p=.3),
			Blur(blur_limit=3, p=.5),
		], p=.4),
		RandomBrightnessContrast(p=.5),
	], p=prob)
	
def aug_daniel_part2(prob=1.0):
	return Compose([
        PadIfNeeded(min_height=img_size, min_width=img_size),
		ShiftScaleRotate(shift_limit=.05, scale_limit=0.0, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=.75),
		OneOf([
			OpticalDistortion(interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
			GridDistortion(distort_limit=0.15, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
			#ElasticTransform(approximate=True, sigma=50, alpha_affine=10, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT), # approximate gives up to 2x speedup on large images. Elastic disabled because it makes images hard to understand.
			IAAPiecewiseAffine(scale=(0.005, 0.015), mode='constant'),
			##IAAPerspective(), #produces interpolation artifacts - already tried setting to 0, 1, but perhapps not all the way?
			JpegCompression(quality_lower=40)
		], p=0.7)
	], p=prob)
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
			
def imgaug_generator(batch_size = 16):
	train_data_path = 'data/train'
	temp_path = 'temp/train'
	images = glob.glob(train_data_path + '/*[0-9].png')
	shuffle(images)
	source_counter = 0
	source_limit = len(images)
	images_per_metabatch = batch_size
	augs_per_image = batch_size

	augs_part1 = aug_daniel_part1()
	augs_part2 = aug_daniel_part2()

	while True:
		returnCount = 0
		batch_img_patches = None
		batch_mask_patches = None
		batch_img = None
		batch_mask = None

		#Process up to 16 images in one batch to maitain randomness
		for i in range(images_per_metabatch):
			#Load images, resetting source "iterator" when reaching the end. Also updates images if new ones are added during training.
			if source_counter == source_limit:
				images = glob.glob(train_data_path + '/*[0-9].png')
				shuffle(images)
				source_counter = 0
				source_limit = len(images)
			image_name = images[source_counter].split(os.path.sep)[-1]
			image_mask_name = image_name.split('.')[0] + '_mask.png'
			img = imread(os.path.join(train_data_path, image_name), as_gray=True) #np.uint16 [0, 65535]
			mask = imread(os.path.join(train_data_path, image_mask_name), as_gray=True).astype(np.float32) #np.uint8 [0, 255]
			img = resize(img, (img_size, img_size), preserve_range=True)  #np.float32 [0.0, 65535.0]
			mask = resize(mask, (img_size, img_size), preserve_range=True) #np.float32 [0.0, 255.0]
	
			source_counter += 1

			#Convert greyscale to RGB greyscale, preserving as max range as possible in uint8 
			#(since it will be normalized again for imagenet means, it's ok if it's not divided by actual max of uint16)
			img = (img * (255.0 / img.max())).astype(np.uint8)
			img_rgb = np.stack((img,)*3, axis=-1)
			mask_rgb = np.stack((mask,)*3, axis=-1)
			gray_lower = 255 * 0.33
			gray_upper = 255 * 0.66
			#Run each image through 8 random augmentations per image
			for j in range(augs_per_image):
				#Augment image.
				dat_1 = augs_part1(image=img_rgb, mask=mask_rgb)
				img_aug_1 = dat_1['image']
				mask_aug_1 = dat_1['mask']
				
				#Calculate edge from mask and dilate.
				mask_aug_1 = mask_aug_1.astype(np.uint8)
				#Gray values represent inderterminate boundaries, and do not create edges along their boundaries
				white_pixels = mask_aug_1 >= gray_upper
				black_pixels = mask_aug_1 <= gray_lower
				gray_pixels = np.logical_not(np.logical_or(white_pixels, black_pixels))
				mask_aug_1[white_pixels]= 255
				mask_aug_1[black_pixels]= 0
				mask_aug_1[gray_pixels]= 127
				mask_edge = cv2.Canny(mask_aug_1.astype(np.uint8), 255*3, 255*4, L2gradient=True)
				kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
				mask_edge = cv2.dilate(mask_edge.astype('float64'), kernel, iterations = 1)
				mask_edge = np.where(mask_edge > np.mean(mask_edge), 1.0, 0.0).astype('float32') #np.float32 [0.0, 1.0]
				mask_edge_rgb = np.stack((mask_edge,)*3, axis=-1)
				
				#Perform second augmentations after edge is generated
				dat_part2 = augs_part2(image=img_aug_1, mask=mask_edge_rgb)
				
				img_aug = np.mean(dat_part2['image'], axis=2)
				mask_aug = np.mean(dat_part2['mask'], axis=2)
				img_aug = (img_aug / img_aug.max() * 255).astype(np.uint8)  #np.uint8 [0, 255]
				
				patches, maskPatches = create_unagumented_data_from_image(img_aug, mask_aug)
				
#				imsave(os.path.join(temp_path, image_name.split('.')[0] + "_" + str(j) + '.png'), np.round((patches[0,:,:,0]+1)/2*255).astype(np.uint8))
#				imsave(os.path.join(temp_path, image_name.split('.')[0] + "_" + str(j) + '_edge_mask.png'), (255 * maskPatches[0,:,:,0]).astype(np.uint8))
				
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
	
	clr_triangular = CyclicLR(mode='triangular2', step_size=4000, base_lr=6e-4, max_lr=6e-5)
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
	model.compile(optimizer=Adam(lr=1e-5), loss=bce_jaccard_loss, metrics=['binary_crossentropy', iou_score, 'accuracy'])
	model.summary()
	#model.load_weights('cfm_weights_e10_iou0.8776.h5')
	

	print('-'*30)
	print('Fitting model...')
	print('-'*30)
	train_generator = imgaug_generator(12)
	history = model.fit_generator(train_generator,
				steps_per_epoch=2000,
				epochs=20,
				validation_data=validation_data,
				verbose=1,
                max_queue_size=64,
				use_multiprocessing=True,
                workers=2,
				callbacks=callbacks_list)
	print(history.history)
