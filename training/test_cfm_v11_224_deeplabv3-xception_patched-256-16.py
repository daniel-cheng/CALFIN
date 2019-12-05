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
from model_cfm_dual_wide_x65 import Deeplabv3, _xception_block
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
from aug_generators_dual import aug_daniel, imgaug_generator_patched, create_unaugmented_data_patches_from_rgb_image

import error_analysis

full_size = 256
img_size = 224
stride = 16
data_path = 'data/'
pred_path = 'preds/'
temp_path = 'temp/'
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def predict(model, image):
	#Load img in as ?X? gray, cast from uint16 to uint8 (0-255)
#	patches, _ = create_unagumented_data_from_image(image, None) #np.float32 [-1.0, 1.0] (mean ~0.45), shape = (1, 224, 224, 1) #np.float32 [0.0, 1.0] (mean<0.5), shape = (1, 224, 224, 1)
	#Cast img to float32, perform imagenet preprocesssing (~0.45 means, -[1,1])
	#Resize to 1x224x224x1
#	imgs_test, nr, nc, nR, nC = load_test_data_from_image(patches, stride)

	imgs_mask_test = model.predict(image, batch_size=16, verbose=1)
#	image_cols = []
#	stride2 = int(stride/2)

#	for i in range(0, nR):
#		image_row = []
#		for j in range(0, nC):
#			index = i * nC + j
#			image = (imgs_mask_test[index, :, :, 0] * 255.).astype(np.uint8)
#			image = image[stride2:img_rows - stride2 - 1, stride2:img_cols - stride2 - 1]
#			image_row.append(image)
#		image_cols.append(np.hstack(image_row))
#	full_image = np.vstack(image_cols)
#	full_image = full_image[0:nr, 0:nc].astype(np.uint8)
	return imgs_mask_test

if __name__ == '__main__':
	initialized = True
	if initialized == False:	
		print('-'*30)
		print('Creating and compiling model...')
		print('-'*30)
		img_shape = (img_size, img_size, 3)
		inputs = Input(shape=img_shape)
		model = Deeplabv3(input_shape=(img_size, img_size,3), classes=16, OS=16, backbone='xception', weights=None)
		
		model.compile(optimizer=AdamAccumulate(lr=1e-4, accum_iters=2))
		model.summary()
		model.load_weights('cfm_weights_patched_dual_wide_x65_224_e65_iou0.5136.h5')
		
		full_size = 256
		img_size = 224
		offset = 16
		dimensions = 3
		
	plotting = True
	processing = False
	if processing == False:
		results = []
		print('-'*30)
		print('Creating images...')
		print('-'*30)
		yara_files = glob.glob(r"D:\Daniel\Documents\Github\CALFIN Repo Intercomp\training\data\validation\*Subset.png")
		validation_files = glob.glob(r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\validation\*B[0-9].png")
		validation_files = yara_files
		total = len(validation_files)
		imgs = None
		imgs_mask = None
		i = 0
		return_images = True
		
		pred_norm_patch = np.ones((img_size, img_size, 2))
		pred_norm_image = np.zeros((full_size, full_size, dimensions))
		for x in range(3):
			for y in range(3):
				x_start = x * offset
				x_end = x_start + img_size
				y_start = y * offset
				y_end = y_start + img_size
				pred_norm_image[x_start:x_end, y_start:y_end, 0:2] += pred_norm_patch
				
		for image_path in validation_files:
			image_name = image_path.split(os.path.sep)[-1]
			image_name_base = image_name.split('.')[0]
			
			img_3_uint16 = imread(image_path) #np.uint16 [0, 65535]
			if img_3_uint16.shape[2] != 3:
				img_3_uint16 = np.concatenate((img_3_uint16, img_3_uint16, img_3_uint16))
			img_3_f64 = resize(img_3_uint16, (full_size, full_size), preserve_range=True)  #np.float64 [0.0, 65535.0]
			
			#Convert greyscale to RGB greyscale
			img_max = img_3_f64.max()
			img_min = img_3_f64.min()
			img_range = img_max - img_min
			if (img_max != 0.0 and img_range > 255.0):
				img_3_uint8 = np.round(img_3_f64 / img_max * 255.0).astype(np.uint8) #np.float32 [0, 65535.0]
			else:
				img_3_uint8 = img_3_f64.astype(np.uint8)
			
			img_final_f32 = img_3_uint8.astype(np.float32) #np.float32 [0.0, 255.0]
			patches = create_unaugmented_data_patches_from_rgb_image(img_final_f32, None, window_shape=(img_size, img_size, 3), stride=stride)
			img_final_uint16 = img_final_f32.astype(np.uint16)
			
			results = predict(model, patches)
			
			raw_image = img_final_f32 / 255.0
			pred_image = np.zeros((full_size, full_size, dimensions))
			#Processes each 3x3 set of overlapping windows.
			for x in range(3):
				for y in range(3):
					x_start = x * offset
					x_end = x_start + img_size
					y_start = y * offset
					y_end = y_start + img_size
					
					pred_patch = results[x*3 + y,:,:,0:2]
					pred_patch_4d = np.expand_dims(pred_patch, axis=0)
					pred_image[x_start:x_end, y_start:y_end, 0:2] += pred_patch
					
			#Assemble 3x3 overlapping windows into single image.
			pred_image = pred_image / pred_norm_image
			overlay = raw_image*0.75 + pred_image *0.25
			
			raw_image_uint8 = (raw_image * 255).astype(np.uint8)
			pred_image_uint8 = (pred_image[:,:,0] * 255.0).astype(np.uint8)
			results_snake = error_analysis.extract_front_indicators(raw_image_uint8[:,:,0], pred_image_uint8, 0, [256, 256])
			
			
			if not results_snake is None:
				snake_image = (results_snake[0] / 255.0)
				pred_snake_image_4d = np.concatenate((snake_image[:,:,0:1], pred_image_uint8[...,np.newaxis]), axis=2)[np.newaxis,]
			else:
				snake_image = np.zeros((full_size, full_size, 3))
			
			#dilate snake image
			thickness = 3
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
			snake_image_dilated = cv2.dilate(snake_image.astype('float64'), kernel, iterations = 1).astype(np.float32) #np.float32 [0.0, 255.0]
			snake_image_dilated = np.where(snake_image_dilated > 0.5, 1.0, 0.0)
			# TODO: Per segment deviation

			
			#Plots results.
			if plotting == True:
#				plt.figure(total_images + num_sets)
				f, axarr = plt.subplots(1, 4, num=i + 1)
				axarr[0].imshow(overlay)
				axarr[1].imshow(pred_image)
				axarr[2].imshow(raw_image)
				axarr[3].imshow(snake_image)
				figManager = plt.get_current_fig_manager()
				figManager.window.showMaximized()
				
			i += 1
			print('Done {0}: {1}/{2} images'.format(image_name, i, total))
