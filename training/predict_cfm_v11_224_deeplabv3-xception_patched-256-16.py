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
from aug_generators_dual import aug_daniel, imgaug_generator_patched

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
	initialized = False
	if initialized == False:
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
			pr_sum = np.sum(pr[:,:,:,0])
			gt_sum = np.sum(gt[:,:,:,0])
			edge_diff = np.abs(gt_sum - pr_sum) / ((pr_sum + gt_sum) / 2)
			return (edge_iou_score * 25) + mask_iou_score/26
	
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
		
		model.compile(optimizer=AdamAccumulate(lr=1e-4, accum_iters=2), loss=bce_ln_jaccard_loss, metrics=['binary_crossentropy', iou_score, edge_iou_score, mask_iou_score, deviation])
		model.summary()
		model.load_weights('cfm_weights_patched_dual_wide_x65_224_e65_iou0.5136.h5')
		
		full_size = 256
		img_size = 224
		offset = 16
		dimensions = 3
		validation_tuples = [(validation_data[0], validation_targets[0], 'all'), (validation_data[1], validation_targets[1], 'Upernavik'), (validation_data[2], validation_targets[2], 'Jakobshavn'), (validation_data[3], validation_targets[3], 'Kong-Oscar'), (validation_data[4], validation_targets[4], 'Kangiata-Nunaata'), (validation_data[5], validation_targets[5], 'Hayes'), (validation_data[6], validation_targets[6], 'Rink-Isbrae'), (validation_data[7], validation_targets[7], 'Kangerlussuaq'), (validation_data[8], validation_targets[8], 'Helheim')]
		
		results = []
		for images, targets, name in validation_tuples:
			print(images.shape, targets.shape, name)
			results.append(predict(model, images))
		num_sets = len(validation_tuples)
		initialized = True
	
	def deviation(gt, pr, smooth=SMOOTH2, per_image=True):
		mismatch = np.sum(np.abs(gt[:,:,:,1] - pr[:,:,:,1]), axis=(1, 2)) #(B)
		length = np.sum(pr[:,:,:,0], axis=(1, 2)) #(B)
		deviation = mismatch / (length + smooth) #(B)
		mean_deviation = np.mean(deviation) * 3.0 #- (account for line thickness of 3 at 224)
		return mean_deviation
	
	def edge_iou_score(gt, pr, smooth=SMOOTH, per_image=True):
		intersection = np.sum(np.abs(gt[:,:,:,0] - pr[:,:,:,0]), axis=(1, 2)) #(B)
		union = np.sum(gt[:,:,:,0] + pr[:,:,:,0] > 0.5, axis=(1, 2)) #(B)
		iou_score = intersection / (union + smooth) #(B)
		mean_iou_score = np.mean(iou_score) #- (account for line thickness of 3 at 224)
		return mean_iou_score
	
	def edge_deviation(gt, pr, smooth=SMOOTH2, per_image=True):
		mismatch = np.sum(np.abs(gt[:,:,:,0] - pr[:,:,:,0]), axis=(1, 2)) #(B)
		length = np.sum(gt[:,:,:,0], axis=(1, 2)) #(B)
		deviation = mismatch / (length + smooth) #(B)
		mean_deviation = np.mean(deviation) * 3.0 #- (account for line thickness of 3 at 224)
		return mean_deviation
	
	replot = True
	if replot == True:
		plt.close('all')
		dimensions = 3
		count = 0
		total_images = 0
		metrics = defaultdict(list)
		deviations = []
		mean_deviations = []
		deviations_snake = []
		mean_deviations_snake = []
		edge_ious = []
		mean_edge_ious = []
		edge_ious_snake = []
		mean_edge_ious_snake = []
		domain_mask_max_sum = []
		pred_norm_patch = np.ones((img_size, img_size, 2))
		pred_norm_image = np.zeros((full_size, full_size, dimensions))
		for x in range(3):
			for y in range(3):
				x_start = x * offset
				x_end = x_start + img_size
				y_start = y * offset
				y_end = y_start + img_size
				pred_norm_image[x_start:x_end, y_start:y_end, 0:2] += pred_norm_patch
						
		for images, targets, name in validation_tuples:
			print(images.shape, targets.shape, name)
			domain_metrics = []
			domain_results = results[count]
			domain_deviations = []
			domain_deviations_snake = []
			domain_mean_deviation = 0
			domain_mean_deviation_snake = 0
			domain_edge_ious = []
			domain_edge_ious_snake = []
			domain_mean_edge_iou = 0
			domain_mean_edge_iou_snake = 0
			domain_mask_sums = [256*256]
			domain_mask_max_sum.append(np.zeros((full_size, full_size, dimensions)))
			for i in range(int(images.shape[0] / 9)):
	#			if count == 0 and total_images + num_sets == 129:
				if True:
					pred_image = np.zeros((full_size, full_size, dimensions))
					raw_image = np.zeros((full_size, full_size, dimensions))
					mask_image = np.zeros((full_size, full_size, dimensions))
					#Processes each 3x3 set of overlapping windows.
					for x in range(3):
						for y in range(3):
							x_start = x * offset
							x_end = x_start + img_size
							y_start = y * offset
							y_end = y_start + img_size
							
							pred_patch = domain_results[i*9 + x*3 + y,:,:,0:2]
							mask_patch = validation_targets[count][i*9 + x*3 + y,:,:,0:2]
							raw_patch = validation_data[count][i*9 + x*3 + y,:,:,0:3]
							
							mask_patch_4d = np.expand_dims(mask_patch, axis=0)
							pred_patch_4d = np.expand_dims(pred_patch, axis=0)
							
							pred_image[x_start:x_end, y_start:y_end, 0:2] += pred_patch
							mask_image[x_start:x_end, y_start:y_end, 0:2] = mask_patch
							raw_image[x_start:x_end, y_start:y_end, 0:3] = raw_patch
							
							domain_deviations.append(deviation(mask_patch_4d, pred_patch_4d))
							domain_edge_ious.append(edge_iou_score(mask_patch_4d, pred_patch_4d))
							
					#Assemble 3x3 overlapping windows into single image.
					pred_image = pred_image / pred_norm_image
					raw_image = (raw_image + 1.0) / 2.0
					overlay = raw_image*0.5 + pred_image *0.5
					
					raw_image_uint8 = (raw_image[:,:,0] * 255.0).astype(np.uint8)
					pred_image_uint8 = (pred_image[:,:,0] * 255.0).astype(np.uint8)
					results_snake = error_analysis.extract_front_indicators(raw_image_uint8, pred_image_uint8, 0, [256, 256])
					
					
					if not results_snake is None:
						snake_image = (results_snake[0] / 255.0)
						mask_image_4d = np.expand_dims(mask_image, axis=0)
						pred_snake_image_4d = np.concatenate((snake_image[:,:,0:1], pred_image_uint8[...,np.newaxis]), axis=2)[np.newaxis,]
						domain_deviations_snake.append(deviation(mask_image_4d, pred_snake_image_4d))
						domain_edge_ious_snake.append(edge_iou_score(mask_image_4d, pred_snake_image_4d))
					else:
						domain_deviations_snake.append(domain_deviations[-1])
						domain_edge_ious_snake.append(domain_edge_ious[-1])
						snake_image = np.zeros((full_size, full_size, 3))
					
					#dilate snake image
					thickness = 3
					kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
					snake_image_dilated = cv2.dilate(snake_image.astype('float64'), kernel, iterations = 1).astype(np.float32) #np.float32 [0.0, 255.0]
					snake_image_dilated = np.where(snake_image_dilated > 0.5, 1.0, 0.0)
					#TODO: fix!
					overlap_g = np.logical_and(snake_image_dilated[:,:,0] > 0.5, mask_image[:,:,0] > 0.5)
					
					overlap_image = np.stack((snake_image_dilated[:,:,0], overlap_g, mask_image[:,:,0]), axis=-1)
					
#					np.nonzero(mask_image[:,:,0] > 0.5)
#					np.nonzero(snake_image_dilated[:,:,0] > 0.5)
#					for 
					print("\r" + str(total_images), end='')
				
				#Plots results.
	#			if count == 0 and total_images + num_sets == 129:
				if count == 0 and True:
	#				plt.figure(total_images + num_sets)
					f, axarr = plt.subplots(2, 3, num=total_images + num_sets + 1)
					axarr[0,0].imshow(overlay)
					axarr[0,1].imshow(pred_image)
					axarr[0,2].imshow(raw_image)
					axarr[1,0].imshow(mask_image)
					axarr[1,1].imshow(snake_image)
					axarr[1,2].imshow(overlap_image)
					figManager = plt.get_current_fig_manager()
					figManager.window.showMaximized()
				total_images += 1
			
			#Save results.
			deviations.append(domain_deviations)
			edge_ious.append(domain_edge_ious)
			domain_mean_deviation = np.mean(np.array(domain_deviations))
			domain_mean_edge_iou = np.mean(np.array(domain_edge_ious))
			mean_deviations.append(domain_mean_deviation)
			mean_edge_ious.append(domain_mean_edge_iou)
			domain_mean_deviation_snake = np.mean(domain_deviations_snake)
			domain_mean_edge_iou_snake = np.mean(domain_edge_ious_snake)
			domain_mean_deviation_snake_change = domain_mean_deviation - domain_mean_deviation_snake
			domain_mean_edge_iou_snake_change = domain_mean_edge_iou - domain_mean_edge_iou_snake
			print('\n', name, 'mean deviation:', domain_mean_deviation, 'mean deviation snake change:', domain_mean_deviation_snake_change)
			print(name, 'mean edge_iou:', domain_mean_edge_iou, 'mean edge_iou snake change:', domain_mean_edge_iou_snake_change)
			count += 1
	
	#Shows histograms of mean deviations per validation set.
#	resolutions = {'Upernavik-NE': (503, 475), 'Kangiata-Nunata': (533, 552), 'Kong-Oscar':(326, 304), 'Rink-Isbrae': (332, 328), 'Hayes': (287, 350), 'Jakobshavn': (644, 912), 'Akugdlerssup': (144, 224), 'Cornell': (174, 173), 'Dietrichson': (173, 241), 'Docker-Smith': (157, 155), 'Docker-Smith-N': (184, 202), 'Eqip': (92, 100), 'Gade': (240, 226), 'Hayes-M': (159, 176), 'Hayes-S': (355, 232), 'Helheim': (1031, 1031), 'Igssussarssuit': (231, 225), 'Illullip': (242, 189), 'Inngia': (391, 405), 'Kangerlussuaq': (569, 569),'Kangilernata': (87, 112), 'Kjer': (302, 286), 'Morell': (233, 224), 'Nansen': (280, 378), 'Narsap': (291, 403), 'Nordenskiold': (362, 281), 'Rink-Gletscher': (204, 257), 'Saqqarliup': (87, 122), 'Sermeq-Avannarleq-N': (70, 122), 'Sermeq-Avannarleq-S': (84, 108), 'Sermeq-Kujalleq': (98, 86), 'Steenstrup': (387, 304), 'Sverdrup': (306, 356), 'Umiammakku': (362, 351), 'Upernavik-SE': (466, 287)}
	resolutions = {'Upernavik': (495, 512), 'Kangiata-Nunaata': (533, 552), 'Kong-Oscar':(354, 369), 'Rink-Isbrae': (525, 560), 'Hayes': (385, 399), 'Jakobshavn': (821, 787), 'Helheim': (1031, 1031), 'Kangerlussuaq': (569, 569)}
	show_bins = False
	if show_bins == True:
		plt.close('all')
		hist_bins = 35
		for i in range(len(validation_tuples)):
			name = validation_tuples[i][2]
	#		plt.clf()
			if name in resolutions:
				domain_deviations = deviations[i]
				domain_deviations_filtered = list(filter(lambda x: x < 128, domain_deviations))
				domain_resolution_average = np.mean(resolutions[name])
				domain_pixel_to_meters = domain_resolution_average / full_size * 30 # get normalized size of pixels in meters, per domain (landsat 3+ bands 4/5/7 = 30meters/pixel)
#				domain_deviations_meters = domain_deviations * np.array(domain_pixel_to_meters)
#				domain_deviations_filtered_meters = domain_deviations_filtered * np.array(domain_pixel_to_meters)
				
				domain_mean_deviation = mean_deviations[i]
				domain_mean_deviation_filtered = np.mean(domain_deviations_filtered)
				domain_mean_deviation_meters = domain_mean_deviation * domain_pixel_to_meters
				domain_mean_deviation_filtered_meters = domain_mean_deviation_filtered * domain_pixel_to_meters
				
				print('\n', name, 'mean deviation (pixels):', domain_mean_deviation, 'mean deviation (meters):', domain_mean_deviation_meters)
				print(name, 'mean deviation filtered (pixels):', domain_mean_deviation_filtered, 'mean deviation filtered (meters):', domain_mean_deviation_filtered_meters)
				print(name, 'failed:', str(len(domain_deviations) - len(domain_deviations_filtered)) + '/' + str(len(domain_deviations)) + ', ', str((len(domain_deviations) - len(domain_deviations_filtered))/len(domain_deviations) * 100) + '%')
				
				f, axarr = plt.subplots(1, 1, num = i + 1)
				plt.title(name)
#				axarr[0].hist(domain_deviations, bins=hist_bins)
				axarr.hist(domain_deviations_filtered, bins=hist_bins)
				axarr.set_xlabel('Mean deviation from the front, filtered (pixels)')
				axarr.set_ylabel('Number of images')
			else:
				domain_deviations = deviations[i]
				domain_deviations_filtered = list(filter(lambda x: x < 128, domain_deviations))
				domain_mean_deviation = mean_deviations[i]
				domain_mean_deviation_filtered = np.mean(domain_deviations_filtered)
				print('\n', name, 'mean deviation (pixels):', domain_mean_deviation)
				print(name, 'mean deviation filtered (pixels):', domain_mean_deviation_filtered)
				print(name, 'failed:', str(len(domain_deviations) - len(domain_deviations_filtered)) + '/' + str(len(domain_deviations)) + ', ', str((len(domain_deviations) - len(domain_deviations_filtered))/len(domain_deviations) * 100) + '%')
				f, axarr = plt.subplots(1, 1, num = i + 1)
				plt.title(name)
#				axarr[0].hist(domain_deviations, bins=hist_bins)
				axarr.hist(domain_deviations_filtered, bins=hist_bins)
				axarr.set_xlabel('Mean deviation from the front, filtered (pixels)')
				axarr.set_ylabel('Number of images')
				
			figManager = plt.get_current_fig_manager()
			figManager.window.showMaximized()
#			fig.canvas.draw()
	plt.show()
#		sum_result = None
#		for angle in [0, 90, 180, 270]:
##			for angle in [0]:
#			rot_img = rotate(img, angle, mode='reflect', preserve_range=True).astype(np.uint8)
#			result = predict(model, rot_img)
#			result = rotate(result, -angle, mode='reflect', preserve_range=True)
#			
#			if sum_result is None:
#				sum_result = result / 4.0
#			else:
#				sum_result += result / 4.0
				
#		imsave(os.path.join(output_domain_path, name + '_raw.png'), img)
#		imsave(os.path.join(output_domain_path, name + '_pred.png'), (sum_result * 255.0).astype(np.uint8))
	#				
	
#	print('-'*30)
#	print('Fitting model...')
#	print('-'*30)
#	steps_per_epoch = 4000
#	train_generator = imgaug_generator_patched(8, img_size=full_size, patch_size=img_size, patch_stride=stride, steps_per_epoch=steps_per_epoch)
#	history = model.fit_generator(train_generator,
#				steps_per_epoch=steps_per_epoch,
#				epochs=80,
#				validation_data=(validation_data[0], validation_targets[0]),
#				verbose=1,
##				max_queue_size=64,
##				use_multiprocessing=True,
##				workers=2,
#				callbacks=callbacks_list)
#	print(history.history)
