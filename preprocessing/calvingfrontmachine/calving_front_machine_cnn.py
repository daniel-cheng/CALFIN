from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model, multi_gpu_model
from keras.models import Model, Input
from keras.layers import Concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, UpSampling2D, BatchNormalization, RepeatVector, Reshape
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l1, l2
from keras.activations import relu
from keras import backend as K
from tensorflow.python.client import device_lib

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import os, cv2, sys, glob
from skimage.io import imsave, imread

from calving_front_machine_cnn_loader import load_train_data, load_test_data_from_image
from scipy.ndimage.filters import median_filter

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 256
img_cols = 256
stride = int((img_rows + img_cols) / 2 / 2) #1/2 of img_window square
smooth = 1.
data_path = 'landsat_raw/'
pred_path = 'landsat_preds/'

def get_available_gpus():
	local_device_protos = device_lib.list_local_devices()
	return [x.name for x in local_device_protos if x.device_type == 'GPU']

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

def conv_block(m, dim, acti, bn, res, do=0):
	n = Conv2D(dim, 3, activation=acti, dilation_rate=2, padding='same')(m)
	n = BatchNormalization()(n) if bn else n
	n = Dropout(do)(n) if do else n
	n = Conv2D(dim, 3, activation=acti, dilation_rate=2, padding='same')(n)
	n = BatchNormalization()(n) if bn else n
	return Concatenate()([m, n]) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
	if depth > 0:
		#n = conv_block(m, dim, acti, bn, res, do)
		n = conv_block(m, dim, acti, bn, res)
		m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
		m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
		if up:
			m = UpSampling2D()(m)
			m = Conv2D(dim, 2, activation=acti, padding='same')(m)
		else:
			m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
		n = Concatenate()([n, m])
		m = conv_block(n, dim, acti, bn, res)
		#m = conv_block(n, dim, acti, bn, res, do)
	else:
		m = conv_block(m, dim, acti, bn, res, do)
	return m

def UNet(img_shape, out_ch=1, start_ch=48, depth=4, inc_rate=2., activation='relu',
		 dropout=0.5, batchnorm=True, maxpool=True, upconv=True, residual=True):
	i = Input(shape=img_shape)
	o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
	o = Conv2D(out_ch, 1, activation='sigmoid')(o)
	return Model(inputs=i, outputs=o)

def predict(model, image):
	imgs_test, nr, nc, nR, nC = load_test_data_from_image(image, stride)

	imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)

	image_cols = []
	stride2 = int(stride/2)

	for i in range(0, nR):
		image_row = []
		for j in range(0, nC):
			index = i * nC + j
			image = (imgs_mask_test[index, :, :, 0] * 255.).astype(np.uint8)
			image = image[stride2:img_rows - stride2 - 1, stride2:img_cols - stride2 - 1]
			image_row.append(image)
		image_cols.append(np.hstack(image_row))
	full_image = np.vstack(image_cols)
#	full_image = median_filter(full_image, 7)
#	full_image = np.where(full_image > 127, 255, 0)
	full_image = full_image[0:nr, 0:nc].astype(np.uint8)
	return full_image

def resolve(name, basepath=None):
	if not basepath:
	  basepath = os.path.dirname(os.path.realpath(__file__))
	return os.path.join(basepath, name)

def removeSmallComponents(image:np.ndarray, start:int, mask_value:int):
	image = image.astype('uint8')
	#find all your connected components (white blobs in your image)
	nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
	#connectedComponentswithStats yields every seperated component with information on each of them, such as size
	sizes = stats[:,cv2.CC_STAT_AREA]
	ordering = np.argsort(-sizes)
	#print(sizes, ordering)
	
	min_size = output.size * 0.05
	# print(min_size)
	#your answer image
	largeComponents = np.zeros((output.shape))
	#for every component in the image, you keep it only if it's above min_size
	for i in range(start, min(3, len(sizes))):
		# print(sizes, ordering[i])
		if sizes[ordering[i]] >= min_size:
			mask_indices = output == ordering[i]
			first_index = np.where(mask_indices==True)
			# print(image[first_index[0][0], first_index[1][0]] == mask_value)
			if image[first_index[0][0], first_index[1][0]] == mask_value:
				largeComponents[mask_indices] = 255
	return largeComponents.astype('uint8')
	
def postprocess(image:np.ndarray) -> np.ndarray:
	"""Description: Postprocesses mask layer to remove small features.
	"""
	# Close edges to join them and dilate them before removing small components
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
	closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
	dilated = cv2.dilate(closing, kernel, iterations = 1)
	largeComponents = removeSmallComponents(dilated, 0, 255)
	# largeComponents = dilated

	# Remove small components inside floodfill area
	largeComponentsInverted = 255 - largeComponents
	largeComponentsInverted = removeSmallComponents(largeComponentsInverted, 0, 255)
	largeComponentsInverted2 = 255 - largeComponentsInverted
	
	# Reverse initial morphological operators to retrieve original edge mask
	eroded = cv2.erode(largeComponentsInverted2, kernel, iterations = 1)
	opening = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
	
	result = removeSmallComponents(opening, 0, 255)
	# result = opening
	
	return result, dilated, largeComponents, largeComponentsInverted, largeComponentsInverted2

def calculate_confidence(averages:np.array, path:str, confidence_threshold=0.35):
	count = averages.shape[2]
	confidence = np.sum(averages==255, axis=0)
	confidence = confidence / count * 255
	confidence = confidence.astype(np.uint8)
	imsave(os.path.join(path, 'all_pred_confidence-' + str(confidence_threshold) + '.png'), confidence)

	confidence_high_0 = np.where(confidence < (confidence_threshold + 0.05) * 255, 255, 0)
	confidence_high_1 = np.where(confidence > 255 - confidence_threshold * 255, 255, 0)
	confidence_low = np.logical_not(np.logical_or(confidence_high_0 == 255, confidence_high_1 == 255)) * 255
	imsave(os.path.join(path, 'confidence_high_0-' + str(confidence_threshold) + '.png'), confidence_high_0)
	imsave(os.path.join(path, 'confidence_high_1-' + str(confidence_threshold) + '.png'), confidence_high_1)
	imsave(os.path.join(path, 'confidence_low-' + str(confidence_threshold) + '.png'), confidence_low)

	# Otsu's thresholding
	ret2, confidence_binary = cv2.threshold(confidence, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	imsave(os.path.join(path, 'all_pred_confidence_binary-' + str(confidence_threshold) + '.png'), confidence_binary)
	return confidence, confidence_high_0, confidence_high_1, confidence_binary, confidence_low

def estimate(images:np.array, confidence_binary:np.array, valid_prediction:np.array, index:int):
	if (index == 0):
		# Predict on 0
		index_0 = index
		index_1 = index + 1
		index_2 = index + 2
		
	elif (index == len(images)):
		# Predict on 2
		index_0 = index - 2
		index_1 = index - 1
		index_2 = index
	else:
		# Predict on 1
		index_0 = index - 1
		index_1 = index
		index_2 = index + 1
	return confidence_binary
		
if __name__ == '__main__':
	#train_and_predict()
	
	print('-'*30)
	print('Predicting masks on test data...')
	print('-'*30)
	path2 = r'D:/Daniel/Documents/GitHub/ultrasound-nerve-segmentation/landsat_raw/train_full/'
	in_path = r'C:/Users/Daniel/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/calvingfrontmachine/landsat_raw/' + sys.argv[1]
	out_path = r'C:/Users/Daniel/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/calvingfrontmachine/landsat_preds/' + sys.argv[1]
	domain_path = path2 + sys.argv[1]
	masking = sys.argv[2] == '1'
	saving = sys.argv[3] == '1'
	postprocessing = sys.argv[4] == '1'
	
	if masking:
		#Load model
		print('-'*30)
		print('Creating and compiling model...')
		print('-'*30)
		hyperparameters = [48, 5, 1.5]
		hyperparameters_string = '-'.join(str(x) for x in hyperparameters)
		model = UNet((img_rows,img_cols,1), start_ch=hyperparameters[0], depth=hyperparameters[1], inc_rate=hyperparameters[2], activation='elu')
		model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
		
		print('-'*30)
		print('Loading saved weights...')
		print('-'*30)
		model.load_weights('landsat_weights_256_48-5-1.5_-0.93202.h5')

		# Clear previous runs if masking again
		files = glob.glob(out_path + '/*_mask.png')
		for f in files:
			if os.path.isfile(f):
				os.remove(f)

	average = []	
	for name in os.listdir(in_path):
		if masking:				
			if '_mask.png' in name or '_bqa.png' in name  or '_mtl.txt' in name or not os.path.isfile(os.path.join(in_path, name)):
				continue		
			image = imread(os.path.join(in_path, name), as_gray = True)
			result = predict(model, image)
			imsave(os.path.join(out_path, name[0:-4] + '_raw.png'), image)
			imsave(os.path.join(out_path, name[0:-4] + '_raw_mask.png'), result)
		else:
			if '_raw.png' in name or '_bqa' in name or '_mtl' in name or 'confidence' in name or not os.path.isfile(os.path.join(in_path, name)):
				continue
			result = imread(os.path.join(out_path, name[0:-4] + '_raw_mask.png'), as_gray = True)
		average.append(result)
	#Eliminate outliers and recalculate confidence image
	average = np.array(average)
	confidence, confidence_high_0, confidence_high_1, confidence_binary, confidence_low = calculate_confidence(average, out_path, 0.35)
	similarity_scores = []
	for i in range(len(average)):
		similarity_scores.append(np.sum(np.logical_xor(confidence_binary, average[i])) / confidence.size)
	#Calculate mean and standard deviation and throw away predictions outside of 1 std.
	mean = np.mean(similarity_scores)
	std = np.std(similarity_scores)
	min_similarity = mean + std / 2
	valid_prediction = np.array(similarity_scores) < min_similarity
	print(min_similarity, mean, std)
	for i in range(len(average)):
		if (not valid_prediction[i]):
			print('estimating', i)
			average[i] = estimate(average, confidence_binary, valid_prediction, i)
	
	#Apply confidence image to masks and save
	confidence, confidence_high_0, confidence_high_1, confidence_binary, confidence_low = calculate_confidence(average, out_path, 0.075)
	count = 0
	if saving:
		if not os.path.exists(domain_path):
			os.mkdir(domain_path)
	for name in os.listdir(in_path):
		if '_raw.png' in name or '_bqa' in name or '_mtl' in name or 'confidence' in name or not os.path.isfile(os.path.join(in_path, name)):
			continue
		image = np.copy(average[count])
		image[255 == confidence_high_0] = 0
		image[255 == confidence_high_1] = 255
		if postprocessing:
			image, dilated, largeComponents, largeComponentsInverted, largeComponentsInverted2 = postprocess(image)
			# imsave(os.path.join(out_path, name[0:-4] + '_dilated.png'), dilated)
			# imsave(os.path.join(out_path, name[0:-4] + '_largeComponents.png'), largeComponents)
			# imsave(os.path.join(out_path, name[0:-4] + '_largeComponentsInverted.png'), largeComponentsInverted)
			# imsave(os.path.join(out_path, name[0:-4] + '_largeComponentsInverted2.png'), largeComponentsInverted2)
		if saving:
			imsave(os.path.join(domain_path, name[0:-4] + '_mask.png'), image)
		imsave(os.path.join(out_path, name[0:-4] + '_mask.png'), image)
		image = ((image != confidence_binary) * 255).astype(np.uint8)
		imsave(os.path.join(out_path, name[0:-4] + '_diff.png'), image)
		count += 1
	