from albumentations import *
import cv2, glob, random, os
import numpy as np
from skimage.io import imsave, imread
from keras.applications import imagenet_utils
from skimage.transform import resize

def aug_validation(prob=1.0, img_size=1024):
	return Compose([
		LongestMaxSize(max_size=img_size),
		PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
	], p=prob)

def aug_resize(prob=1.0, img_size=1024):
	return Compose([
		LongestMaxSize(max_size=img_size)
	], p=prob)
	
def aug_pad(prob=1.0, img_size=1024):
	return Compose([
		PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT)
	], p=prob)

def aug_daniel_part1(prob=1.0, img_size=1024):
	return Compose([
		OneOf([
			CLAHE(clip_limit=2, p=.6),
			IAASharpen(p=.2),
			IAAEmboss(p=.2)
		], p=.7),
		OneOf([
			IAAAdditiveGaussianNoise(p=.3),
			GaussNoise(p=.7),
		], p=.5),
		RandomRotate90(p=0.5),
		Flip(p=0.5),
		Transpose(p=0.5),
		#OneOf([
		#	MotionBlur(p=.2),
		#	MedianBlur(blur_limit=3, p=.3),
		#	Blur(blur_limit=3, p=.5),
		#], p=.4),
		RandomBrightnessContrast(p=.5),
	], p=prob)

def aug_daniel_part2(prob=1.0, img_size=1024):
	return Compose([
		PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
		ShiftScaleRotate(shift_limit=.1, scale_limit=0.0, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=.75),
		#OneOf([
		#	OpticalDistortion(interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
		#	GridDistortion(distort_limit=0.2, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
		#	#ElasticTransform(approximate=True, sigma=50, alpha_affine=10, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT), # approximate gives up to 2x speedup on large images. Elastic disabled because it makes images hard to understand.
		#	IAAPiecewiseAffine(scale=(0.005, 0.015), mode='constant'),
		#	##IAAPerspective(), #produces interpolation artifacts - already tried setting to 0, 1, but perhapps not all the way?
		#	JpegCompression(quality_lower=40)
		#], p=0.7)
        JpegCompression(quality_lower=60, p=.3)
	], p=prob)
			
def imgaug_generator(batch_size = 16, img_size=1024):
	train_data_path = 'landsat_raw_boundaries/train_full'
	temp_path = 'landsat_temp_boundaries/train_full'
	images = glob.glob(train_data_path + '/*[!_mask].png')
	random.shuffle(images)
	source_counter = 0
	source_limit = len(images)
	images_per_metabatch = batch_size
	augs_per_image = batch_size

	augs_resize = aug_resize(img_size=img_size)
	augs_part1 = aug_daniel_part1(img_size=img_size)
	augs_part2 = aug_daniel_part2(img_size=img_size)

	while True:
		returnCount = 0
		batch_img = None
		batch_mask = None

		#Process up to 16 images in one batch to maitain randomness
		for i in range(images_per_metabatch):
			#Load images, resetting source "iterator" when reaching the end
			if source_counter == source_limit:
				random.shuffle(images)
				source_counter = 0
			image_name = images[source_counter].split(os.path.sep)[-1]
			image_mask_name = image_name.split('.')[0] + '_mask.png'
			img = imread(os.path.join(train_data_path, image_name), as_gray=True).astype(np.float32) #np.uint16 [0, 65535]
			mask = imread(os.path.join(train_data_path, image_mask_name), as_gray=True).astype(np.float32) #np.uint8 [0, 255]
#			img = resize(img, (img_size, img_size), preserve_range=True)  #np.float32 [0.0, 65535.0]
#			mask = resize(mask, (img_size, img_size), preserve_range=True) #np.float32 [0.0, 255.0]
	
			source_counter += 1

			#Convert greyscale to RGB greyscale, preserving as max range as possible in uint8 
			#(since it will be normalized again for imagenet means, it's ok if it's not divided by actual max of uint16)
			img = (img * (255.0 / img.max())).astype(np.uint8)

			#Run each image through 8 random augmentations per image
			for j in range(augs_per_image):
				#Augment image.
				dat_1 = augs_part1(image=img, mask=mask)
				img_aug_1 = dat_1['image']
				mask_aug_1 = dat_1['mask']
				
				#Calculate edge from mask and dilate.
				mask_aug_1 = mask_aug_1.astype(np.uint8)
				mask_aug_1 = np.where(mask_aug_1 > np.mean(mask_aug_1), 255.0, 0.0).astype(np.uint8) #np.float32 [0.0, 1.0]
				mask_edge = cv2.Canny(mask_aug_1, 100, 200)	
				kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
				mask_edge = cv2.dilate(mask_edge.astype('float64'), kernel, iterations = 1)
				mask_edge = np.where(mask_edge > np.mean(mask_edge), 1.0, 0.0).astype('float32') #np.float32 [0.0, 1.0]
				
				dat_part2 = augs_part2(image=img_aug_1, mask=mask_edge)
				img_aug = dat_part2['image']
				mask_aug = dat_part2['mask']
				if (img_aug.shape != (img_size, img_size) or mask_aug.shape != (img_size, img_size)):
					img_aug = resize(img_aug, (img_size, img_size), preserve_range=True)  #np.float32 [0.0, 65535.0]
					mask_aug = resize(mask_aug, (img_size, img_size), preserve_range=True) #np.float32 [0.0, 255.0]
				
				
				patches, maskPatches = create_unagumented_data_from_image(img_aug, mask_aug)
				
#				imsave(os.path.join(temp_path, image_name.split('.')[0] + "_" + str(j) + '.png'), np.round((patches[0,:,:,0]+1)/2*255).astype(np.uint8))
#				imsave(os.path.join(temp_path, image_name.split('.')[0] + "_" + str(j) + '_edge_mask.png'), (255 * maskPatches[0,:,:,0]).astype(np.uint8))
				
				#Easier/more efficient to discard than to resize every one
				#Suspects - Largest/Pad. Distortions unlikely (did not trigger earlier)

				#Add to batches
				if batch_img is not None:
					try:
						batch_img = np.concatenate((batch_img, patches)) #np.float32 [-1.0, 1.0], imagenet mean (~0.45)
						batch_mask = np.concatenate((batch_mask, maskPatches))  #np.float32 [0.0, 1.0]
					except:
						print('hello')
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
		while returnCount + batch_size <= totalPatches:
			batch_image_return = batch_img[returnCount:returnCount+batch_size,:,:,:]
			batch_mask_return = batch_mask[returnCount:returnCount+batch_size,:,:,:]
			returnCount += batch_size
			yield (batch_image_return, batch_mask_return)

def preprocess_input(x):
	"""Preprocesses a numpy array encoding a batch of images.
	# Arguments
		x: a 4D numpy array consists of RGB values within [0, 255].
	# Returns
		Input array scaled to [-1.,1.]
	"""
	return imagenet_utils.preprocess_input(x, mode='tf')

def create_unagumented_data_from_image(img, mask):	

	#Normalize inputs.
	img_pre = img.astype('float32')
	img_pre = preprocess_input(img_pre)
#	print(img.shape, mask.shape)
#	img -= 145.68411499 # Calulated from 33 sample images
#	img /= 33.8518720956

#	#Generate image patches from preprocessed data.
#	image_stride_random = image_stride + random.randint(-image_stride_range, image_stride_range)
#	patches, nr, nc, nR, nC = extractPatches(img, (img_rows, img_cols), image_stride_random)
#	maskPatches, nr, nc, nR, nC = extractPatches(mask, (img_rows, img_cols), image_stride_random)
	img_resize = img_pre[np.newaxis,:,:,np.newaxis]
	if mask is None:
		mask_resize = None
	else:
		mask_resize = mask[np.newaxis,:,:,np.newaxis]
#	patches = preprocess(img)
#	maskPatches = preprocess(mask)

	return img_resize, mask_resize

#train_generator = imgaug_generator(2, 384)
#for i in range(100):
#	next(train_generator)
