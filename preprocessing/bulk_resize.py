import os
import numpy as np
from skimage.io import imsave, imread
from skimage.transform import resize

use_reference_folder = False
if not use_reference_folder:
	source_path = r"D:\Daniel\Documents\Github\CALFIN Repo\processing\landsat_raw_processed"
	
	# Generate mask confidence from masks
	for domain in os.listdir(source_path):
		domain_path = os.path.join(source_path, domain)
		resolutions = []
		for file_name in os.listdir(domain_path):
			file_path = os.path.join(domain_path, file_name)
			result = imread(file_path, as_gray = True)
			resolutions.append(result.shape)
		
		resolution = np.ceil(np.median(resolutions, axis=0))
		print(domain, resolution)
		for file_name in os.listdir(domain_path):
			file_path = os.path.join(domain_path, file_name)
			img = imread(file_path)
			if img.dtype == np.uint8:
				img = (img.astype(np.float32) / 255 * 65535).astype(np.uint16)
				imsave(file_path, img)
			if img.shape[0] != resolution[0] or img.shape[1] != resolution[1]:
				print('Saving:', file_path)
				img = resize(img, resolution, preserve_range=True).astype(np.uint16)
				imsave(file_path, img)
				error()
else:
	source_path = r"./calvingfrontmachine/landsat_raw"
	ref_path = r"./calvingfrontmachine/landsat_raw_old"
	
	# Generate mask confidence from masks
	for domain in os.listdir(source_path):
		domain_path = os.path.join(source_path, domain)
		domain_ref_path = os.path.join(ref_path, domain)
		resolutions = []
		for file_name in os.listdir(domain_ref_path):
			file_path = os.path.join(domain_ref_path, file_name)
			result = imread(file_path, as_gray = True)
			resolutions.append(result.shape)
			break
		
		resolution = np.ceil(np.median(resolutions, axis=0))
		print(domain, resolution)
		for file_name in os.listdir(domain_path):
			file_path = os.path.join(domain_path, file_name)
			img = imread(file_path)
			if img.dtype == np.uint8:
				img = (img.astype(np.float32) / 255 * 65535).astype(np.uint16)
				imsave(file_path, img)
			if img.shape[0] != resolution[0] or img.shape[1] != resolution[1]:
				print('Saving:', file_path)
				img = resize(img, resolution, preserve_range=True).astype(np.uint16)
				imsave(file_path, img)