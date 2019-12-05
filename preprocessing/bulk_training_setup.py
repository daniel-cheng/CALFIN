# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:38:07 2019

@author: Daniel
"""
import numpy as np
import os, shutil, glob, random
from skimage.io import imsave, imread
import sys
sys.path.insert(0, '../training')
from aug_generators import aug_resize

source_path = r'C:\Users\Daniel\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\calvingfrontmachine\landsat_raw'
dest_path = r'D:\Daniel\Documents\Github\CALFIN Repo\reprocessing\images_1024'
old_dest_path = r'D:\Daniel\Documents\Github\CALFIN Repo\reprocessing\images_1024_twentieth'
calendar_path = r'D:\Daniel\Documents\Github\CALFIN Repo\preprocessing\calendars'

domains = sorted(os.listdir(source_path))
domains_count = len(domains)

augs = aug_resize(img_size=1024)
counter = 0
total = 0
removed = 0
for domain in os.listdir(source_path):
	source_domain_path = os.path.join(source_path, domain)
	dest_domain_path = os.path.join(dest_path, domain)
	old_dest_domain_path = os.path.join(old_dest_path, domain)
	
	if not os.path.exists(dest_domain_path):
		os.mkdir(dest_domain_path)
	count = len(glob.glob(os.path.join(source_domain_path, '*B[0-9].png')))
	print(domain + ':', count)
	total += count
	for file_path in glob.glob(os.path.join(source_domain_path, '*B[0-9].png')):
		image_name = os.path.basename(file_path)
		satellite = image_name.split('_')[1]
		if counter % 5 == 0 or (counter % 4 == 0 and satellite == 'LE07'):
#			img = imread(file_path, as_gray=True)
#			if img.dtype == np.uint8:
#				img = img.astype(np.uint16) * 257
#			elif img.dtype == np.float64:
#				img = (img * 65535).astype(np.uint16)
#			
			image_mask_name = image_name.split('.')[0] + '_mask.png'
			
			raw_dest_path = os.path.join(dest_domain_path, image_name)
			mask_dest_path = os.path.join(dest_domain_path, image_mask_name)
			old_raw_dest_path = os.path.join(old_dest_domain_path, image_name)
			old_mask_dest_path = os.path.join(old_dest_domain_path, image_mask_name)
#			dat = augs(image=img)
#			img_aug = dat['image'] #np.uint15 [0, 65535]
					
			if os.path.exists(old_raw_dest_path) and os.path.exists(raw_dest_path):
				os.remove(raw_dest_path)
			if os.path.exists(old_mask_dest_path) and os.path.exists(mask_dest_path):
				os.remove(mask_dest_path)
				print('removing:', old_raw_dest_path)
				removed += 1
#				imsave(raw_dest_path, img_aug)
#				imsave(mask_dest_path, img_aug)
		counter += 1