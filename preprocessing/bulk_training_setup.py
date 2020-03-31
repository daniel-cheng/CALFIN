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

source_path = r'D:\Daniel\Documents\Github\CALFIN Repo\processing\landsat_raw'
dest_path = r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\redo "
source_name_path = r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\redo_source"
calendar_path = r'D:\Daniel\Documents\Github\CALFIN Repo\preprocessing\calendars'

domains = sorted(os.listdir(source_path))
domains_count = len(domains)

augs = aug_resize(img_size=1024)
counter = 0
total = 0
removed = 0
for file_name in os.listdir(source_name_path):
	file_name_parts = file_name.split('_')
	domain = file_name_parts[0]
	source_file_path = os.path.join(source_path, domain, file_name)
	

	img = imread(source_file_path, as_gray=True)
	if img.dtype == np.uint8:
		img = img.astype(np.uint16) * 257
	elif img.dtype == np.float64:
		img = (img * 65535).astype(np.uint16)
		
	dat = augs(image=img)
	img_aug = dat['image'] #np.uint15 [0, 65535]
					
	imsave(raw_dest_path, img_aug)
	imsave(mask_dest_path, img_aug)
	counter += 1