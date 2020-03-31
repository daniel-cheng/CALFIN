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
import numpngw

dest_path = r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\redo"
source_path = r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\all"

domains = sorted(os.listdir(source_path))
domains_count = len(domains)

augs = aug_resize(img_size=1024)
counter = 0
total = 0
removed = 0
for file_path in glob.glob(os.path.join(source_path, '*B[0-9].png')):
	file_path_old = file_path
	file_name = os.path.basename(file_path)
	file_name_base = file_name.split('.')[0]
	mask_file_name = file_name_base + '_mask.png'
	source_mask_file_path = os.path.join(source_path, mask_file_name)
	
	if not os.path.exists(source_mask_file_path):
		print('Does not exist:', source_mask_file_path)