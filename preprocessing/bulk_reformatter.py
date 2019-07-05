# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:38:07 2019

@author: Daniel
"""
import numpy as np
import os, shutil, glob
from skimage.io import imsave, imread

source_path = r'D:\Daniel\Documents\Github\CALFIN Repo\training\data\all'
dest_path = r'D:\Daniel\Documents\Github\CALFIN Repo\training\data\all'

for file_path in glob.glob(source_path + '\\*_mask.png'):
	mask = imread(file_path, as_gray=True)
	print(mask.dtype, mask.max())
	if mask.dtype == np.uint8:
		mask = mask.astype(np.uint16) * 257
	elif mask.dtype == np.float64:
		mask = (mask * 65535).astype(np.uint16)
	print(mask.dtype, mask.max()) 
	imsave(file_path, mask)
				
for file_path in glob.glob(source_path + '\\*B[0-9].png'):
	img = imread(file_path, as_gray=True)
	print(img.dtype, img.max())
	if img.dtype == np.uint8:
		img = img.astype(np.uint16) * 257
		print(img.dtype, img.max())
		imsave(file_path, img)
	elif img.dtype == np.float64:
		img = (img * 65535).astype(np.uint16)
		print(img.dtype, img.max())
		imsave(file_path, img)