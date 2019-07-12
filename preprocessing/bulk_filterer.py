# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:38:07 2019

@author: Daniel
"""
import numpy as np
import os, shutil, glob
from skimage.io import imsave, imread

source_path = r'C:\Users\Daniel\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\calvingfrontmachine\landsat_raw'
dest_path = r'C:\Users\Daniel\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\calvingfrontmachine\dumpster'
mean_threshold = 65535 * 0.05
for domain in os.listdir(source_path):
	for file_path in glob.glob(os.path.join(source_path, domain, '*B[0-9].png')):
		img = imread(file_path, as_gray=True)
		if img.dtype == np.uint8:
			img = img.astype(np.uint16) * 257
		elif img.dtype == np.float64:
			img = (img * 65535).astype(np.uint16)
			
		img_mean = np.mean(img)
		img_max = img.max()
		img_std = np.std(img)
		if img_max == 0 or img_mean < mean_threshold or img_std < 65:
			print(file_path, ' moving into dumpster fire')
			shutil.move(file_path, os.path.join(dest_path, os.path.basename(file_path)))