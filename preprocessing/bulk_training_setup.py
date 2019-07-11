# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:38:07 2019

@author: Daniel
"""
import numpy as np
import os, shutil, glob
from skimage.io import imsave, imread

source_path = r'C:\Users\Daniel\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\calvingfrontmachine\landsat_raw'
dest_path = r'D:\Daniel\Documents\Github\CALFIN Repo\reprocessing\images_1024'

domains = os.listdir(source_path)
domains_count = len(domains)
years = list(range(1972, 2020))
years_count = len(years)
monthly_counts = np.zeros((domains_count, years_count, 12))
yearly_counts = np.zeros((domains_count, years_count))

for i in domains:
	for file_path in glob.glob(os.path.join(source_path, domain, '*B[0-9].png')):
		name = os.path.basename(file_path)
		name_parts = name.split('_')
		date = name_parts[3].split('-')
		year = int(date[0])
		month = int(date[2])
		
		monthly_counts[
		Dietrichson_LC08_L1TP_2014-03-08_022-006_T1_B5_mask
		
		
		img = imread(file_path, as_gray=True)
		if img.dtype == np.uint8:
			img = img.astype(np.uint16) * 257
		elif img.dtype == np.float64:
			img = (img * 65535).astype(np.uint16)
			
		img_mean = np.mean(img)
		img_max = img.max()
		img_std = np.std(img)
		if img_max == 0 or img_mean < 655 or img_std < 65:
			print(file_path, ' moving into dumpster fire')
			shutil.move(file_path, os.path.join(dest_path, os.path.basename(file_path)))