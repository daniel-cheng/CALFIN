# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:38:07 2019

@author: Daniel
"""
import numpy as np
import os, shutil, glob
from skimage.io import imsave, imread

def filter_files(source_path, dest_path):
	sh_path = r'D:\Daniel\Documents\Github\CALFIN Repo\processing\landsat_raw_temp'
	threshold = 0.05
	min_threshold = 65535 * threshold
	max_threshold = 65535 - min_threshold
	mean_threshold = 65535 * threshold
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
			percent_bad = (np.sum(img > max_threshold, axis=(0, 1)) + np.sum(img < min_threshold, axis=(0, 1))) / np.size(img)
			if img_max == 0 or img_mean < mean_threshold or img_std < 65 or percent_bad > 0.75:
				print(file_path, ' moving into dumpster fire')
				shutil.move(file_path, os.path.join(dest_path, os.path.basename(file_path)))
				sh_file_path = os.path.join(sh_path, file_path.split(os.path.sep)[-1])
				if os.path.exists(sh_file_path):
					os.remove(sh_file_path)
if __name__ == "__main__":
	source_path = r'D:\Daniel\Documents\Github\CALFIN Repo\preprocessing\calvingfrontmachine\landsat_raw'
	dest_path = r'D:\Daniel\Documents\Github\CALFIN Repo\preprocessing\calvingfrontmachine\dumpster'
	filter_files(source_path, dest_path)