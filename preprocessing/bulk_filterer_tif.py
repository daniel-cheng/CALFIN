# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:38:07 2019

@author: Daniel
"""
import numpy as np
import os, shutil, glob
from skimage.io import imsave, imread

def filter_files(source_path, reference_path, dest_path):
	reference_files = os.listdir(reference_path)
	for domain in os.listdir(source_path):
		for file_path in glob.glob(os.path.join(source_path, domain, '**/*B[0-9].tif')):
			file_name = file_path.split(os.path.sep)[-1]
			base_name = file_name.split('.')[0]
			png_name = base_name + '.png'
			if png_name in reference_files:
				print('moving:', png_name)
				shutil.move(file_path, os.path.join(dest_path, file_name))


if __name__ == "__main__":
	source_path = r'../../CALFIN Repo Intercomp/preprocessing/tif'
	reference_path = r'../training/data/train_yara'
	dest_path = r'../../CALFIN Repo Intercomp/preprocessing/train'
	filter_files(source_path, reference_path, dest_path)
	
	source_path = r'../../CALFIN Repo Intercomp/preprocessing/tif'
	reference_path = r'../training/data/validation_yara'
	dest_path = r'../../CALFIN Repo Intercomp/preprocessing/validation'
	filter_files(source_path, reference_path, dest_path)