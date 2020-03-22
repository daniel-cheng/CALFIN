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

dest_path = r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\all"
dest_path = r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\train"
source_path = r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\validation"

domains = sorted(os.listdir(source_path))
domains_count = len(domains)

for file_name in os.listdir(source_path):
	try:
		os.remove(os.path.join(dest_path, file_name))
	except:
		print(file_name, 'not found')