# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 00:23:01 2019

@author: Daniel
"""

import os, glob
  
# Function to rename multiple files 
source_path = r'../processing/landsat_raw_temp'
for source_file_path in glob.glob(os.path.join(source_path, '*')): 
	dest_file_path = source_file_path[0:-4] + '_sh.png'
	os.rename(source_file_path, dest_file_path)
