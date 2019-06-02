# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 00:23:01 2019

@author: Daniel
"""

import os, glob
  
# Function to rename multiple files 
for src in glob.glob(r'D:\Daniel\Pictures\CALFIN\train\train_partial\*_in*'): 
	dst = src.replace("_in", "")
	os.rename(src, dst)
