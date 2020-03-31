# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:38:07 2019

@author: Daniel
"""

import os, shutil

#dest_path = r'D:\Daniel\Documents\Github\CALFIN Repo\training\data\test'
#root_source_mask_path = r'D:\Daniel\Documents\Github\CALFIN Repo\reprocessing\temp'
#root_source_raw_path = r'D:\Daniel\Documents\Github\CALFIN Repo\reprocessing\images_full\Kangerlussuaq'
#
#for file_name in os.listdir(root_source_mask_path):
#	basename = file_name[0:-9]
#	raw_name = basename + '.png'
#	mask_name = basename + '_mask.png'
#	source_raw_path = os.path.join(root_source_raw_path, raw_name)
#	source_mask_path = os.path.join(root_source_mask_path, mask_name)
#	dest_raw_path = os.path.join(dest_path, raw_name)
#	dest_mask_path = os.path.join(dest_path, mask_name)
#
#	shutil.copy2(source_raw_path, dest_raw_path)
#	shutil.copy2(source_mask_path, dest_mask_path)
	
	
dest_path = r'D:\Daniel\Documents\Github\CALFIN Repo\training\data\test'
root_source_mask_path = r'D:\Daniel\Documents\Github\CALFIN Repo\reprocessing\temp'
root_source_raw_path = r'D:\Daniel\Documents\Github\CALFIN Repo\reprocessing\images_full\Kangerlussuaq'

for file_name in os.listdir(root_source_mask_path):
	basename = file_name[0:-9]
	raw_name = basename + '.png'
	mask_name = basename + '_mask.png'
	source_raw_path = os.path.join(root_source_raw_path, raw_name)
	source_mask_path = os.path.join(root_source_mask_path, mask_name)
	dest_raw_path = os.path.join(dest_path, raw_name)
	dest_mask_path = os.path.join(dest_path, mask_name)

	shutil.copy2(source_raw_path, dest_raw_path)
	shutil.copy2(source_mask_path, dest_mask_path)
	
