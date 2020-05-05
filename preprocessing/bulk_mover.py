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

#root_source_path = r'D:\Daniel\Documents\Github\CALFIN Repo\preprocessing\calvingfrontmachine\CalvingFronts\tif'
#dest_path = root_source_path
#for file_path in glob.glob(os.path.join(root_source_path, '**', 'GRDM')):
#	file_name = os.path.basename(file_path)
##	basename = os.path.splitext(file_name)[0]
##	letter = basename[0]
##	date = basename.split('_')[5]
##	year = date.split('-')[0]
##	domain_path = os.path.sep.join(file_path.split(os.path.sep)[0:-2])
##	year_path = os.path.join(domain_path, year)
##	letter_path = os.path.join(domain_path, letter)
##	if not os.path.exists(year_path):
##		os.mkdir(year_path)
##	if os.path.exists(letter_path):
##		shutil.rmtree(letter_path)
##	dest_file_path = os.path.join(year_path, file_name)
#	print(file_path)
##	shutil.move(file_path, dest_file_path)
#	if len(os.listdir(file_path)) == 0:
#		shutil.rmtree(file_path)
##	source_raw_path = os.path.join(root_source_raw_path, raw_name)
##	source_mask_path = os.path.join(root_source_mask_path, mask_name)
##	dest_raw_path = os.path.join(dest_path, raw_name)
##	dest_mask_path = os.path.join(dest_path, mask_name)
##
##	shutil.copy2(source_raw_path, dest_raw_path)
##	shutil.copy2(source_mask_path, dest_mask_path)