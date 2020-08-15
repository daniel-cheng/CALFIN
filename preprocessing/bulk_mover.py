# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:38:07 2019

@author: Daniel
"""

import os, shutil, glob

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

root_source_path = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\mask_extractor\quality_assurance'
root_dest_path = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\mask_extractor\quality_assurance_bad'
total = 0
total_bad = 0
for domain in os.listdir(root_source_path):
	if '.' in domain:
		continue
	dest_path = os.path.join(root_dest_path, domain)
	if os.path.exists(dest_path):
		ref_list = os.listdir(dest_path)
		for file_path in glob.glob(os.path.join(root_source_path, domain, '*_results.png')):
			file_name = os.path.basename(file_path)
			basename = os.path.splitext(file_name)[0]
			stripped_basename = basename[0:-8]
		#	letter = basename[0]
		# Upernavik-NE_LT05_L1TP_1987-07-26_016-008_T1_B4_21680-1_results.png
			# date = basename.split('_')[5]
		#	year = date.split('-')[0]
		#	domain_path = os.path.sep.join(file_path.split(os.path.sep)[0:-2])
		#	year_path = os.path.join(domain_path, year)
		#	letter_path = os.path.join(domain_path, letter)
			total += 1
			if file_name in ref_list:
				print(file_path)
				total_bad += 1
				
	# 			for file_group_path in glob.glob(os.path.join(root_source_path, domain, stripped_basename + '*')):
	# 				file_group_name = os.path.basename(file_group_path)
	# 				dest_file_path = os.path.join(dest_path, file_group_name)
	# 				shutil.move(file_group_path, dest_file_path)
	# 			for file_group_path in glob.glob(os.path.join(root_source_path, domain, stripped_basename + '*')):
	# 				file_group_name = os.path.basename(file_group_path)
	# 				dest_file_path = os.path.join(dest_path, file_group_name)
	# 				shutil.move(file_group_path, dest_file_path)
		#	if not os.path.exists(year_path):
		#		os.mkdir(year_path)
		#	if os.path.exists(letter_path):
		#		shutil.rmtree(letter_path)
		#	dest_file_path = os.path.join(year_path, file_name)
			
		#	shutil.move(file_path, dest_file_path)
		#	if len(os.listdir(file_path)) == 0:
		#		shutil.rmtree(file_path)
		#	source_raw_path = os.path.join(root_source_raw_path, raw_name)
		#	source_mask_path = os.path.join(root_source_mask_path, mask_name)
		#	dest_raw_path = os.path.join(dest_path, raw_name)
		#	dest_mask_path = os.path.join(dest_path, mask_name)
		#
		#	shutil.copy2(source_raw_path, dest_raw_path)
		#	shutil.copy2(source_mask_path, dest_mask_path)
print(total_bad, total)