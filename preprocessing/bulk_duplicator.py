# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:38:07 2019

@author: Daniel
"""

import os, shutil

source_path = r'C:\Users\Daniel\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\calvingfrontmachine\landsat_raw'
dest_path = r'D:\Daniel\Documents\Github\CALFIN Repo\reprocessing\images_full'
domains = ['Kangerlussuaq']

for domain in domains:
	domain_source_path = os.path.join(source_path, domain)
	domain_dest_path = os.path.join(dest_path, domain)
	if not os.path.exists(domain_dest_path):
		os.mkdir(domain_dest_path)
	for file_name in os.listdir(domain_source_path):
		basename = file_name[0:-4]
		raw_name = basename + '.png'
		mask_name = basename + '_mask.png'
		source_path = os.path.join(domain_source_path, file_name)
		dest_raw_path = os.path.join(domain_dest_path, raw_name)
		dest_mask_path = os.path.join(domain_dest_path, mask_name)

		shutil.copy2(source_path, dest_raw_path)
		shutil.copy2(source_path, dest_mask_path)