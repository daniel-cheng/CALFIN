# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:25:54 2019

@author: Daniel
"""

import os, glob, shutil, tarfile
from zipfile import ZipFile
  
steps = ['landsat_unzip', 'landsat_rename', 'sentinel-1_unzip']
steps = ['senitnel-1_unzip']

if 'landsat_unzip' in steps:
	source = r"D:\Daniel\Documents\Github\CALFIN Repo\downloader\rasters\Landsat\Greenland\zipped"
	unzip_dest = r"D:\Daniel\Documents\Github\CALFIN Repo\downloader\rasters\Landsat\Greenland"
	move_dest = r"D:\Daniel\Documents\Github\CALFIN Repo\downloader\rasters\Landsat\Greenland\zipped"
	dry_run = False
	# domains_to_move = ['Hayes', 'Kangiata-Nunata', 'Kong-Oscar', 'Rink-Isbrae']
	# domains_to_move = ['Helheim', 'Kangerlussuaq']
	domains_to_move = []
	
	# Function to rename multiple files 
	for domain in os.listdir(source):
		# if domain != 'Kangerlussuaq':
		if domain != 'Helheim':
			continue
		#Make unzip domain folder if not existing already
		unzip_domain_path = os.path.join(unzip_dest, domain)
		if not os.path.exists(unzip_domain_path):
			os.mkdir(unzip_domain_path)
			
		#Make move domain folder if not existing already, only if we want to move them
		if domain in domains_to_move:
			move_domain_path = os.path.join(move_dest, domain)
			if not os.path.exists(move_domain_path):
				os.mkdir(move_domain_path)
		
		#For each file in domain folder, unzip to appropiate unzip destination/year.
		for source_path in glob.glob(os.path.join(source, domain, '*.tar.gz')):  
			filename = source_path.split(os.path.sep)[-1]
			basename = filename.split('.')[0]
			basename_parts = basename.split('_')
			
			satellite = basename_parts[0]
			level = basename_parts[1]
			path_row = basename_parts[2]
			date = basename_parts[3]
			processing_date = basename_parts[4]
			collection_number = basename_parts[5]
			tier = basename_parts[6]
			
			year = date[0:4]
			month = date[4:6]
			day = date[6:8]
			
			path = path_row[0:3]
			row = path_row[3:6]
			
			band = ''
			if satellite == 'LC08':
				band = 'B5'
			elif satellite == 'LE07':
				band = 'B4'
			elif satellite == 'LT05':
				band = 'B4'
			elif satellite == 'LT04':
				band = 'B4'
			elif satellite == 'LM05':
				band = 'B4'
			elif satellite == 'LM04':
				band = 'B4'	
			elif satellite == 'LM03':
				band = 'B7'
			elif satellite == 'LM02':
				band = 'B7'
			elif satellite == 'LM01':
				band = 'B7'
			else:
				band = 'B5'
				
			unzip_domain_year_path = os.path.join(unzip_domain_path, year)
			if not os.path.exists(unzip_domain_year_path):
				os.mkdir(unzip_domain_year_path)
			
			dest_basename = '_'.join([satellite, level, year + '-' + month + '-' + day, path + '-' + row, tier])
			bNIR_source_filename = basename + '_' + band + '.TIF'
			bNIR_dest_filename = '_'.join([dest_basename, band + '.TIF'])
			bNIR_temp_dest_file_path = os.path.join(unzip_domain_year_path, bNIR_source_filename)
			bNIR_final_dest_file_path = os.path.join(unzip_domain_year_path, bNIR_dest_filename)
			
			bQA_source_filename = basename + '_BQA.TIF'
			bQA_dest_filename = '_'.join([dest_basename, 'BQA.TIF'])
			bQA_temp_dest_file_path = os.path.join(unzip_domain_year_path, bQA_source_filename)
			bQA_final_dest_file_path = os.path.join(unzip_domain_year_path, bQA_dest_filename)
			
			if not os.path.exists(bNIR_final_dest_file_path) or not os.path.exists(bQA_final_dest_file_path):
				print('Extracting:', source_path, 'to:', bNIR_final_dest_file_path, bQA_final_dest_file_path)
				if dry_run == False:
					try:
						t = tarfile.open(source_path, 'r')
						bNIR_member = t.getmember(bNIR_source_filename)
						bQA_member = t.getmember(bQA_source_filename)
						t.extractall(unzip_domain_year_path, members=[bNIR_member, bQA_member])
						shutil.move(bNIR_temp_dest_file_path, bNIR_final_dest_file_path)
						shutil.move(bQA_temp_dest_file_path, bQA_final_dest_file_path)
						t.close()
						print('Success!')
					except:
						print('Error in tarfile')
						try:
							t.close()
						except:
							print('unable to close tarfile')
						continue
	
				if domain in domains_to_move:
					move_file_path = os.path.join(move_domain_path, filename)
					print(move_file_path)
					if dry_run == False:
						shutil.move(source_path, move_file_path)
			else:
				print('Skipping, already exists:', bNIR_final_dest_file_path, bQA_final_dest_file_path)


if 'landsat_rename' in steps:
	source = r'D:/Daniel/Pictures/CALFIN/test_full/'
	dest = r'D:/Daniel/Pictures/CALFIN/test_all/'
	dry_run = 1
	
	# Function to rename multiple files 
	for domain_path in glob.glob(source + '*'):
		for filename in os.listdir(domain_path):
			source_path = os.path.join(domain_path, filename)
			dest_path = os.path.join(dest, filename)
#			print(source_path, dest_path)
			if dry_run != 0:
				shutil.copy(source_path, dest_path)


if 'senitnel-1_unzip' in steps:
	source = r"D:\Daniel\Documents\Github\CALFIN Repo\downloader\rasters\Sentinel-1\Antarctica\zipped"
	unzip_dest = r"D:\Daniel\Documents\Github\CALFIN Repo\downloader\rasters\Sentinel-1\Antarctica"
	move_dest = r"D:\Daniel\Documents\Github\CALFIN Repo\downloader\rasters\Sentinel-1\Antarctica\zipped"
	dry_run = False
	# domains_to_move = ['Hayes', 'Kangiata-Nunata', 'Kong-Oscar', 'Rink-Isbrae']
	# domains_to_move = ['Helheim', 'Kangerlussuaq']
	domains_to_move = []
	
	# Function to rename multiple files 
	for domain in os.listdir(source):
		# if domain != 'Kangerlussuaq':
		#Make unzip domain folder if not existing already
		unzip_domain_path = os.path.join(unzip_dest, domain)
		if not os.path.exists(unzip_domain_path):
			os.mkdir(unzip_domain_path)
			
		#Make move domain folder if not existing already, only if we want to move them
		if domain in domains_to_move:
			move_domain_path = os.path.join(move_dest, domain)
			if not os.path.exists(move_domain_path):
				os.mkdir(move_domain_path)
		
		#For each file in domain folder, unzip to appropiate unzip destination/year.
		for source_path in glob.glob(os.path.join(source, domain, '*.zip')): 
			filename = source_path.split(os.path.sep)[-1]
			basename = filename.split('.')[0]
			basename_parts = basename.split('_')
			
			satellite = basename_parts[0]
			acquisition_mode = basename_parts[1]
			
			product_type_full = basename_parts[2]
			product_type = product_type_full[0:3]
			resolution_class = product_type_full[3]
			
			polarization_full = basename_parts[3]
			level = polarization_full[0]
			product_class = polarization_full[1]
			polarization = polarization_full[2:]
			
			start_date = basename_parts[4][0:8]
			start_year = start_date[0:4]
			start_month = start_date[4:6]
			start_day = start_date[6:8]
			
			end_date = basename_parts[5][0:8]
			end_year = end_date[0:4]
			end_month = end_date[4:6]
			end_day = end_date[6:8]
			
			orbit_number = basename_parts[6]
			mission_data_take_id = basename_parts[7]
			product_id = basename_parts[8]
			
			unzip_domain_year_path = os.path.join(unzip_domain_path, start_year)
			if not os.path.exists(unzip_domain_year_path):
				os.mkdir(unzip_domain_year_path)
			
			dest_basename = '_'.join([satellite, acquisition_mode, product_type_full, polarization_full, start_year + '-' + start_month + '-' + start_day, orbit_number, mission_data_take_id, product_id])
			bNIR_source_filename = basename + '.tiff'
			bNIR_dest_filename = dest_basename + '.tiff'
			bNIR_final_dest_file_path = os.path.join(unzip_domain_year_path, bNIR_dest_filename)
			
			if not os.path.exists(bNIR_final_dest_file_path):
				if dry_run == False:
#					try:
					print('Extracting:', source_path, 'to:', bNIR_final_dest_file_path)
					with ZipFile(source_path, 'r') as zipObj:
						listOfiles = zipObj.namelist()
						for filepath in listOfiles:
							if filepath.endswith('.tiff'):
								if 'ew-grd-hh' in filepath:
									print(filepath)
									bNIR_temp_dest_dir = os.path.join(unzip_domain_year_path, 'temp')
									zipObj.extract(filepath, bNIR_temp_dest_dir)
									bNIR_temp_dest_file_path = os.path.join(bNIR_temp_dest_dir, filepath)
									shutil.move(bNIR_temp_dest_file_path, bNIR_final_dest_file_path)
									shutil.rmtree(bNIR_temp_dest_dir)
					print('Success!')
#					except:
#						print('Error in zipfile')
#						try:
#							t.close()
#						except:
#							print('unable to close zipfile')
#						continue
	
				if domain in domains_to_move:
					move_file_path = os.path.join(move_domain_path, filename)
					print(move_file_path)
					if dry_run == False:
						shutil.move(source_path, move_file_path)
			else:
				print('Skipping, already exists:', bNIR_final_dest_file_path)