# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:38:07 2019

@author: Daniel
"""
import numpy as np
import os, shutil, glob
from skimage.io import imsave, imread



#level 0 should inlcude all subsets (preprocessed)
#Make individual ones, domain ones, and all available
#indivudal ones include QA, tif, and shapefile

def filter_files(source_path, dest_path, match_expression):
	source_quality_assurance_path = os.path.join(source_path, 'quality_assurance')
	source_quality_assurance_bad_path = os.path.join(source_path, 'quality_assurance_bad')
	source_domain_path = os.path.join(source_path, 'domain')
	for domain in os.listdir(source_quality_assurance_path):
		for file_path in glob.glob(os.path.join(source_quality_assurance_path, domain, match_expression)):
			
			dest_domain_path = os.path.join(dest_path, domain)
			if not os.path.exists(dest_domain_path):
				os.mkdir(dest_domain_path)
			file_name = os.path.basename(file_path)
			file_name_parts = file_name.split('_')
			file_basename = "_".join(file_name_parts[0:-2])
			satellite = file_name_parts[1]
			if satellite.startswith('S'):
				#Astakhov-Chugunov-Astapenko_S1B_EW_GRDM_1SDH_2018-06-26_011542_01536C_EB6F
				datatype = file_name_parts[2]
				level = file_name_parts[3]
				date = file_name_parts[4].replace('-', '')
				orbit = file_name_parts[5]
				bandpol = 'hh'
			elif satellite.startswith('L'):
				#Brückner_LC08_L1TP_2015-06-14_232-014_T1_B5_66-1_validation
				datatype = file_name_parts[2]
				date = file_name_parts[3].replace('-', '')
				orbit = file_name_parts[4].replace('-', '')
				level = file_name_parts[5]
				bandpol = file_name_parts[6]
			else:
				print('Unrecognized sattlelite!')
				return
			new_file_basename = "{domain}_{date}_{satellite}_{datatype}_{orbit}_{level}_{bandpol}".format(
				domain=domain, 
				date=date,
				satellite=satellite,
				datatype=datatype,
				orbit=orbit,
				level=level,
				bandpol=bandpol
				)
			
			reprocessing_id = file_name_parts[-2][-1]
			old_file_qa_name = file_basename + '_' + file_name_parts[-2] + '_' + file_name_parts[-1]
			old_file_tif_name = file_basename + '.tif'
			old_file_cpg_name = file_basename + '_' + reprocessing_id + '_cf.cpg'
			old_file_dbf_name = file_basename + '_' + reprocessing_id + '_cf.dbf'
			old_file_prj_name = file_basename + '_' + reprocessing_id + '_cf.prj'
			old_file_shp_name = file_basename + '_' + reprocessing_id + '_cf.shp'
			old_file_shx_name = file_basename + '_' + reprocessing_id + '_cf.shx'
			
			new_file_qa_name = new_file_basename + '_' + reprocessing_id + '_qa.png'
			new_file_tif_name = new_file_basename + '_subset.tif'
			new_file_cpg_name = new_file_basename + '_' + reprocessing_id + '_cf.cpg'
			new_file_dbf_name = new_file_basename + '_' + reprocessing_id + '_cf.dbf'
			new_file_prj_name = new_file_basename + '_' + reprocessing_id + '_cf.prj'
			new_file_shp_name = new_file_basename + '_' + reprocessing_id + '_cf.shp'
			new_file_shx_name = new_file_basename + '_' + reprocessing_id + '_cf.shx'
			
			dest_domain_path = os.path.join(dest_path, domain)
			source_quality_assurance_domain_path = os.path.join(source_quality_assurance_path, domain)
			source_domain_domain_path = os.path.join(source_domain_path, domain)
			
			shutil.copy(os.path.join(source_quality_assurance_domain_path, old_file_qa_name), os.path.join(dest_domain_path, new_file_qa_name))
			shutil.copy(os.path.join(source_domain_domain_path, old_file_tif_name), os.path.join(dest_domain_path, new_file_tif_name))
			shutil.copy(os.path.join(source_domain_domain_path, old_file_cpg_name), os.path.join(dest_domain_path, new_file_cpg_name))
			shutil.copy(os.path.join(source_domain_domain_path, old_file_dbf_name), os.path.join(dest_domain_path, new_file_dbf_name))
			shutil.copy(os.path.join(source_domain_domain_path, old_file_prj_name), os.path.join(dest_domain_path, new_file_prj_name))
			shutil.copy(os.path.join(source_domain_domain_path, old_file_shp_name), os.path.join(dest_domain_path, new_file_shp_name))
			shutil.copy(os.path.join(source_domain_domain_path, old_file_shx_name), os.path.join(dest_domain_path, new_file_shx_name))

#format
#
if __name__ == "__main__":
	source_path_manual = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\mask_extractor'
	source_path_auto = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\production'
	dest_path = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\upload_production\v1.0\level-1\domain-daily'
#	filter_files(source_path_manual, dest_path, '*validation.png')
	filter_files(source_path_auto, dest_path, '*results.png')