# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:38:07 2019

@author: Daniel
"""
import numpy as np
import os, shutil, glob
os.environ['GDAL_DATA'] = r'D:\\ProgramData\\Anaconda3\\envs\\cfm\\Library\\share\\gdal' #Ensure crs are exported correctly by gdal/osr/fiona

from skimage.io import imsave, imread
from scipy.spatial import KDTree
from pyproj import Proj, transform
from shapely.geometry import mapping, Polygon, LineString
from collections import defaultdict
import fiona
from fiona.crs import from_epsg

#level 0 should inlcude all subsets (preprocessed)
#Make individual ones, domain ones, and all available
#indivudal ones include QA, tif, and shapefile

name_id_dict = dict()
name_id_dict['Akullikassaap'] = 32621
name_id_dict['Alangorssup'] = 32621
name_id_dict['Alanngorliup'] = 32621
name_id_dict['Brückner'] = 32624
name_id_dict['Christian-IV'] = 32624
name_id_dict['Cornell'] = 32621
name_id_dict['Courtauld'] = 32624
name_id_dict['Dietrichson'] = 32621
name_id_dict['Docker-Smith'] = 32621
name_id_dict['Eqip'] = 32621
name_id_dict['Fenris'] = 32624
name_id_dict['Frederiksborg'] = 32624
name_id_dict['Gade'] = 32621
name_id_dict['Glacier-de-France'] = 32624
name_id_dict['Hayes'] = 32621
name_id_dict['Heim'] = 32624
name_id_dict['Helheim'] = 32624
name_id_dict['Hutchinson'] = 32624
name_id_dict['Illullip'] = 32621
name_id_dict['Inngia'] = 32621
name_id_dict['Issuusarsuit'] = 32621
name_id_dict['Jakobshavn'] = 32621
name_id_dict['Kakiffaat'] = 32621
name_id_dict['Kangerdluarssup'] = 32621
name_id_dict['Kangerlussuaq'] = 32621
name_id_dict['Kangerlussuup'] = 32621
name_id_dict['Kangiata-Nunaata'] = 32621
name_id_dict['Kangilerngata'] = 32621
name_id_dict['Kangilinnguata'] = 32621
name_id_dict['Kangilleq'] = 32621
name_id_dict['Kjer'] = 32621
name_id_dict['Kong-Oscar'] = 32621
name_id_dict['Kælvegletscher'] = 32624
name_id_dict['Lille'] = 32621
name_id_dict['Midgård'] = 32624
name_id_dict['Morell'] = 32621
name_id_dict['Nansen'] = 32621
name_id_dict['Narsap'] = 32621
name_id_dict['Nordenskiold'] = 32621
name_id_dict['Nordfjord'] = 32624
name_id_dict['Nordre-Parallelgletsjer'] = 32624
name_id_dict['Nunatakassaap'] = 32621
name_id_dict['Nunatakavsaup'] = 32621
name_id_dict['Perlerfiup'] = 32621
name_id_dict['Polaric'] = 32624
name_id_dict['Qeqertarsuup'] = 32621
name_id_dict['Rink-Gletsjer'] = 32621
name_id_dict['Rink-Isbrae'] = 32621
name_id_dict['Rosenborg'] = 32624
name_id_dict['Saqqarliup'] = 32621
name_id_dict['Sermeq-Avannarleq-69'] = 32621
name_id_dict['Sermeq-Avannarleq-70'] = 32621
name_id_dict['Sermeq-Avannarleq-73'] = 32621
name_id_dict['Sermeq-Kujalleq-70'] = 32621
name_id_dict['Sermeq-Kujalleq-73'] = 32621
name_id_dict['Sermeq-Silarleq'] = 32621
name_id_dict['Sermilik'] = 32621
name_id_dict['Sorgenfri'] = 32624
name_id_dict['Steenstrup'] = 32621
name_id_dict['Store'] = 32621
name_id_dict['Styrtegletsjer'] = 32624
name_id_dict['Sverdrup'] = 32621
name_id_dict['Søndre-Parallelgletsjer'] = 32624
name_id_dict['Umiammakku'] = 32621
name_id_dict['Upernavik-NE'] = 32621
name_id_dict['Upernavik-SE'] =  32621

def landsat_sort(file_path):
	"""Sorting key function derives date from landsat file path."""
	return file_path.split(os.path.sep)[-1].split('_')[3]

def consolidate_shapefiles(source_path, dest_path, mask_type, match_expression):
	schema = {
		'geometry': 'LineString',
		'properties': {
			'GlacierID': 'int',
			'Center_X': 'float',
			'Center_Y': 'float',
			'Sequence#': 'int',
			'QualFlag': 'int',
			'Satellite': 'str',
			'Date': 'str',
			'ImageID': 'str',
			'GrnlndcN': 'str',
			'OfficialN': 'str',
			'AltName': 'str',
			'RefName': 'str',
			'Author': 'str'
		},
	}
	
	outProj = Proj('epsg:3413') #3413 (NSIDC Polar Stereographic North)
	crs = from_epsg(3413)
	source_quality_assurance_path = os.path.join(source_path, 'quality_assurance')
	for domain in os.listdir(source_quality_assurance_path):
		source_quality_assurance_path = os.path.join(source_path, 'quality_assurance')
		source_domain_path = os.path.join(source_path, 'domain')
		file_list = glob.glob(os.path.join(source_quality_assurance_path, domain, match_expression))
		file_list.sort(key=landsat_sort)
		for file_path in file_list:
			file_name = os.path.basename(file_path)
			file_name_parts = file_name.split('_')
			file_basename = "_".join(file_name_parts[0:-2])
			satellite = file_name_parts[1]
			if satellite.startswith('S'):
				#Astakhov-Chugunov-Astapenko_S1B_EW_GRDM_1SDH_2018-06-26_011542_01536C_EB6F
#				datatype = file_name_parts[2]
				level = file_name_parts[3]
				date_dashed = file_name_parts[4]
				date = date_dashed.replace('-', '')
				orbit = file_name_parts[5]
#				bandpol = 'hh'
			elif satellite.startswith('L'):
				#Brückner_LC08_L1TP_2015-06-14_232-014_T1_B5_66-1_validation
#				datatype = file_name_parts[2]
				date_dashed = file_name_parts[3]
				date = date_dashed.replace('-', '')
				orbit = file_name_parts[4].replace('-', '')
				level = file_name_parts[5]
#				bandpol = file_name_parts[6]
				scene_id = landsat_scene_id_lookup(date, orbit, satellite, level)
			else:
				print('Unrecognized sattlelite!')
				continue
			
			reprocessing_id = file_name_parts[-2][-1]
			old_file_shp_name = file_basename + '_' + reprocessing_id + '_cf.shp'
			old_file_shp_file_path = os.path.join(source_domain_path, domain, old_file_shp_name)
			with fiona.open(old_file_shp_file_path, 'r', encoding='utf-8') as source_shp:
				coords = np.array(list(source_shp)[0]['geometry']['coordinates'])
				inProj = Proj('epsg:' + str(name_id_dict[domain]), preserve_units=True) #32621 or 32624 (WGS 84 / UTM zone 21N or WGS 84 / UTM zone 24N)
				x = coords[:,0]
				y = coords[:,1]
				x2, y2 = transform(inProj, outProj, x, y)
				polyline = np.stack((x2, y2), axis=-1)
				polyline_center = np.mean(polyline, axis=0)
				
				closest_glacier = centers_kdtree.query(polyline_center)
				closest_feature = list(glacierIds)[closest_glacier[1]]
				closest_feature_id = closest_feature['properties']['GlacierID']
				closest_feature_reference_name = closest_feature['properties']['RefName']
				closest_feature_greenlandic_name =  closest_feature['properties']['GrnlndcNam']
				closest_feature_official_name =  closest_feature['properties']['Official_n']
				closest_feature_alt_name =  closest_feature['properties']['AltName']
				if closest_feature_reference_name is None:
					print('No reference name! id:', closest_feature_id)
					continue
				if closest_feature_greenlandic_name is None:
					closest_feature_greenlandic_name = ''
				if closest_feature_official_name is None:
					closest_feature_official_name = ''
				if closest_feature_alt_name is None:
					closest_feature_alt_name = ''
				
				output_domain_shp_path = os.path.join(dest_path, 'termini_1972-2019_' + closest_feature_reference_name.replace(' ','-') + '.shp')
				if not os.path.exists(output_domain_shp_path):
					mode = 'w'
				else:
					mode = 'a'
					
				with fiona.open(output_domain_shp_path, 
					mode, 
					driver='ESRI Shapefile', 
					crs=fiona.crs.from_epsg(3413), 
					schema=schema, 
					encoding='utf-8') as output_domain_shp_file:
					
					if mask_type == 'manual':
						qual_flag = 0
					elif mask_type == 'auto':
						qual_flag = 10
					sequence_id = len(output_domain_shp_file)
					print(closest_feature_reference_name, closest_feature_id, sequence_id)
					output_domain_shp_file.write({
						'geometry': mapping(LineString(polyline)),
						'properties': {
							'GlacierID': closest_feature_id,
							'Center_X': float(polyline_center[0]),
							'Center_Y': float( polyline_center[1]),
							'Sequence#': sequence_id,
							'QualFlag': qual_flag,
							'Satellite': satellite,
							'Date': date_dashed,
							'ImageID': scene_id,
							'GrnlndcN': closest_feature_greenlandic_name,
							'OfficialN': closest_feature_official_name,
							'AltName': closest_feature_alt_name,
							'RefName': closest_feature_reference_name,
							'Author': 'Cheng_D'},
					})

def center(x):
	return x['geometry']['coordinates']


def landsat_scene_id_lookup(date, orbit, satellite, level):
	return scene_hash_table[date][orbit][satellite][level]

if __name__ == "__main__":
	source_path_manual = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\mask_extractor'
	source_path_auto = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\production'
	source_path = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\upload_production\v1.0\level-1\domain-daily'
	dest_path = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\upload_production\v1.0\level-1\domain-full'
	
	glacierIds = fiona.open(r'D:\Daniel\Downloads\GlacierIDs\GlacierIDsRef.shp', 'r', encoding='utf-8')
	glacier_centers = np.array(list(map(center, glacierIds)))
	centers_kdtree = KDTree(glacier_centers)
	
	
	with open(r"D:\Daniel\Documents\Github\CALFIN Repo\downloader\scenes\all_scenes.txt", 'r') as scenes_file:
		scene_list = scenes_file.readlines()
		scene_hash_table = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(str))))
		for scene in scene_list:
			scene = scene.split('\t')[0]
			scene_parts = scene.split('_')
			satellite = scene_parts[0]
			orbit = scene_parts[2]
			date = scene_parts[3]
			level = scene_parts[6]
			if date in scene_hash_table and orbit in scene_hash_table[date] and satellite in scene_hash_table[date][orbit] and level in scene_hash_table[date][orbit][satellite]:
				print('hash collision:', scene.split()[0], scene_hash_table[date][orbit][satellite][level].split()[0])
			else:
				scene_hash_table[date][orbit][satellite][level] = scene
				
	consolidate_shapefiles(source_path_manual, dest_path, 'manual', '*validation.png')
	consolidate_shapefiles(source_path_auto, dest_path, 'auto', '*results.png')
	
	