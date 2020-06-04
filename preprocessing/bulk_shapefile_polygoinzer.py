# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:38:07 2019

@author: Daniel
"""
import numpy as np
import cv2
import os, shutil, glob, copy, sys
os.environ['GDAL_DATA'] = r'D:\\ProgramData\\Anaconda3\\envs\\cfm\\Library\\share\\gdal' #Ensure crs are exported correctly by gdal/osr/fiona

import rasterio
from scipy.ndimage.morphology import distance_transform_edt
from rasterio import features
from skimage.io import imsave, imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from pyproj import Proj, transform
from shapely.geometry import mapping, Polygon, LineString
from collections import defaultdict
import fiona
from fiona.crs import from_epsg
from dateutil.parser import parse

sys.path.insert(1, '../postprocessing')
from error_analysis import extract_front_indicators
from mask_to_shp import mask_to_polygon_shp
from rasterio.windows import from_bounds

#level 0 should inlcude all subsets (preprocessed)
#Make individual ones, domain ones, and all available
#indivudal ones include QA, tif, and shapefile

def landsat_sort(file_path):
	"""Sorting key function derives date from landsat file path. Also orders manual masks in front of auto masks."""
	file_name_parts = file_path.split(os.path.sep)[-1].split('_')
	if 'validation' in file_name_parts[-1]:
		return file_name_parts[3] + 'a'
	else:
		return file_name_parts[3] + 'b'

def duplicate_prefix_filter(file_list):
	caches = set()
	results = []
	for file_path in file_list:
		file_name = os.path.basename(file_path)
		file_name_parts = file_name.split('_')
		prefix = "_".join(file_name_parts[0:-2])
		# check whether prefix already exists
		if prefix not in caches:
			results.append(file_path)
			caches.add(prefix)
		else:
			print('override:', prefix)
	return results

def window_from_extent(xmin, xmax, ymin, ymax, aff):
	col_start, row_start = ~aff * (xmin, ymax)
	col_stop, row_stop = ~aff * (xmax, ymin)
	return ((int(row_start), int(row_stop)), (int(col_start), int(col_stop)))


def consolidate_shapefiles(source_path_manual, source_path_auto, dest_domain_path, dest_all_path, version, fjord_boundary_path, pred_path):
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
			'Year': 'int',
			'Month': 'int',
			'Day': 'int',
			'ImageID': 'str',
			'GrnlndcN': 'str',
			'OfficialN': 'str',
			'AltName': 'str',
			'RefName': 'str',
			'Author': 'str'
		},
	}
	plt.close('all')
	outProj = Proj('epsg:3413') #3413 (NSIDC Polar Stereographic North)
	crs = from_epsg(3413)
	source_manual_quality_assurance_path = os.path.join(source_path_manual, 'quality_assurance')
	source_auto_quality_assurance_path = os.path.join(source_path_auto, 'quality_assurance')
	output_all_shp_path = os.path.join(dest_all_path, 'termini_1972-2019_calfin_' + version + '.shp')
# 	with fiona.open(output_all_shp_path,
# 		'w',
# 		driver='ESRI Shapefile',
# 		crs=fiona.crs.from_epsg(3413),
# 		schema=schema,
# 		encoding='utf-8') as output_all_shp_file:
	counter = 0
	for domain in os.listdir(source_manual_quality_assurance_path):
		if '.' in domain:
			continue
# 		if '.' in domain:
# 			continue
		if not('Upernavik' in domain or 'Rink-Isbrae' in domain or 'Inngia' in domain or 'Umiammakku' in domain):
			continue
		file_list_manual = glob.glob(os.path.join(source_manual_quality_assurance_path, domain, '*_pred.tif'))
		file_list_auto = glob.glob(os.path.join(source_auto_quality_assurance_path, domain, '*_pred.tif'))
		file_list = file_list_manual + file_list_auto
		file_list.sort(key=landsat_sort)
		fjord_boundary_file_path = os.path.join(fjord_boundary_path, domain + '_fjord_boundaries.tif')
		with rasterio.open(fjord_boundary_file_path) as fjord_boundary_tif:
# 			mask = fjord_boundary_tif.read(1)

#			file_list = duplicate_prefix_filter(file_list)
			for file_path in file_list:
				print(file_path)
				counter += 1
# 				if counter < 5:
# 					plt.show()
 					# continue
				file_name = os.path.basename(file_path)
				file_name_parts = file_name.split('_')
				file_basename = "_".join(file_name_parts[0:-2])
				satellite = file_name_parts[1]
				if satellite.startswith('S'):
					#Astakhov-Chugunov-Astapenko_S1B_EW_GRDM_1SDH_2018-06-26_011542_01536C_EB6F
	#				datatype = file_name_parts[2]
					level = file_name_parts[3]
					date_dashed = file_name_parts[4]
					date_parts = date_dashed.split('-')
					year = date_parts[0]
					month = date_parts[1]
					day = date_parts[2]
					date = date_dashed.replace('-', '')
					orbit = file_name_parts[5]
	#				bandpol = 'hh'
				elif satellite.startswith('L'):
					#Brückner_LC08_L1TP_2015-06-14_232-014_T1_B5_66-1_validation
	#				datatype = file_name_parts[2]
					date_dashed = file_name_parts[3]
					date_parts = date_dashed.split('-')
					year = date_parts[0]
					month = date_parts[1]
					day = date_parts[2]
					date = date_dashed.replace('-', '')
					orbit = file_name_parts[4].replace('-', '')
					level = file_name_parts[5]
	#				bandpol = file_name_parts[6]
					scene_id = landsat_scene_id_lookup(date, orbit, satellite, level)
				else:
					print('Unrecognized sattlelite!')
					continue

# 				if 'validation' in file_path:
# 					source_domain_path = os.path.join(source_path_manual, 'domain')
# 				elif 'results' in file_path:
				source_domain_path = os.path.join(source_path_auto, 'domain')

	#				if not landsat_output_lookup(domain, date, orbit, satellite, level):
	#					print('duplicate pick, continuing:', date, orbit, satellite, level, domain)
	#					continue
				reprocessing_id = file_name_parts[-2][-1]
				old_file_shp_name = file_basename + '_' + reprocessing_id + '_cf.shp'
				old_file_shp_file_path = os.path.join(source_domain_path, domain, old_file_shp_name)

				with fiona.open(old_file_shp_file_path, 'r', encoding='utf-8') as source_shp:
					pred_img_name = file_basename +'_' + file_name_parts[-2] + '_pred.tif'
					pred_img_path = os.path.join(pred_path, domain, pred_img_name)
					with rasterio.open(pred_img_path) as pred_tif:

						mask = fjord_boundary_tif.read(1)
						fjord_bounds = rasterio.transform.array_bounds(mask.shape[0], mask.shape[1], fjord_boundary_tif.transform)
						xMin = fjord_bounds[0]
						yMin = fjord_bounds[1]
						xMax = fjord_bounds[2]
						yMax = fjord_bounds[3]
						# Read croped array
						window=from_bounds(xMin, yMin, xMax, yMax, pred_tif.transform)
						pred_mask = pred_tif.read(2, window=window, out_shape=mask.shape, boundless=True, fill_value=127)
# 						print(pred_mask.shape, fjord_boundary_tif.shape)
#
# 						plt.figure(1)
# 						plt.imshow(pred_mask)

# 						plt.figure(2)
# 						plt.imshow(mask)

						fjord_distances, fjord_indices = distance_transform_edt(mask, return_indices=True)
						polyline_coords = source_shp[0]['geometry']['coordinates']
						endpoint_pixel_coord_1 = rasterio.transform.rowcol(fjord_boundary_tif.transform, polyline_coords[0][0], polyline_coords[0][1])
						endpoint_pixel_coord_2 = rasterio.transform.rowcol(fjord_boundary_tif.transform, polyline_coords[-1][0], polyline_coords[-1][1])
						closest_pixel_1 = [fjord_indices[0][endpoint_pixel_coord_1], fjord_indices[1][endpoint_pixel_coord_1]]
						closest_pixel_2 = [fjord_indices[0][endpoint_pixel_coord_2], fjord_indices[1][endpoint_pixel_coord_2]]
						closest_coord_1 = rasterio.transform.xy(fjord_boundary_tif.transform, closest_pixel_1[0], closest_pixel_1[1])
						closest_coord_2 = rasterio.transform.xy(fjord_boundary_tif.transform, closest_pixel_2[0], closest_pixel_2[1])

						geometry = copy.deepcopy(source_shp[0]['geometry'])
						geometry['coordinates'] = [closest_coord_1] + geometry['coordinates'] + [closest_coord_2]

# 						plt.figure(4)
# 						plt.imshow(fjord_distances)
# 						aff = src.affine
# 						meta = src.meta.copy()
# 						window = window_from_extent(xmin, xmax, ymin, ymax, aff)
# 						# Read croped array
# 						arr = src.read(1, window=window)

# 						img = features.rasterize([(geometry, 0)],
#  							# 						all_touched=True,
#  							out=np.ones(mask.shape) * 255,
#  							out_shape=fjord_boundary_tif.shape,
#  							transform=fjord_boundary_tif.transform)
# 						plt.figure(5)
# 						plt.imshow(img)
						img = features.rasterize([(geometry, 0)],
							# 						all_touched=True,
							out=mask,
							out_shape=fjord_boundary_tif.shape,
							transform=fjord_boundary_tif.transform)
# 						plt.figure(3)
# 						plt.imshow(img)


						#loop through others of same date as long as no dates are in qa_bad
						#use connected components

						# sample random number of pixels in dindices within each component to determine mean value
						nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img.astype(np.uint8), connectivity=4)

						#connectedComponentswithStats yields every seperated component with information on each of them, such as size
						sizes = stats[:,cv2.CC_STAT_AREA]
						ordering = list(reversed(np.argsort(sizes)))

						#for every component in the image, keep it only if it's above min_size
	# 					min_size_floor = output.size * min_size_percentage
						if len(ordering) > 1:
							min_size = sizes[ordering[0]] * 0.1

						#Isolate large components
	# 					largeComponents = np.zeros((output.shape))

						#Store the bounding boxes of components, so they can be isolated and reprocessed further. Default box is entire image.
	# 					bounding_boxes = [[0, 0, image.shape[0], image.shape[1]]]
						#Skip first component, since it's the background color in edge masks
						#Restrict number of components returned depending on limit
	# 					number_returned = 0
						min_size_percentage = 0.025
						min_size_floor = output.size * min_size_percentage
						mean_values = []

# 						plt.figure(str(counter) + '-Pre')
						mask = 255 - mask
# 						plt.imshow(mask)

						new_mask = np.ones(mask.shape) * 255
						window=from_bounds(xMin, yMin, xMax, yMax, fjord_boundary_tif.transform)
# 						print(mean_values)
						for i in range(len(sizes)):
							if sizes[ordering[i]] >= min_size_floor and sizes[ordering[i]] >= min_size:
								mask_indices = output == ordering[i]
								image_component = pred_mask[mask_indices] #get green channel
		# 						largeComponents[mask_indices] = image_component
								largeComponents = np.zeros((pred_mask.shape))
								largeComponents[mask_indices] = 255
# 								plt.figure(str(counter) + '-' + str(i))
# 								plt.imshow(largeComponents)

								mean_value = np.mean(image_component)
								mean_values.append(mean_value)
# 								print(mean_values)
								if mean_value < 128:
									new_mask[mask_indices] = 0


# 						plt.figure(4)
# 						plt.imshow(pred_mask)
						print(mean_values)
# 						plt.figure(5)
# 						plt.figure(str(counter) + '-Final')
# 						plt.imshow(new_mask)


						# replace all values in land/ice mask with correct color
						mask_edge = cv2.Canny(new_mask.astype(np.uint8), 250, 255 * 2) #thresholds = Use diagonals to detect strong edges, then connect anything with at least a single edge
						results_polyline = extract_front_indicators(mask_edge, z_score_cutoff=25.0)
						empty_image = np.zeros(mask_edge.shape)
						if results_polyline is None:
							continue
						polyline_image = np.stack((results_polyline[0][:,:,0] / 255.0, empty_image, empty_image), axis=-1)

# 						plt.figure(str(counter) + '-Polyline')
# 						plt.imshow(polyline_image)

						image_name_base = file_basename
						front_line = results_polyline[1]
						source_tif_path = fjord_boundary_file_path
						dest_root_path = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\mask_extractor'
						mask_to_polygon_shp(image_name_base, front_line, source_tif_path, dest_root_path, domain)

def center(x):
	return x['geometry']['coordinates']

def landsat_output_lookup(domain, date, orbit, satellite, level):
	output_hash_table[domain][date][orbit][satellite][level] += 1
	if output_hash_table[domain][date][orbit][satellite][level] > 1:
		return False
	else:
		return True

def landsat_scene_id_lookup(date, orbit, satellite, level):
	return scene_hash_table[date][orbit][satellite][level]

if __name__ == "__main__":
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
	name_id_dict['Kangerlussuaq'] = 32624
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

	version = "v1.0"
	source_path_manual = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\\calfin_on_calfin_train'
	source_path_auto = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\mask_extractor'
	dest_domain_path = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\upload_production\v1.0\level-1_shapefiles-domain-termini'
	dest_all_path = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\upload_production\v1.0\level-1_shapefiles-greenland-termini'
	fjord_boundary_path = r'D:\Daniel\Documents\Github\CALFIN Repo\training\data\fjord_boundaries_tif'
	pred_path = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\mask_extractor\quality_assurance'
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

	output_hash_table = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))))

	consolidate_shapefiles(source_path_manual, source_path_auto, dest_domain_path, dest_all_path, version, fjord_boundary_path, pred_path)


# for each shapefile, for each date, check if there are no bad QA files, then




