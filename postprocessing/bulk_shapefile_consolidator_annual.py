# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:38:07 2019

@author: Daniel
"""

import os, shutil, glob
import numpy as np
import cv2
os.environ['GDAL_DATA'] = r'D:\ProgramData\Anaconda3\envs\cfm\Library\share\gdal' #Ensure crs are exported correctly by gdal/osr/fiona

from skimage.io import imsave, imread
from scipy.spatial import KDTree
from pyproj import Proj, transform
from shapely.geometry import mapping, Polygon, LineString
from collections import defaultdict
import fiona
from fiona.crs import from_epsg
from dateutil.parser import parse
from osgeo import gdal, osr

#level 0 should inlcude all subsets (preprocessed)
#Make individual ones, domain ones, and all available
#indivudal ones include QA, tif, and shapefile

def consolidate_shapefiles(source_path_manual, source_path_auto, dest_domain_path, dest_all_path, domain_path, version, shp_type):
    schema = {
        'geometry': shp_type,
        'properties': {
            'GlacierID': 'int',
            'Center_X': 'float',
            'Center_Y': 'float',
            'Latitude': 'float',
            'Longitude': 'float',
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
    schema2 = {
        'geometry': 'LineString',
        'properties': {
            'GlacierID': 'int',
            'Center_X': 'float',
            'Center_Y': 'float',
            'Latitude': 'float',
            'Longitude': 'float',
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
    latlongProj = Proj('epsg:4326') #4326 (WGS 84)
    source_auto_qa_path = os.path.join(source_path_auto, 'quality_assurance')
    
    if shp_type == 'LineString':
        suffix = '_' + version + '.shp'
    elif shp_type == 'Polygon':
        suffix = '_closed_' + version + '.shp'
    else:
        raise ValueError('Unrecognized shp_type (should be "line" or "polygon"):', shp_type)
    input_all_shp_path = os.path.join(dest_all_path, 'termini_1972-2019_Greenland_closed_' + version + '.shp')
    output_all_shp_path = os.path.join(dest_all_path, 'termini_1972-2019_Greenland_annual_' + version + '.shp')
    domain_dates = defaultdict(lambda: defaultdict(str))
    with fiona.open(output_all_shp_path, 
            'w', 
            driver='ESRI Shapefile', 
            crs=fiona.crs.from_epsg(3413), 
            schema=schema2, 
            encoding='utf-8') as output_all_shp_file:
        with fiona.open(input_all_shp_path, 
            'r', 
            driver='ESRI Shapefile', 
            crs=fiona.crs.from_epsg(3413), 
            schema=schema, 
            encoding='utf-8') as input_all_shp_file:
            for feature in input_all_shp_file:
                date = feature['properties']['Date']
                domain = feature['properties']['RefName']
                year = date.split('-')[0]
                if year not in domain_dates[domain] or domain_dates[domain][year] == date:
                    domain_dates[domain][year] = date
                    new_geometry = {'type': 'LineString', 'coordinates': feature["geometry"]["coordinates"][0]}
                    output_all_shp_file.write({"properties": feature["properties"], 
                                               "geometry": new_geometry})
            return domain_dates
        

def center(x):
    return x['geometry']['coordinates']

def epsg_from_domain(domain_path, domain):
    """Returns the epsg code as an integer, given the domain shpaefile path and the domain name."""
    domain_prj_path = os.path.join(domain_path, domain + '.prj')
    prj_txt = open(domain_prj_path, 'r').read()
    srs = osr.SpatialReference()
    srs.ImportFromESRI([prj_txt])
    srs.AutoIdentifyEPSG()
    return srs.GetAuthorityCode(None)

def landsat_output_lookup(domain, date, orbit, satellite, level):
    output_hash_table[domain][date][orbit][satellite][level] += 1
    if output_hash_table[domain][date][orbit][satellite][level] > 1:
        return False
    else:
        return True

def scene_id_lookup(file_name_parts):
    satellite = file_name_parts[1]
    if satellite.startswith('S'):
        #Astakhov-Chugunov-Astapenko_S1B_EW_GRDM_1SDH_2018-06-26_011542_01536C_EB6F
        # datatype = file_name_parts[2]
        level = file_name_parts[3]
        date_dashed = file_name_parts[4]
        date = date_dashed.replace('-', '')
        orbit = file_name_parts[5]
        # bandpol = 'hh'
    elif satellite.startswith('L'):
        #Brückner_LC08_L1TP_2015-06-14_232-014_T1_B5_66-1_validation
        # datatype = file_name_parts[2]
        date_dashed = file_name_parts[3]
        date = date_dashed.replace('-', '')
        orbit = file_name_parts[4].replace('-', '')
        level = file_name_parts[5]
        # bandpol = file_name_parts[6]
        scene_id = scene_hash_table[date][orbit][satellite][level]
    else:
        raise ValueError('Unrecognized sattelite!')
                
    return satellite, date_dashed, scene_id

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

def get_file_lists(source_path_manual, source_path_auto, domain, shp_type):
    """Converts file lists into date indexed dictionaries."""
    source_manual_qa_path = os.path.join(source_path_manual, 'quality_assurance')
    source_auto_qa_path = os.path.join(source_path_auto, 'quality_assurance')
    source_manual_qa_bad_path = os.path.join(source_path_manual, 'quality_assurance_bad')
    source_auto_qa_bad_path = os.path.join(source_path_auto, 'quality_assurance_bad')
    
    if shp_type == 'LineString':
        extension = '_overlay_front.png'
    elif shp_type == 'Polygon':
        extension = '_overlay_polygon.png'
    
    file_list_manual = glob.glob(os.path.join(source_manual_qa_path, domain, '*' + extension))
    file_list_auto = glob.glob(os.path.join(source_auto_qa_path, domain, '*' + extension))
    file_list_bad_manual = glob.glob(os.path.join(source_manual_qa_bad_path, domain, '*' + extension))
    file_list_bad_auto = glob.glob(os.path.join(source_auto_qa_bad_path, domain, '*' + extension))
    
    file_list_bad = file_list_bad_manual + file_list_bad_auto
    file_list_bad.sort(key=landsat_sort)
    file_list_bad = [os.path.basename(x).split('_overlay')[0] for x in file_list_bad]
    file_list = file_list_manual + file_list_auto
    file_list.sort(key=landsat_sort)
    
    return file_list, file_list_bad

def get_file_paths(file_path, file_name, domain, shp_type):
    if 'mask_extractor' in file_path:
        source_domain_path = os.path.join(source_path_manual, 'domain')
    elif 'production' in file_path:
        source_domain_path = os.path.join(source_path_auto, 'domain')
    else:
        raise ValueError('Neither "mask_extractor" nor "production" found in source path:', file_path)
    
    file_basename = file_name.split('_overlay')[0]
    if shp_type == 'LineString':
        file_basename_parts = file_basename.split('_')
        new_file_basename = '_'.join(file_basename_parts[0:-1]) #strip away processing id
        reprocessing_id = file_basename_parts[-1][-1] #isolate reprocessing id
        old_file_shp_name = new_file_basename + '_' + reprocessing_id + '_cf.shp'
    elif shp_type == 'Polygon':
        old_file_shp_name = file_basename + '_cf_closed.shp'
    old_file_shp_file_path = os.path.join(source_domain_path, domain, old_file_shp_name)
    
    return source_domain_path, old_file_shp_file_path

def write_feature(coords, inProj, outProj, latlongProj, file_name_parts, file_path, output_all_shp_file, suffix, shp_type, schema):
    #Simplify coords within error tolerance to reduce output size
    coords_array = np.float32([coords])
    error_tolerance = 0.5 #error tolerance in 3413 meters
    closed = shp_type == 'Polygon'
    approx = cv2.approxPolyDP(coords_array, error_tolerance, closed)
    x = approx[:,:,0].flatten()
    y = approx[:,:,1].flatten()
    x2, y2 = transform(inProj, outProj, x, y)
    polyline = np.stack((x2, y2), axis=-1)
    polyline_center = np.mean(polyline, axis=0)
    latitude, longitude = transform(outProj, latlongProj, polyline_center[0], polyline_center[1])
    
    closest_glacier = centers_kdtree.query(polyline_center, k=3)
    for i in range(0, 3):
        closest_feature = list(glacierIds)[closest_glacier[1][i]]
        closest_feature_id = closest_feature['properties']['GlacierID']
        closest_feature_reference_name = closest_feature['properties']['RefName']
        closest_feature_greenlandic_name =  closest_feature['properties']['GrnlndcNam']
        closest_feature_official_name =  closest_feature['properties']['Official_n']
        closest_feature_alt_name =  closest_feature['properties']['AltName']
        if closest_feature_reference_name is None:
            print('No reference name, rechoosing...! id:', closest_feature_id)
            continue
        if closest_feature_greenlandic_name is None:
            closest_feature_greenlandic_name = ''
        if closest_feature_official_name is None:
            closest_feature_official_name = ''
        if closest_feature_alt_name is None:
            closest_feature_alt_name = ''
        break
    
    ref_name = closest_feature_reference_name.replace(' ','-')
    output_domain_shp_path = os.path.join(dest_domain_path, 'termini_1972-2019_' + ref_name + suffix)
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
        
        satellite, date_dashed, scene_id = scene_id_lookup(file_name_parts)
        date_parsed = parse(date_dashed)
        date_cutoff = parse('2003-05-31')
        if satellite == 'LE07' and date_parsed > date_cutoff:
            if 'mask_extractor' in file_path:
                qual_flag = 3
            elif 'production' in file_path:
                qual_flag = 13
        else:
            if 'mask_extractor' in file_path:
                qual_flag = 0
            elif 'production' in file_path:
                qual_flag = 10
        
        if shp_type == 'LineString':
            geometry = mapping(LineString(polyline))
        elif shp_type == 'Polygon':
            geometry = mapping(Polygon(polyline))
        
        sequence_id = len(output_domain_shp_file)
        print(closest_feature_reference_name, closest_feature_id, sequence_id)
        output_data = {
            'geometry': geometry,
            'properties': {
                'GlacierID': closest_feature_id,
                'Center_X': float(polyline_center[0]),
                'Center_Y': float(polyline_center[1]),
                'Latitude': float(latitude),
                'Longitude': float(longitude),  
                'QualFlag': qual_flag,
                'Satellite': satellite,
                'Date': date_dashed,
                'ImageID': scene_id,
                'GrnlndcN': closest_feature_greenlandic_name,
                'OfficialN': closest_feature_official_name,
                'AltName': closest_feature_alt_name,
                'RefName': closest_feature_reference_name,
                'Author': 'Cheng_D'},
        }
        output_domain_shp_file.write(output_data)
        output_all_shp_file.write(output_data)

if __name__ == "__main__":    
    version = "v1.0"
    all_scenes_path = r"../downloader/scenes/all_scenes.txt"
    source_path_manual = r'../outputs\mask_extractor'
    source_path_auto = r'../outputs/production_staging'
    dest_domain_path = r'../outputs/upload_production/v1.0/level-1_shapefiles-domain-termini'
    dest_all_path = r'../outputs/upload_production/v1.0/level-1_shapefiles-greenland-termini'
    domain_path = r'../preprocessing/domains'
    
    glacierIds = fiona.open(r'../postprocessing/GlacierIDsRef.shp', 'r', encoding='utf-8')
    glacier_centers = np.array(list(map(center, glacierIds)))
    centers_kdtree = KDTree(glacier_centers)
    shp_types = 'Polygon' #'Polygon', 'LineString'
    
    with open(r"../downloader\scenes/all_scenes.txt", 'r') as scenes_file:
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
    
    
    domain_dates = consolidate_shapefiles(source_path_manual, source_path_auto, dest_domain_path, dest_all_path, domain_path, version, shp_types)
