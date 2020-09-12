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
from dateutil.parser import parse
from osgeo import gdal, osr

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

def consolidate_shapefiles(source_path_manual, source_path_auto, dest_domain_path, dest_all_path, domain_path, version):
    schema = {
        'geometry': 'LineString',
        'properties': {
            'GlacierID': 'int',
            'Center_X': 'float',
            'Center_Y': 'float',
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
    source_manual_quality_assurance_path = os.path.join(source_path_manual, 'quality_assurance')
    source_auto_quality_assurance_path = os.path.join(source_path_auto, 'quality_assurance')
    source_manual_quality_assurance_bad_path = os.path.join(source_path_manual, 'quality_assurance_bad')
    source_auto_quality_assurance_bad_path = os.path.join(source_path_auto, 'quality_assurance_bad')
    output_all_shp_path = os.path.join(dest_all_path, 'termini_1972-2019_calfin_manual_' + version + '.shp')
    # domains = 
    domains = ['Qeqertarsuup', 'Kakiffaat', 'Nunatakavsaup', 'Alangorssup', 'Akullikassaap', 'Upernavik-NE', 'Upernavik-NW',
               'Upernavik-SE', 'Inngia', 'Umiammakku', 'Rink-Isbrae', 'Kangerlussuup', 
               'Kangerdluarssup', 'Perlerfiup', 'Sermeq-Silarleq', 'Kangilleq', 'Sermilik', 'Lille', 'Store']
    # domains = ['Jakobshavn', 'Sermeq-Avannarleq-69', 'Petermann']
    # domains = ['Umiammakku']
    with fiona.open(output_all_shp_path, 
        'w', 
        driver='ESRI Shapefile', 
        crs=fiona.crs.from_epsg(3413), 
        schema=schema, 
        encoding='utf-8') as output_all_shp_file:
        # domains =  os.listdir(source_manual_quality_assurance_path)
        for domain in os.listdir(source_auto_quality_assurance_path):
            if '.' in domain:
                continue
            # if domain not in domains:
            #     continue
            file_list_manual = glob.glob(os.path.join(source_manual_quality_assurance_path, domain, '*_overlay_front.png'))
            file_list_auto = glob.glob(os.path.join(source_auto_quality_assurance_path, domain, '*_overlay_front.png'))
            file_list_bad_manual = glob.glob(os.path.join(source_manual_quality_assurance_bad_path, domain, '*_overlay_front.png'))
            file_list_bad_auto = glob.glob(os.path.join(source_auto_quality_assurance_bad_path, domain, '*_overlay_front.png'))
            # file_list_bad_poly_auto = glob.glob(os.path.join(source_auto_quality_assurance_bad_path, domain, '*_overlay_polygon.png'))
            file_list_bad = file_list_bad_manual + file_list_bad_auto
            file_list_bad.sort(key=landsat_sort)
            file_list_bad = [os.path.basename(x) for x in file_list_bad]
            file_list = file_list_manual + file_list_auto
            file_list.sort(key=landsat_sort)
#            file_list = duplicate_prefix_filter(file_list)
            for file_path in file_list:
                
                file_name = os.path.basename(file_path)
                file_name_parts = file_name.split('_')
                file_basename = "_".join(file_name_parts[0:-3])
                # file_overlay_name = "_".join(file_name_parts[0:-1]) + '_overlay_front.png'
                if file_name in file_list_bad:
                    print('Skipping manually pruned:', file_basename)
                    continue
                satellite = file_name_parts[1]
                if satellite.startswith('S'):
                    #Astakhov-Chugunov-Astapenko_S1B_EW_GRDM_1SDH_2018-06-26_011542_01536C_EB6F
    #                datatype = file_name_parts[2]
                    level = file_name_parts[3]
                    date_dashed = file_name_parts[4]
                    date_parts = date_dashed.split('-')
                    year = date_parts[0]
                    month = date_parts[1]
                    day = date_parts[2]
                    date = date_dashed.replace('-', '')
                    orbit = file_name_parts[5]
    #                bandpol = 'hh'
                elif satellite.startswith('L'):
                    #Brückner_LC08_L1TP_2015-06-14_232-014_T1_B5_66-1_validation
    #                datatype = file_name_parts[2]
                    date_dashed = file_name_parts[3]
                    date_parts = date_dashed.split('-')
                    year = date_parts[0]
                    month = date_parts[1]
                    day = date_parts[2]
                    date = date_dashed.replace('-', '')
                    orbit = file_name_parts[4].replace('-', '')
                    level = file_name_parts[5]
    #                bandpol = file_name_parts[6]
                    scene_id = landsat_scene_id_lookup(date, orbit, satellite, level)
                else:
                    print('Unrecognized sattlelite!')
                    continue
                
                if 'mask_extractor' in file_path:
                    source_domain_path = os.path.join(source_path_manual, 'domain')
                elif 'production' in file_path:
                    source_domain_path = os.path.join(source_path_auto, 'domain')
                
#                if not landsat_output_lookup(domain, date, orbit, satellite, level):
#                    print('duplicate pick, continuing:', date, orbit, satellite, level, domain)
#                    continue
                reprocessing_id = file_name_parts[-3][-1]
                old_file_shp_name = file_basename + '_' + reprocessing_id + '_cf.shp'    
                old_file_shp_file_path = os.path.join(source_domain_path, domain, old_file_shp_name)
#                print(old_file_shp_file_path)
                with fiona.open(old_file_shp_file_path, 'r', encoding='utf-8') as source_shp:
                    coords = np.array(list(source_shp)[0]['geometry']['coordinates'])
                    inProj = Proj('epsg:' + epsg_from_domain(domain_path, domain), preserve_units=True) #32621 or 32624 (WGS 84 / UTM zone 21N or WGS 84 / UTM zone 24N)
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
                    
                    output_domain_shp_path = os.path.join(dest_domain_path, 'termini_1972-2019_' + closest_feature_reference_name.replace(' ','-') + '_' + version + '.shp')
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
                        
                        sequence_id = len(output_domain_shp_file)
                        print(closest_feature_reference_name, closest_feature_id, sequence_id)
                        output_data = {
                            'geometry': mapping(LineString(polyline)),
                            'properties': {
                                'GlacierID': closest_feature_id,
                                'Center_X': float(polyline_center[0]),
                                'Center_Y': float( polyline_center[1]),
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

def epsg_from_domain(domain_path, domain):
    """Returns the epsg code as an integer, given the domain shpaefile path and the domain name."""
    domain_prj_path = os.path.join(domain_path, domain + '.prj')
    prj_txt = open(domain_prj_path, 'r').read()
    srs = osr.SpatialReference()
    srs.ImportFromESRI([prj_txt])
    srs.AutoIdentifyEPSG()
    return srs.GetAuthorityCode(None)

if __name__ == "__main__":
    version = "v1.0"
    source_path_manual = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\mask_extractor'
    source_path_auto = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\production_staging'
    dest_domain_path = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\upload_production\v1.0\level-1_shapefiles-domain-termini'
    dest_all_path = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\upload_production\v1.0\level-1_shapefiles-greenland-termini'
    domain_path = r'D:\Daniel\Documents\Github\CALFIN Repo\preprocessing\domains'
    
    glacierIds = fiona.open(r'D:\Daniel\Documents\Github\CALFIN Repo\postprocessing\GlacierIDsRef.shp', 'r', encoding='utf-8')
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
    
    consolidate_shapefiles(source_path_manual, source_path_auto, dest_domain_path, dest_all_path, domain_path, version)
