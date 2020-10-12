# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:38:07 2019

@author: Daniel
"""

import os, glob
from collections import defaultdict
import pandas as pd
os.environ['GDAL_DATA'] = r'D:/ProgramData/Anaconda3/envs/cfm/Library/share/gdal' #Ensure crs are exported correctly by gdal/osr/fiona
import fiona
from fiona.crs import from_epsg

def output_scene_list_csv(dest_all_path, file_list, dest_prefix='calfin'):
    """Generate a 2d Table of the annual number of calving front per domain from consolidated shapefiles."""      
    
    calendar = []
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        file_name_parts = file_name.split('_')
        domain = file_name_parts[0]
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
        calendar.append([domain, scene_id])
            
    calendar_path = os.path.join(dest_all_path, dest_prefix + '_scene_list.csv')
    pd.DataFrame.from_dict(data=pd.DataFrame(calendar), orient='columns').to_csv(calendar_path, header=False, index=False, encoding='utf-8')
    return calendar

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

if __name__ == "__main__":    
    version = "v1.0"
    all_scenes_path = r"../downloader/scenes/all_scenes.txt"
    validation_files_train = glob.glob(r"../training/data/train/*B[0-9].png")
    validation_files_val = glob.glob(r"../training/data/validation/*B[0-9].png")
    dest_all_path = r'../paper'
    shp_types = 'LineString' #'Polygon', 'LineString'
    with open(r"../downloader/scenes/all_scenes.txt", 'r') as scenes_file:
        scene_list = scenes_file.readlines()
        scene_hash_table = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(str))))
        for scene in scene_list:
            scene = scene.split('/t')[0]
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
    
    calendar = output_scene_list_csv(dest_all_path, validation_files_val, dest_prefix='validation')
    calendar = output_scene_list_csv(dest_all_path, validation_files_train, dest_prefix='train')
