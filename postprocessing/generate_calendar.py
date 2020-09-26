# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:38:07 2019

@author: Daniel
"""

import os
from collections import defaultdict
import pandas as pd
os.environ['GDAL_DATA'] = r'D:\ProgramData\Anaconda3\envs\cfm\Library\share\gdal' #Ensure crs are exported correctly by gdal/osr/fiona
import fiona
from fiona.crs import from_epsg

def output_calendar_csv(dest_all_path, version, shp_type):
    """Generate a 2d Table of the annual number of calving front per domain from consolidated shapefiles."""      
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
    if shp_type == 'LineString':
        suffix = '_' + version + '.shp'
    elif shp_type == 'Polygon':
        suffix = '_closed_' + version + '.shp'
    else:
        raise ValueError('Unrecognized shp_type (should be "line" or "polygon"):', shp_type)    
    output_all_shp_path = os.path.join(dest_all_path, 'termini_1972-2019_Greenland' + suffix)
    calendar = defaultdict(lambda: defaultdict(int))
    with fiona.open(output_all_shp_path, 
            'r', 
            driver='ESRI Shapefile', 
            crs=fiona.crs.from_epsg(3413), 
            schema=schema, 
            encoding='utf-8') as output_all_shp_file:
        for feature in output_all_shp_file:
            date = feature['properties']['Date']
            name = feature['properties']['RefName']
            year = date.split('-')[0]
            calendar[name][year] += 1
    
    calendar_path = os.path.join(dest_all_path, 'calendar.csv')
    pd.DataFrame.from_dict(data=calendar, orient='columns').to_csv(calendar_path, header=True)
    return calendar

            

if __name__ == "__main__":    
    version = "v1.0"
    dest_all_path = r'../outputs/upload_production/v1.0/level-1_shapefiles-greenland-termini'
    shp_types = 'LineString' #'Polygon', 'LineString'
    
    calendar = output_calendar_csv(dest_all_path, version, shp_types)
