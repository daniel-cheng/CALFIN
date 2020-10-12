# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:38:07 2019

@author: Daniel
"""
import numpy as np
import cv2
import os, shutil, glob, copy, sys
from datetime import datetime
os.environ['GDAL_DATA'] = r'D://ProgramData//Anaconda3//envs//cfm//Library//share//gdal' #Ensure crs are exported correctly by gdal/osr/fiona

#import rasterio
#from rasterio import features
#from rasterio.windows import from_bounds
from scipy.ndimage.morphology import distance_transform_edt
from scipy.spatial import KDTree
from scipy.ndimage import median_filter
from skimage.io import imsave, imread
from skimage import measure
from skimage.transform import resize
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, YearLocator
from pyproj import Proj, transform
from shapely.geometry import mapping, Polygon, LineString, Point
from collections import defaultdict
import fiona
from fiona.crs import from_epsg
#sys.path.insert(1, '../postprocessing'

#level 0 should inlcude all subsets (preprocessed)
#Make individual ones, domain ones, and all available
#indivudal ones include QA, tif, and shapefile

#have front line,
#to use smoother, must have full mask
#to get full mask, repeated polygonization using regular front line


def compare_centerline():
    """main function."""
    pass
    
def calfin_read(path):
    """Reads calfin Shapefile into dictionary with date keys and polyline values."""
    result = dict()
    with fiona.open(path, 'r', encoding='utf-8') as shp:
         for feature in shp:
             polyline_coords = feature['geometry']['coordinates']
             date = feature['properties']['Date']
             result[date] = {'coords': polyline_coords}
    return result

def esacci_read(path):
    """Reads ESA-CCI Shapefiles into dictionary with date keys and polyline values."""
    result = dict()
    file_paths = glob.glob(os.path.join(path, '*.shp'))
    inProj = Proj('epsg:4326')
    outProj = Proj('epsg:3413')
    
    for file_path in file_paths:
        with fiona.open(file_path, 'r', encoding='utf-8') as shp:
            polyline_coords = np.array(shp[0]['geometry']['coordinates'])
            lat = polyline_coords[:,0]
            long = polyline_coords[:,1]
            x2, y2 = transform(inProj, outProj, long, lat)
            polyline_coords = list(zip(x2, y2))
            
            date = shp[0]['properties']['acq_time'].split(' ')[0]
            result[date] = {'coords': polyline_coords}
#            print(date)
    return result

def measures_read(path):
    """Reads MEaSUREs Shapefiles into dictionary with date keys and polyline values."""
    result = dict()
    file_paths = glob.glob(os.path.join(path, '*', '*.shp'))
    for file_path in file_paths:
         with fiona.open(file_path, 'r', encoding='utf-8') as shp:
             for feature in shp:
                 if feature['properties']['GlacierID'] == 175:
                     if 'DateRange' in feature['properties']:
                         polyline_coords = feature['geometry']['coordinates']
                         date_range = feature['properties']['DateRange'].split('-')
                         date_time_objs = [datetime.strptime(date_range[0], '%d%b%Y'), datetime.strptime(date_range[1], '%d%b%Y')]
                         date_time_objs = [datetime.strftime(date_time_objs[0], '%Y-%m-%d'), datetime.strftime(date_time_objs[1], '%Y-%m-%d')]
                         result[date_time_objs[0]] = {'coords': polyline_coords, 'end_date':date_time_objs[1]}
                     elif 'DATE' in feature['properties']:
                         polyline_coords = feature['geometry']['coordinates']
                         date = feature['properties']['DATE']
                         result[date] = {'coords': polyline_coords}
    return result

def promice_read(path):
    """Reads ESA-CCI Shapefiles into dictionary with date keys and polyline values."""
    result = dict()
    file_paths = glob.glob(os.path.join(path, '*', '*.shp'))
    inProj = Proj('epsg:4326')
    outProj = Proj('epsg:3413')
    
    for file_path in file_paths:
        with fiona.open(file_path, 'r', encoding='utf-8') as shp:
            try:
                polyline_coords = np.array(shp[0]['geometry']['coordinates'])
#                print(file_path, len(polyline_coords))
                lat = polyline_coords[:,0]
                long = polyline_coords[:,1]
                x2, y2 = transform(inProj, outProj, long, lat)
                polyline_coords = list(zip(x2, y2))
                
                date = file_path.split(os.path.sep)[-1][7:11] + '-08-15'
#                print(date)
                result[date] = {'coords': polyline_coords}
    #            print(date)
            except Exception as e:
                print(e)
                print('skipping', file_path)
#    print(result)
    return result

def centerline_read(path):
    """Reads centerline Shapefile."""
    result = None
    with fiona.open(path, 'r', encoding='utf-8') as shp:
         for feature in shp:
             if feature['properties']['order'] == 'C':
                 result = LineString(feature['geometry']['coordinates'])
                 break
    return result


def centerline_intersection(line_dict, centerline):
    """Calculates intersection point between a list of lines and a centerline."""
    """Takes in date-intersection dict and centerline to determine change relative to given reference point."""
    result = dict()
    forward = np.array(centerline.coords[0])
    back = np.array(centerline.coords[1])
    advance_vector = forward - back
    retreat_vector = back - forward
    for date, line in line_dict.items():
        line_obj = LineString(line['coords'])
        intersect = centerline.intersection(line_obj)
        if type(intersect) == Point:
            absolute = np.array(intersect.coords)
            t = ((absolute - forward) / retreat_vector)[0,0] #calculate percentage change as t
#            absolute = t * advance_vector + back
#            print(date)
            line['center'] = absolute
            line['t'] = t
            result[date] = line
#        else:
#            print('esa', date, intersect)
    return result

def calculate_relative_change(line_dict, centerline, reference_point):
    """Calculates intersection point between a list of lines and a centerline."""
    """Takes in date-intersection dict and centerline to determine change relative to given reference point."""
#    result = dict()
    forward = np.array(centerline.coords[0])
    back = np.array(centerline.coords[1])
    advance_vector = forward - back
    retreat_vector = back - forward
    for date, line in line_dict.items():
#        if 'center' not in line:
#            print(date)
#        else:
        direction = line['center'] - reference_point['center']
        change = np.dot(direction, retreat_vector) / np.linalg.norm(retreat_vector) 
#        print(change)
        line['relative'] = change[0] / 1000
    return line_dict


def graph_change(line_dict_list, dest_path):
    """Takes in list of dicts and plots them."""
    # Converter function
    datefunc = lambda x: mdates.date2num(datetime.strptime(x, '%Y-%m-%d'))

    #Initialize plots
#    fig = plt.figure(1)
    fig, ax = plt.subplots(1, 1, num='1')
    fig.suptitle('Helheim Relative Length Change, 1972-2019', fontsize=22)
    plt.subplots_adjust(top = 0.900, bottom = 0.1, right = 0.95, left = 0.05, hspace = 0.25, wspace = 0.25)
    
    ax.set_title('CALFIN vs ESA-CCI vs MEaSUREs vs PROMICE', fontsize=18)
    ax.set_xlabel('Year')
    ax.set_ylabel('Relative Length Retreat (km)')
#    loc = MonthLocator(bymonth=[1, 6])
    loc = YearLocator(1)
    formatter = mdates.DateFormatter('%Y')
    
    for i in range(len(line_dict_list)):
        line_dict = line_dict_list[i]
        dates = []
        changes = []
        for date in sorted(line_dict.keys()):
            line = line_dict[date]
            dates.append(datefunc(date))
            changes.append(line['relative'])
            if 'end_date' in line:
                dates.append(datefunc(line['end_date']))
                changes.append(line['relative'])
        if i == 0: #CALFIN
            ax.plot_date(dates, changes, ls='-', marker='o', c='blue', label='CALFIN')
        elif i == 1: #ESA-CCI
            ax.plot_date(dates, changes, ls='-.', marker='*', c='orangered', label='ESA-CCI')
        elif i == 2: #MEaSUREs
            ax.plot_date(dates, changes, ls=':', marker='x', c='magenta', label='MEaSUREs')
        elif i == 3: #PROMICE
            ax.plot_date(dates, changes, ls=':', marker='P', c='green', label='PROMICE')
        else:
            ax.plot_date(dates, changes, ls='-', marker='o')
    calfin_dates = sorted(line_dict_list[0].keys())
    start = datefunc(calfin_dates[0]) - 100
    end = datefunc(calfin_dates[-1]) + 100
    plt.xlim(start, end)
    
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_tick_params(labelsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.grid(True)
    ax.legend()
    plt.show()
#    plt.savefig(os.path.join(dest_path, 'Helheim_relative_change.png'))
    fig.savefig(os.path.join(dest_path, 'Helheim_relative_change_a.png'), bbox_inches='tight', pad_inches=0, frameon=False)

def graph_change_esa(line_dict_list, dest_path):
    """Takes in list of dicts and plots them."""
    # Converter function
    datefunc = lambda x: mdates.date2num(datetime.strptime(x, '%Y-%m-%d'))

    #Initialize plots
#    fig = plt.figure(2)
    fig, ax = plt.subplots(1, 1, num='2')
    fig.suptitle('Helheim Relative Length Change, 1995-2016', fontsize=22)
    plt.subplots_adjust(top = 0.900, bottom = 0.1, right = 0.95, left = 0.05, hspace = 0.25, wspace = 0.25)
    
    ax.set_title('CALFIN vs ESA-CCI', fontsize=18)
    ax.set_xlabel('Year')
    ax.set_ylabel('Relative Length Retreat (km)')
#    loc = MonthLocator(bymonth=[1, 6])
    loc = YearLocator(1)
    formatter = mdates.DateFormatter('%Y')
    
    for i in range(len(line_dict_list)):
        line_dict = line_dict_list[i]
        dates = []
        changes = []
        for date in sorted(line_dict.keys()):
            line = line_dict[date]
            dates.append(datefunc(date))
            changes.append(line['relative'])
            if 'end_date' in line:
                dates.append(datefunc(line['end_date']))
                changes.append(line['relative'])
        if i == 0: #CALFIN
            ax.plot_date(dates, changes, ls='-', marker='o', c='blue', label='CALFIN')
        elif i == 1: #ESA-CCI
            ax.plot_date(dates, changes, ls='-.', marker='*', c='orangered', label='ESA-CCI')
        elif i == 2: #MEaSUREs
            ax.plot_date(dates, changes, ls=':', marker='x', c='magenta', label='MEaSUREs')
        elif i == 3: #PROMICE
            ax.plot_date(dates, changes, ls=':', marker='P', c='green', label='PROMICE')
        else:
            ax.plot_date(dates, changes, ls='-', marker='o')
    calfin_dates = sorted(line_dict_list[0].keys())
    start = datefunc('1995-01-01') - 100
    end = datefunc('2015-12-30') + 100
    plt.xlim(start, end)
    plt.ylim(-2.66, 8)
    
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_tick_params(labelsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.grid(True)
    ax.legend()
    plt.show()
#    plt.savefig(os.path.join(dest_path, 'Helheim_relative_change_esa.png'))
    fig.savefig(os.path.join(dest_path, 'Helheim_relative_change_esa_a.png'), bbox_inches='tight', pad_inches=0, frameon=False)


if __name__ == "__main__":
    calfin_path = r"../outputs/upload_production/v1.0/level-1_shapefiles-domain-termini/termini_1972-2019_Helheim-Gletsjer_v1.0.shp"
    esacci_path = r"../postprocessing/esacci/Products/v3.0/Helheim_Gletsjer_G321627E66422N"
    measures_path = r"../postprocessing/measures"
    promice_path = r"../postprocessing/promice/Helheim_frontlines_4326"
    centerline_path = r"../postprocessing/centerlines/Helheim_centerline.shp"
    dest_path = r"../paper"
    polylines_calfin = calfin_read(calfin_path)
    polylines_esacci = esacci_read(esacci_path)
    polylines_measures = measures_read(measures_path)
    polylines_promice = promice_read(promice_path)
    centerline = centerline_read(centerline_path)
    
    #process intersections
    polylines_calfin = centerline_intersection(polylines_calfin, centerline)
    polylines_esacci = centerline_intersection(polylines_esacci, centerline)
    polylines_measures = centerline_intersection(polylines_measures, centerline)
    polylines_promice = centerline_intersection(polylines_promice, centerline)
    
    #Calculate relative to reference point
    reference_point = polylines_calfin['1972-09-06']
    polylines_calfin = calculate_relative_change(polylines_calfin, centerline, reference_point)
    polylines_esacci = calculate_relative_change(polylines_esacci, centerline, reference_point)
    polylines_measures = calculate_relative_change(polylines_measures, centerline, reference_point)
    polylines_promice = calculate_relative_change(polylines_promice, centerline, reference_point)
    
    #Graph results
    #Set figure size for 1600x900 resolution, tight layout
    plt.close('all')
    plt.rcParams["figure.figsize"] = (22,9)
    plt.rcParams["font.size"] = "20"
    plt.subplots_adjust(top = 0.925, bottom = 0.05, right = 0.95, left = 0.05, hspace = 0.25, wspace = 0.25)
    
    # line_dict_list = [polylines_calfin, polylines_esacci, polylines_measures, polylines_promice]
    # graph_change(line_dict_list, dest_path)
    line_dict_list = [polylines_calfin, polylines_esacci]
    graph_change_esa(line_dict_list, dest_path)



