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
from ordered_line_from_unordered_points import ordered_line_from_unordered_points_tree
#sys.path.insert(1, '../postprocessing'

#level 0 should inlcude all subsets (preprocessed)
#Make individual ones, domain ones, and all available
#indivudal ones include QA, tif, and shapefile

#have front line,
#to use smoother, must have full mask
#to get full mask, repeated polygonization using regular front line
    
def calfin_read(path):
    """Reads calfin Shapefile into dictionary with date keys and polyline values."""
    result = dict()
    with fiona.open(path, 'r', encoding='utf-8') as shp:
        for feature in shp:
            polygons_coords = feature['geometry']['coordinates']
            date = feature['properties']['Date']
            
            #Handle overlapping dates
            # if date in result:
            #     area = result[date]['area']
            #     polygons = result[date]['polygons']
            # else:
            #     area = 0
            #     polygons = []
            area = 0
            polygons = []
           
            #Accumalate results
            for polygon_coords in polygons_coords:
                polygon = Polygon(polygon_coords) 
                polygons.append(polygons)
                area += polygon.area
            result[date] = {'polygons': polygons, 'area':area}
    return result


def calculate_relative_change(polygon_dict, reference_area):
    """Calculates intersection point between a list of lines and a centerline."""
    """Takes in date-intersection dict and centerline to determine change relative to given reference point."""
    for date, polygon_obj in polygon_dict.items():
        polygon_obj['relative'] = polygon_obj['area'] - reference_area
    return polygon_dict


def graph_change(polygon_dict_list, dest_path, domain):
    """Takes in list of dicts and plots them."""
    # Converter function
    datefunc = lambda x: mdates.date2num(datetime.strptime(x, '%Y-%m-%d'))

    #Initialize plots
#    fig = plt.figure(1)
    fig, ax = plt.subplots(1, 1, num=domain + ' Relative Area Change, 1972-2019')
    plt.subplots_adjust(top = 0.900, bottom = 0.1, right = 0.95, left = 0.05, hspace = 0.25, wspace = 0.25)
    ax.set_xlabel('Year')
    ax.set_ylabel('Relative Ice Area Change ($km^{2}$)')
#    loc = MonthLocator(bymonth=[1, 6])
    loc = YearLocator(1)
    formatter = mdates.DateFormatter('%Y')
    
    max_relative_change = 0
    for i in range(len(polygon_dict_list)):
        polygon_dict = polygon_dict_list[i]
        dates = []
        changes = []
        for date in sorted(polygon_dict.keys()):
            polygon = polygon_dict[date]
            dates.append(datefunc(date))
            changes.append(polygon['relative'] / 1000000) #Convert from m^2 to km^2
            max_relative_change = max(max_relative_change, changes[-1])
        ax.plot_date(dates, changes, ls='-', marker='o', c='blue', label='CALFIN')
        
    print("domain", domain, "max_relative_change", max_relative_change)
    calfin_dates = sorted(polygon_dict_list[0].keys())
    start = datefunc(calfin_dates[0]) - 100
    end = datefunc(calfin_dates[-1]) + 100
    plt.xlim(start, end)
    start_year = calfin_dates[0][0:4]
    end_year = calfin_dates[-1][0:4]
    fig.suptitle(domain + ' Relative Ice Area Change, ' + start_year + '-' + end_year, fontsize=22)
    
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_tick_params(labelsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.grid(True)
    ax.legend()
    # plt.show()
    fig.savefig(os.path.join(dest_path, domain + '_relative_area_change.png'), bbox_inches='tight', pad_inches=0, frameon=False)
    
    return max_relative_change 


def generate_centerline_graphs(calfin_path):
    dest_path = r"../paper/area_change/"
    polygons_calfin = calfin_read(calfin_path)
    
    #Calculate relative to reference point
    calfin_dates = sorted(polygons_calfin.keys())
    reference_area = polygons_calfin[calfin_dates[0]]['area']
    polylines_calfin = calculate_relative_change(polygons_calfin, reference_area)
    
    polygon_dict_list = [polylines_calfin]
    domain = os.path.basename(calfin_path).split('_')[2]
    max_relative_change = graph_change(polygon_dict_list, dest_path, domain)
    
    return max_relative_change, domain


if __name__ == "__main__":
    #Set figure size for 1600x900 resolution, tight layout
    plt.close('all')
    plt.rcParams["figure.figsize"] = (22,9)
    plt.rcParams["font.size"] = "20"
    
    change_dict = dict()
    for path in glob.glob('../outputs/upload_production/v1.0/level-1_shapefiles-domain-termini/*_closed_v1.0.shp'):
        max_relative_change, domain = generate_centerline_graphs(path)
        change_dict[domain] = max_relative_change
    for key in sorted(change_dict, key=change_dict.get):
        print(key, change_dict[key])
    # generate_centerline_graphs("../outputs/upload_production/v1.0/level-1_shapefiles-greenland-termini/termini_1972-2019_Greenland_closed_v1.0.shp")
    plt.show()
        
    



