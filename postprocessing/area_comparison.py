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
from scipy.ndimage import median_filter, gaussian_filter1d
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
from ordered_line_from_unordered_points import ordered_line_from_unordered_points_tree, is_outlier
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
            # print(date)
            if date[0:4] != '2018':
                continue
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
    fig1, ax1 = plt.subplots(1, 1, num=domain + ' Relative Area Change, 1972-2019')
    plt.subplots_adjust(top = 0.900, bottom = 0.1, right = 0.95, left = 0.05, hspace = 0.25, wspace = 0.25)
    # fig2, ax2 = plt.subplots(1, 1, num=domain + ' Relative Area Change Rate, 1972-2019')
    # plt.subplots_adjust(top = 0.900, bottom = 0.1, right = 0.95, left = 0.05, hspace = 0.25, wspace = 0.25)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Relative Ice Area Change ($km^{2}$)')
    # ax2.set_xlabel('Year')
    # ax2.set_ylabel('Relative Ice Area Change Rate ($km^{2}/yr$)')
#    loc = MonthLocator(bymonth=[1, 6])
    loc = YearLocator(1)
    formatter = mdates.DateFormatter('%Y')
    
    max_relative_change = 0
    for i in range(len(polygon_dict_list)):
        polygon_dict = polygon_dict_list[i]
        dates = []
        changes = []
        annual_changes = defaultdict(list)
        annual_changes_values = dict()
        sorted_dates_keys = list(sorted(polygon_dict.keys()))
        for i in range(len(sorted_dates_keys)):
            date = sorted_dates_keys[i]
            polygon = polygon_dict[date]
            dates.append(datefunc(date))
            changes.append(-polygon['relative'] / 1000000) #Convert from m^2 to km^2
            max_relative_change = max(max_relative_change, changes[-1])
            
            #prep annual
            year = date[0:4]
            annual_changes[year].append(changes[-1])
        annual_changes_values = []
        annual_changes_dates = []
        for year in sorted(annual_changes.keys()):
            year_date = datefunc(year + '-07-01')
            annual_changes_dates.append(year_date)
            annual_changes_values.append(np.mean(annual_changes[year]))
        # annual_relative_change_rate = np.gradient(np.array(annual_changes_values), np.array(annual_changes_dates))                                    
                                              
        relative_change_rate_grad = np.gradient(np.array(changes), np.array(dates) / 365)
        relative_change_rate = np.diff(changes) / np.diff(np.array(dates)/365)
        relative_change_rate_smoothed = gaussian_filter1d(relative_change_rate, 1)
        relative_change_rate_derivative = np.diff(relative_change_rate_smoothed) / np.diff(np.array(dates[1:])/365)
        
        # relative_change_rate = np.diff(changes)
        outliers = is_outlier(relative_change_rate)
        relative_change_rate_masked = []
        dates_gapped = []
        dates_filtered = []
        changes_filtered = []
        print('domain outliers:', domain, sum(outliers))
        # for i in range(len(sorted_dates_keys)):
        #     date = sorted_dates_keys[i]
        #     rate = relative_change_rate[i]
        #     change = changes[i]
        #     #Checking for year match
        #     if outliers[i]:
        #         dates_filtered.append(datefunc(date))
        #         changes_filtered.append(change)
        #         dates_gapped.append(datefunc(date))
        #         relative_change_rate_masked.append(rate)
                        
        #         if i > 0 and i < len(sorted_dates_keys) - 1:
        #             date_prev = sorted_dates_keys[i - 1]
        #             date_next = sorted_dates_keys[i + 1] 
        #             if date[0:4] == date_prev[0:4] and date[0:4] == date_next[0:4]:
        #                 dates_gapped.append(datefunc(date))
        #                 relative_change_rate_masked.append(rate)
        #             else:
        #                 # dates_gapped.append(datefunc(date) + 1)
        #                 # relative_change_rate_masked.append(np.nan)
        #                 dates_gapped.append(datefunc(date))
        #                 relative_change_rate_masked.append(rate)
        #         else:
        #             # dates_gapped.append(datefunc(date) + 1)
        #             # relative_change_rate_masked.append(np.nan)
        #             dates_gapped.append(datefunc(date))
        #             relative_change_rate_masked.append(rate)
                
        average_relative_change_rate = np.mean(relative_change_rate)
        ax1.plot_date(dates, changes, ls='-', marker='o', c='blue', label='Relative Change')
        ax1.plot_date(dates[2:], relative_change_rate_derivative / 1000, ls='-', marker='x', c='red', label='Relative Area Change Rate')
        ax1.plot_date(dates[1:], relative_change_rate_smoothed / 10, ls='-', marker='x', c='orange', label='Relative Area Change Rate')
        # ax2.plot_date(annual_changes_dates, annual_relative_change_rate, ls='-', marker='o', c='blue', label='Annual Relative Area Change Rate')
        
    print("domain", domain, "max_relative_change", max_relative_change, "average_relative_change_rate", average_relative_change_rate)
    calfin_dates = sorted(polygon_dict_list[0].keys())
    start = datefunc(calfin_dates[0]) - 100
    end = datefunc(calfin_dates[-1]) + 100
    plt.xlim(start, end)
    start_year = calfin_dates[0][0:4]
    end_year = calfin_dates[-1][0:4]
    fig1.suptitle(domain + ' Relative Ice Area Change, ' + start_year + '-' + end_year, fontsize=22)
    # fig2.suptitle(domain + ' Relative Ice Area Change Rate, ' + start_year + '-' + end_year, fontsize=22)
    
    ax1.xaxis.set_major_locator(loc)
    ax1.xaxis.set_major_formatter(formatter)
    ax1.xaxis.set_tick_params(labelsize=14)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # ax2.xaxis.set_major_locator(loc)
    # ax2.xaxis.set_major_formatter(formatter)
    # ax2.xaxis.set_tick_params(labelsize=14)
    # plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax1.grid(True)
    ax1.legend()
    # ax2.grid(True)
    # ax2.legend()
    fig1.savefig(os.path.join(dest_path, domain + '_relative_area_change.png'), bbox_inches='tight', pad_inches=0, frameon=False)
    # fig2.savefig(os.path.join(dest_path, domain + '_relative_area_change_rate.png'), bbox_inches='tight', pad_inches=0, frameon=False)
    
    return max_relative_change 


def generate_graphs(calfin_path):
    domain = os.path.basename(calfin_path).split('_')[2]
    domains = ['Hayes-Gletsjer', 'Helheim-Gletsjer', 'Jakobshavn-Isbrae', 'Kangerlussuaq-Gletsjer', 
               'Kangiata-Nunaata-Sermia', 'Kong-Oscar-Gletsjer', 'Petermann-Gletsjer', 'Rink-Isbrae', 
               'Upernavik-Isstrom-N-C', 'Upernavik-Isstrom-S']
    if domain not in domains:
        return 0, domain
    
    dest_path = r"../paper/area_change2/"
    polygons_calfin = calfin_read(calfin_path)
    
    #Calculate relative to reference point
    calfin_dates = sorted(polygons_calfin.keys())
    
    reference_area = polygons_calfin[calfin_dates[0]]['area']
    polylines_calfin = calculate_relative_change(polygons_calfin, reference_area)
    
    polygon_dict_list = [polylines_calfin]
    if len(polygons_calfin) > 1:
        max_relative_change = graph_change(polygon_dict_list, dest_path, domain)
        return max_relative_change, domain
    else:
        return 0, domain


if __name__ == "__main__":
    #Set figure size for 1600x900 resolution, tight layout
    plt.close('all')
    plt.rcParams["figure.figsize"] = (22,9)
    plt.rcParams["font.size"] = "20"
    
    change_dict = dict()
    for path in glob.glob('../outputs/upload_production/v1.0/level-1_shapefiles-domain-termini/*_closed_v1.0.shp'):
        max_relative_change, domain = generate_graphs(path)
        change_dict[domain] = max_relative_change
    # for key in sorted(change_dict, key=change_dict.get):
    #     print(key, change_dict[key])
    plt.show()
        
    



