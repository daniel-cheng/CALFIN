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

def measures_read(path, glacier_id):
    """Reads MEaSUREs Shapefiles into dictionary with date keys and polyline values."""
    result = dict()
    file_paths = glob.glob(os.path.join(path, '*', '*.shp'))
    for file_path in file_paths:
         with fiona.open(file_path, 'r', encoding='utf-8') as shp:
             for feature in shp:
                 if feature['properties']['GlacierID'] == glacier_id:
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
                if type(polyline_coords[0]) == list:
                    polyline_coords = np.array(polyline_coords[0])
                lat = polyline_coords[:,0]
                long = polyline_coords[:,1]
                x2, y2 = transform(inProj, outProj, long, lat)
                polyline_coords = list(zip(x2, y2))
                
                date = file_path.split(os.path.sep)[-1][-8:-4] + '-08-15' #Use end of melt season as date (Aug 15)
                result[date] = {'coords': polyline_coords}
            except Exception as e:
                print(e)
                print('skipping:', file_path)
    return result

def centerline_read(path, glacier_id):
    """Reads centerline Shapefile."""
    result = None
    with fiona.open(path, 'r', encoding='utf-8') as shp:
         for feature in shp:
             if feature['properties']['GlacierID'] == glacier_id:
                 result = LineString(feature['geometry']['coordinates'])
                 break
    return result


def centerline_intersection(line_dict, centerline):
    """Calculates intersection point between a list of lines and a centerline."""
    """Takes in date-intersection dict and centerline to determine change relative to given reference point."""
    result = dict()
    for date, line in line_dict.items():
        line_obj = LineString(line['coords'])
        arclength = 0
        for i in range(len(centerline.coords) - 1):
            forward = np.array(centerline.coords[i])
            back = np.array(centerline.coords[i + 1])
            advance_vector = forward - back
            retreat_vector = back - forward
            intersect = LineString([forward, back]).intersection(line_obj)
            if type(intersect) == Point:
                absolute = np.array(intersect.coords)
                t = ((absolute - forward) / retreat_vector)[0,0] #calculate percentage change as t
                # absolute = t * advance_vector + back
                # print(date)
                arclength += np.linalg.norm(retreat_vector * t)
                line['center'] = absolute
                line['arclength'] = arclength
                line['t'] = t
                result[date] = line
                break
            arclength += np.linalg.norm(retreat_vector)
            # else:
            #     print('esa', date, intersect)
    return result

def calculate_relative_change(line_dict, centerline, reference_point):
    """Calculates intersection point between a list of lines and a centerline."""
    """Takes in date-intersection dict and centerline to determine change relative to given reference point."""
#    result = dict()
    # forward = np.array(centerline.coords[0])
    # back = np.array(centerline.coords[1])
    # advance_vector = forward - back
    # retreat_vector = back - forward
    for date, line in line_dict.items():
#        if 'center' not in line:
#            print(date)
#        else:
#         direction = line['center'] - reference_point['center']
#         change = np.dot(direction, retreat_vector) / np.linalg.norm(retreat_vector) 
# #        print(change)
#         line['relative'] = change[0] / 1000
        line['relative'] = (line['arclength'] - reference_point['arclength']) / 1000
    return line_dict


def graph_change(line_dict_list, dest_path, domain):
    """Takes in list of dicts and plots them."""
    # Converter function
    datefunc = lambda x: mdates.date2num(datetime.strptime(x, '%Y-%m-%d'))

    #Initialize plots
#    fig = plt.figure(1)
    fig, ax = plt.subplots(1, 1, num=domain + ' Relative Length Change, 1972-2019')
    fig.suptitle(domain + ' Relative Length Change, 1972-2019', fontsize=22)
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
            ax.plot_date(dates, changes, ls='', marker='*',  ms=16, c='orangered', label='ESA-CCI')
        elif i == 2: #MEaSUREs
            ax.plot_date(dates, changes, ls='', marker='x',  ms=16, c='magenta', label='MEaSUREs')
        elif i == 3: #PROMICE
            ax.plot_date(dates, changes, ls='', marker='P',  ms=8, c='green', label='PROMICE')
        else:
            ax.plot_date(dates, changes, ls='', marker='o', ms=16)
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
    # fig.savefig(os.path.join(dest_path, domain + '_relative_change.png'), bbox_inches='tight', pad_inches=0, frameon=False)

def graph_change_esa(line_dict_list, dest_path, domain):
    """Takes in list of dicts and plots them."""
    # Converter function
    datefunc = lambda x: mdates.date2num(datetime.strptime(x, '%Y-%m-%d'))

    #Initialize plots
#    fig = plt.figure(2)
    fig, ax = plt.subplots(1, 1, num=domain + ' Relative Length Change, 1995-2016')
    fig.suptitle(domain + ' Relative Length Change, 1995-2016', fontsize=22)
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
            ax.plot_date(dates, changes, ls='', marker='*', ms=16, c='orangered', label='ESA-CCI')
        elif i == 2: #MEaSUREs
            ax.plot_date(dates, changes, ls='', marker='x', ms=16, c='magenta', label='MEaSUREs')
        elif i == 3: #PROMICE
            ax.plot_date(dates, changes, ls='', marker='P', ms=16, c='green', label='PROMICE')
        else:
            ax.plot_date(dates, changes, ls='', marker='o', ms=16)
    # calfin_dates = sorted(line_dict_list[0].keys())
    start = datefunc('1995-01-01') - 100
    end = datefunc('2015-12-30') + 100
    plt.xlim(start, end)
    # plt.ylim(-2.66, 8)
    
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_tick_params(labelsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.grid(True)
    ax.legend()
    plt.show()
    # fig.savefig(os.path.join(dest_path, domain + '_relative_change_esa.png'), bbox_inches='tight', pad_inches=0, frameon=False)



def generate_centerline_graphs(domain):
    calfin_path = r"../outputs/upload_production/v1.0/level-1_shapefiles-domain-termini/termini_1972-2019_" + domain[0] + "_v1.0.shp"
    esacci_path = r"../postprocessing/esacci/Products/v3.0/" + domain[1]
    measures_path = r"../postprocessing/measures"
    promice_path = r"../postprocessing/promice/" + domain[2] + "_frontlines_4326"
    centerline_path = r"../postprocessing/centerlines/Centerlines.shp"
    dest_path = r"../paper"
    polylines_calfin = calfin_read(calfin_path)
    polylines_esacci = esacci_read(esacci_path)
    polylines_measures = measures_read(measures_path, domain[3])
    polylines_promice = promice_read(promice_path)
    centerline = centerline_read(centerline_path, domain[3])
    
    #process intersections
    polylines_calfin = centerline_intersection(polylines_calfin, centerline)
    polylines_esacci = centerline_intersection(polylines_esacci, centerline)
    polylines_measures = centerline_intersection(polylines_measures, centerline)
    polylines_promice = centerline_intersection(polylines_promice, centerline)
    
    #Calculate relative to reference point
    calfin_dates = sorted(polylines_calfin.keys())
    reference_point = polylines_calfin[calfin_dates[0]]
    polylines_calfin = calculate_relative_change(polylines_calfin, centerline, reference_point)
    polylines_esacci = calculate_relative_change(polylines_esacci, centerline, reference_point)
    polylines_measures = calculate_relative_change(polylines_measures, centerline, reference_point)
    polylines_promice = calculate_relative_change(polylines_promice, centerline, reference_point)
    
    line_dict_list = [polylines_calfin, polylines_esacci, polylines_measures, polylines_promice]
    graph_change(line_dict_list, dest_path, domain[0])
    line_dict_list = [polylines_calfin, polylines_esacci]
    graph_change_esa(line_dict_list, dest_path, domain[0])
    
def generate_calfin_centerline_graphs(domain):
    calfin_path = r"../outputs/upload_production/v1.0/level-1_shapefiles-domain-termini/termini_1972-2019_" + domain[0] + "_v1.0.shp"
    centerline_path = r"../postprocessing/centerlines/Centerlines.shp"
    dest_path = r"../paper"
    polylines_calfin = calfin_read(calfin_path)
    polylines_esacci = esacci_read(esacci_path)
    polylines_measures = measures_read(measures_path, domain[3])
    polylines_promice = promice_read(promice_path)
    centerline = centerline_read(centerline_path, domain[3])
    
    #process intersections
    polylines_calfin = centerline_intersection(polylines_calfin, centerline)
    polylines_esacci = centerline_intersection(polylines_esacci, centerline)
    polylines_measures = centerline_intersection(polylines_measures, centerline)
    polylines_promice = centerline_intersection(polylines_promice, centerline)
    
    #Calculate relative to reference point
    calfin_dates = sorted(polylines_calfin.keys())
    reference_point = polylines_calfin[calfin_dates[0]]
    polylines_calfin = calculate_relative_change(polylines_calfin, centerline, reference_point)
    polylines_esacci = calculate_relative_change(polylines_esacci, centerline, reference_point)
    polylines_measures = calculate_relative_change(polylines_measures, centerline, reference_point)
    polylines_promice = calculate_relative_change(polylines_promice, centerline, reference_point)
    
    line_dict_list = [polylines_calfin, polylines_esacci, polylines_measures, polylines_promice]
    graph_change(line_dict_list, dest_path, domain[0])
    line_dict_list = [polylines_calfin, polylines_esacci]
    graph_change_esa(line_dict_list, dest_path, domain[0])
    
if __name__ == "__main__":
    #Graph results
    #Set figure size for 1600x900 resolution, tight layout
    plt.close('all')
    plt.rcParams["figure.figsize"] = (22,9)
    plt.rcParams["font.size"] = "20"
    # plt.subplots_adjust(top = 0.925, bottom = 0.05, right = 0.95, left = 0.05, hspace = 0.25, wspace = 0.25)
    
    #['CALFIN', 'ESACCI', 'PROMICE', 'MEaSUREs GlacierID']
    domains = [['Akullersuup-Sermia', '', 'Akullersuup_Sermia', 208],
               ['Docker-Smith-Gletsjer', '', 'Døcker_Smith', 58],
               ['Docker-Smith-Gletsjer', '', 'Døcker_Smith', 60],
               ['Docker-Smith-Gletsjer', '', 'Døcker_Smith', 59],
               ['Fenris-Gletsjer', '', 'Fenris', 174],
               ['Hayes-Gletsjer', '', 'Hayes', 40],
               ['Helheim-Gletsjer', 'Helheim_Gletsjer_G321627E66422N', 'Helheim', 175],
               ['Inngia-Isbrae', 'Inngia_Isbrae_G307495E72082N', 'Ingia', 19],
               ['Jakobshavn-Isbrae', 'Jakobshavn_Isbrae_G310846E69083N', 'Jakobshavn', 3],
               ['Kangerluarsuup-Sermia', '', 'Kangerdluarssup_Sermia', 15],
               ['Kangerlussuaq-Gletsjer', 'Kangerlussuaq_Gletsjer_G326914E68667N', 'Kangerdlugssuaq', 153],
               ['Kangerlussuup-Sermia', '', 'Kangerdlugssup_Sermerssua', 16],
               ['Kangiata-Nunaata-Sermia', 'Kangiata_Nunaata_Sermia_G310427E64274N', 'KNS', 207],
               ['Kangilleq-Kangigdleq-Isbrae', 'Kangilleq_G309415E70752N', 'Kangigdleq', 12],
               ['Kong-Oscar-Gletsjer', 'Kong_Oscar_Gletsjer_G300347E76024N', 'Kong_Oscars', 52],
               ['Kong-Oscar-Gletsjer', 'Kong_Oscar_Gletsjer_G300347E76024N', 'Kong_Oscars', 53],
               ['Lille-Gletsjer', '', 'Lille', 10],
               ['Midgard-Gletsjer', '', 'Midgaard', 173],
               ['Alison-Gletsjer', 'Nunatakassaap_Sermia_G304089E74641N', 'Nunatakassaap', 35],
               ['Naajarsuit-Sermiat', '', 'Nunatakavsaup', 25],
               ['Perlerfiup-Sermia', 'Perlerfiup_Sermia_G309225E70985N', 'Perdlerfiup', 14],
               ['Petermann-Gletsjer', 'Petermann_Gletsjer_G299936E80548N', 'Petermann', 93],
               ['Rink-Isbrae', 'Rink_Isbrae_G308503E71781N', 'Rink', 17],
               ['Sermeq-Avannarleq-69N', 'Sermeq_Avannarleq_G309720E69381N', 'Sermeq_Avannarleq', 4],
               ['Sermeq-Silarleq', '', 'Sermeq_Silardleq', 13],
               ['Sermilik-Isbrae', 'Sermilik_Brae_G313080E61025N', 'Sermilik', 11],
               ['Steenstrup-Gletsjer', 'Steenstrup_Gletsjer_G302212E75326N', 'Steenstrup', 44],
               ['Store-Gletsjer', 'Store_Gletsjer_G309511E70416N', 'Store', 9],
               ['Upernavik-Isstrom-S', 'Upernavik_Isstroem_G305731E72859N', 'UpernavikA', 20],
               ['Upernavik-Isstrom-S', 'Upernavik_Isstroem_G305731E72859N', 'UpernavikB', 21],
               ['Upernavik-Isstrom-N-C', 'Upernavik_Isstroem_G305731E72859N', 'UpernavikC', 22],
               ['Upernavik-Isstrom-N-C', 'Upernavik_Isstroem_G305731E72859N', 'UpernavikD', 23],
               ['Upernavik-Isstrom-N-C', 'Upernavik_Isstroem_G305731E72859N', 'UpernavikE', 23],
               ['Upernavik-Isstrom-NW', 'Upernavik_Isstroem_G305731E72859N', 'UpernavikF', 24]]
    domains = [['Petermann-Gletsjer', 'Petermann_Gletsjer_G299936E80548N', 'Petermann', 93],
               ['Kangerlussuaq-Gletsjer', 'Kangerlussuaq_Gletsjer_G326914E68667N', 'Kangerdlugssuaq', 153],
               ['Jakobshavn-Isbrae', 'Jakobshavn_Isbrae_G310846E69083N', 'Jakobshavn', 3]]
    for domain in domains:
        generate_centerline_graphs(domain)
    



