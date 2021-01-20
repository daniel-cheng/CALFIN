# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:38:07 2019

@author: Daniel
"""
import numpy as np
import os, glob
from datetime import datetime
# os.environ['GDAL_DATA'] = r'D://ProgramData//Anaconda3//envs//cfm//Library//share//gdal' #Ensure crs are exported correctly by gdal/osr/fiona

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import YearLocator
from pyproj import Proj, transform
from shapely.geometry import Polygon, LineString, Point
from collections import defaultdict
import fiona

    
def calfin_read(path):
    """Reads calfin Shapefile into dictionary with date keys and polyline values."""
    result = defaultdict(list)
    with fiona.open(path, 'r', encoding='utf-8') as shp:
        for feature in shp:
            polyline_coords = feature['geometry']['coordinates']
            date = feature['properties']['Date']
            result[date].append({'coords': polyline_coords})
    return result

def esacci_read(paths):
    """Reads ESA-CCI Shapefiles into dictionary with date keys and polyline values."""
    result = defaultdict(list)
    inProj = Proj('epsg:4326')
    outProj = Proj('epsg:3413')
    file_paths = []
    for path in paths:
        file_paths += glob.glob(os.path.join(path, '*.shp'))
    
    for file_path in file_paths:
        with fiona.open(file_path, 'r', encoding='utf-8') as shp:
            polyline_coords = np.array(shp[0]['geometry']['coordinates'])
            lat = polyline_coords[:,0]
            long = polyline_coords[:,1]
            x2, y2 = transform(inProj, outProj, long, lat)
            polyline_coords = list(zip(x2, y2))
            
            date = shp[0]['properties']['acq_time'].split(' ')[0]
            result[date].append({'coords': polyline_coords})
    return result

def measures_read(path, glacier_ids):
    """Reads MEaSUREs Shapefiles into dictionary with date keys and polyline values."""
    result = defaultdict(list)
    file_paths = glob.glob(os.path.join(path, '*', '*.shp'))
    for file_path in file_paths:
        with fiona.open(file_path, 'r', encoding='utf-8') as shp:
            for feature in shp:
                if feature['properties']['GlacierID'] in glacier_ids:
                    if 'DateRange' in feature['properties']:
                        polyline_coords = feature['geometry']['coordinates']
                        date_range = feature['properties']['DateRange'].split('-')
                        date_time_objs = [datetime.strptime(date_range[0], '%d%b%Y'), datetime.strptime(date_range[1], '%d%b%Y')]
                        date_time_objs = [datetime.strftime(date_time_objs[0], '%Y-%m-%d'), datetime.strftime(date_time_objs[1], '%Y-%m-%d')]
                        result[date_time_objs[0]].append({'coords': polyline_coords, 'end_date':date_time_objs[1]})
                    elif 'DATE' in feature['properties']:
                        polyline_coords = feature['geometry']['coordinates']
                        date = feature['properties']['DATE']
                        result[date].append({'coords': polyline_coords})
    return result

def promice_read(paths):
    """Reads ESA-CCI Shapefiles into dictionary with date keys and polyline values."""
    result = defaultdict(list)
    inProj = Proj('epsg:4326')
    outProj = Proj('epsg:3413')
    file_paths = []
    for path in paths:
        file_paths += glob.glob(os.path.join(path, '*', '*.shp'))
    
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
                result[date].append({'coords': polyline_coords})
            except Exception as e:
                print(e)
                print('skipping:', file_path)
    return result

def centerline_read(path, ref_name):
    """Reads centerline Shapefile."""
    result = []
    with fiona.open(path, 'r', encoding='utf-8') as shp:
        for feature in shp:
            if feature['properties']['RefName'] == ref_name:
                result.append({'coords':LineString(feature['geometry']['coordinates']), 
                               'GlacierID':feature['properties']['GlacierID'],
                               'BranchID':feature['properties']['BranchID'],
                               'BranchName':feature['properties']['BranchName']})
    return result

def region_read(path):
    """Reads centerline Shapefile."""
    result = []
    with fiona.open(path, 'r', encoding='utf-8') as shp:
        for feature in shp:
            result.append({'coords':Polygon(feature['geometry']['coordinates'][0]), 
                           'Name':feature['properties']['Name']})
    return result


def centerline_intersection(line_dict, centerlines):
    """Calculates intersection point between a list of lines and a centerline.
    Takes in date-intersection dict and centerline to determine change relative to given reference point."""
    results = defaultdict(list)
    #Handle time series
    for date, lines in line_dict.items():
        
        #Handle multiple centerlines
        for i in range(len(centerlines)):
            centerline = centerlines[i]['coords']
            results[date].append(dict())
            
            #Handle multiple calving fronts
            intersection_found = False
            for line in lines:
                line_obj = LineString(line['coords'])
                arclength = 0
                #Handle multi-segment centerlines
                for j in range(len(centerline.coords) - 1):
                    forward = np.array(centerline.coords[j])
                    back = np.array(centerline.coords[j + 1])
                    retreat_vector = back - forward
                    intersect = LineString([forward, back]).intersection(line_obj)
                    if type(intersect) == Point:
                        absolute = np.array(intersect.coords)
                        t = ((absolute - forward) / retreat_vector)[0,0] #calculate percentage change as t
                        arclength += np.linalg.norm(retreat_vector * t)
                        results[date][i]['arclength'] = arclength
                        results[date][i]['center'] = absolute
                        #Handle MEaSUREs
                        if 'end_date' in line:
                            results[date][i]['end_date'] = line['end_date']
                        intersection_found = True
                        break
                    else:
                        arclength += np.linalg.norm(retreat_vector)
                #If intersection is found, skip the other calving fronts, if any
                if intersection_found:
                    break
    return results

def calculate_relative_change(line_dict, centerlines, reference_intersects):
    """Calculates intersection point between a list of lines and a centerline.
    Takes in date-intersection dict and centerline to determine change relative to given reference point."""
    #Handle time series
    for date, intersects in line_dict.items():
        #Handle multiple centerlines
        for i in range(len(centerlines)):
            if 'arclength' in intersects[i]:
            # if intersects[i]['arclength'] > 0:
                try:
                    intersects[i]['relative'] = -(reference_intersects[i]['arclength'] - intersects[i]['arclength']) / 1000
                except:
                    print("Error:", date, reference_intersects[i])
    return line_dict

def get_region(regions, mean_center):
    center = Point(mean_center)
    for region in regions:
        if center.within(region['coords']):
            return region['Name']
    return 'ALL'
    
def generate_centerline_graphs(path, mappings, regions, domains, interp_results_dotted, interp_results_lined, dest_path):
    domain = os.path.basename(path).split('_')[2]
    # if domain not in domains:
    #     return
    
    calfin_path = path
    centerline_path = r"../postprocessing/centerlines/Centerlines.shp"
    
    #Read centerline and calfin
    centerlines = centerline_read(centerline_path, domain.replace('-', ' '))
    polylines_calfin = calfin_read(calfin_path)
    line_intersects_calfin = centerline_intersection(polylines_calfin, centerlines)
    calfin_dates = sorted(line_intersects_calfin.keys())
    if len(calfin_dates) == 0:
        return
    
    #Ensure there exists reference points for all centerlines
    for i in range(len(calfin_dates)):
        reference_point = line_intersects_calfin[calfin_dates[i]]
        if {} not in reference_point:
            break
        
    line_intersects_calfin = calculate_relative_change(line_intersects_calfin, centerlines, reference_point)
    line_dict_list = []
    
    #Read ESA-CCI, PROMICE, and MEaSURES
    if domain in mappings and True:
        if len(mappings[domain]['esa']) > 0:
            esacci_paths = []
            for path in mappings[domain]['esa']:
                esacci_paths.append(r"../postprocessing/esacci/Products/v3.0/" + path)
            polylines_esacci = esacci_read(esacci_paths)
            line_intersects = centerline_intersection(polylines_esacci, centerlines)
            line_intersects = calculate_relative_change(line_intersects, centerlines, reference_point)
            line_dict_list.append(line_intersects)
        if len(mappings[domain]['measures']) > 0:
            measures_path = r"../postprocessing/measures"
            polylines_measures = measures_read(measures_path, mappings[domain]['measures'])
            line_intersects = centerline_intersection(polylines_measures, centerlines)
            line_intersects = calculate_relative_change(line_intersects, centerlines, reference_point)
            line_dict_list.append(line_intersects)
        if len(mappings[domain]['promice']) > 0:
            promice_paths = []
            for path in mappings[domain]['promice']:
                promice_paths.append(r"../postprocessing/promice/" + path + "_frontlines_4326")
            polylines_promice = promice_read(promice_paths)
            line_intersects = centerline_intersection(polylines_promice, centerlines)
            line_intersects = calculate_relative_change(line_intersects, centerlines, reference_point)
            line_dict_list.append(line_intersects)
    line_dict_list.append(line_intersects_calfin)
    
    graph_change(line_dict_list, centerlines, dest_path, domain, regions, interp_results_dotted, interp_results_lined)


def graph_change(line_dict_list, centerlines, dest_path, domain, regions, interp_results_dotted, interp_results_lined):
    """Takes in list of dicts and plots them."""
    # Converter function
    datefunc = lambda x: mdates.date2num(datetime.strptime(x, '%Y-%m-%d'))

    #Initialize plots
    fig, ax = plt.subplots(1, 1, num=domain + ' Relative Length Change, 1972-2019')
    fig.suptitle(domain + ' Relative Length Change, 1972-2019', fontsize=34, fontweight='bold')
    plt.subplots_adjust(top=0.900, bottom=0.1, right=0.95, left=0.05, hspace=0.25, wspace=0.25)
    
    # ax.set_title('CALFIN vs ESA-CCI vs MEaSUREs vs PROMICE', fontsize=18)
    ax.set_xlabel('Year', fontsize=28, fontweight='bold')
    ax.set_ylabel('Relative Length Retreat (km)', fontsize=28, fontweight='bold')
    
    calfin_colors = ['blue', 'lightskyblue', 'midnightblue', 'black']
    esacci_colors = ['orangered', 'orange', 'firebrick', 'maroon']
    measures_colors = ['magenta', 'pink', 'darkviolet', 'indigo']
    promice_colors = ['limegreen', 'lightgreen', 'green', 'darkgreen']
    
    if len(centerlines) > 4:
        raise Exception('Error: Define more colors for additional branches')
    
    start_year = 1972.75
    middle_year = 1985.75
    end_year = 2019.5
    dotted_months = int((middle_year - start_year) * 12) #Start Sept. 1972, end June 2019
    lined_months = int((end_year - middle_year) * 12) #Start Sept. 1972, end June 2019
    start_time = datefunc('1972-09-01')
    middle_time = datefunc('1985-01-01')
    end_time = datefunc('2019-07-01')

    interp_dates_dotted = np.linspace(start_time, middle_time, dotted_months)
    interp_dates_lined = np.linspace(middle_time, end_time, lined_months)
    
    #Handle multiple datasets
    for i in range(len(line_dict_list)):
        line_dict = line_dict_list[i]
        #Handle multiple centerlines
        for j in range(len(centerlines)):
            centerline = centerlines[j]
            if centerline['BranchName'] != None:
                line_id = 'Branch ' + str(centerline['BranchName'])
            else:
                line_id = ''
            dates = []
            changes = []
            line_export = []
            #Handle time series
            for date in sorted(line_dict.keys()):           
                lines = line_dict[date]
                line = lines[j]
                if 'relative' in line:
                    dates.append(datefunc(date))
                    changes.append(line['relative'])
                    line_export.append(line['center'])
                    #Handle measures
                    if 'end_date' in line:
                        dates.append(datefunc(line['end_date']))
                        changes.append(line['relative'])
            if i == len(line_dict_list) - 1: #CALFIN
                # changes = savgol_filter(changes, 3, 2)
                f = interp1d(dates, changes, bounds_error=False)
                interp_changes_dotted = f(interp_dates_dotted)
                interp_changes_lined = f(interp_dates_lined)
                # derivative(f, 1.0, dx=1e-6)
                middle_index = int(np.argwhere(np.array(dates) > middle_time)[0])
                dotted_dates = dates[:middle_index+1]
                lined_dates = dates[middle_index:]
                dotted_changes = changes[:middle_index+1]
                lined_changes = changes[middle_index:]
                
                ax.plot_date(dotted_dates, dotted_changes, ls='--', linewidth=3, marker='o', c=calfin_colors[j])
                ax.plot_date(lined_dates, lined_changes, ls='-', linewidth=2, marker='o', c=calfin_colors[j], label='CALFIN ' + line_id)
                
                coords = np.squeeze(np.array(line_export))
                mean_center = np.mean(coords,axis=0)
                region = get_region(regions, mean_center)
                interp_results_dotted[region].append(interp_changes_dotted)
                interp_results_lined[region].append(interp_changes_lined)
                interp_results_dotted['Greenland'].append(interp_changes_dotted)
                interp_results_lined['Greenland'].append(interp_changes_lined)
                
                # plt.figure(domain + str(j))
                # coords = np.squeeze(np.array(line_export))
                # plt.plot(coords[:,0], coords[:,1])
                # error()
            elif i == 0: #ESA-CCI
                ax.plot_date(dates, changes, ls='', marker='*',  ms=16, c=esacci_colors[j], label='ESA-CCI ' + line_id)
            elif i == 1: #MEaSUREs
                ax.plot_date(dates, changes, ls='', marker='P',  ms=12, c=measures_colors[j], label='MEaSUREs ' + line_id)
            elif i == 2: #PROMICE
                ax.plot_date(dates, changes, ls='', marker='o',  ms=12, c=promice_colors[j], label='PROMICE ' + line_id)
            else:
                ax.plot_date(dates, changes, ls='', marker='o', ms=16)
            
    plt.xlim(start_time, end_time)
    min_loc = YearLocator(1)
    maj_loc = YearLocator(5)
    min_formatter = mdates.DateFormatter('')
    maj_formatter = mdates.DateFormatter('%Y')
    ax.xaxis.set_minor_locator(min_loc)
    ax.xaxis.set_minor_formatter(min_formatter)
    ax.xaxis.set_major_locator(maj_loc)
    ax.xaxis.set_major_formatter(maj_formatter)
    ax.xaxis.set_tick_params(labelsize=22)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")   
    ax.set_xticklabels([], minor=True)
    ax.grid(True, which='both', linewidth=1.5)
    ax.legend()
    plt.show()
        
    plot_path = os.path.join(dest_path, domain + '_relative_change.png')
    fig.savefig(plot_path, bbox_inches='tight', pad_inches=0.05, frameon=False)
    
def plot_mean_results(interp_changes_dotted, interp_changes_lined, region, count, dest_path):
    # Converter function
    datefunc = lambda x: mdates.date2num(datetime.strptime(x, '%Y-%m-%d'))
    
    fig, ax = plt.subplots(1, 1, num=region + ' Mean Relative Length Change (' +  str(count) + ' glaciers)')
    fig.suptitle(region + ' Relative Length Change (' +  str(count) + ' glaciers)', fontsize=34, fontweight='bold')
    plt.subplots_adjust(top=0.900, bottom=0.1, right=0.95, left=0.05, hspace=0.25, wspace=0.25)
    
    ax.set_xlabel('Year', fontsize=28, fontweight='bold')
    ax.set_ylabel('Mean Relative Length Retreat (km)', fontsize=28, fontweight='bold')
    
    region_colors = {'NO':'limegreen', 'CE':'blue', 'NW':'firebrick', 'CW':'pink', 'SW':'darkgreen', 'SE':'gold', 'Gr':'black', 'NE':'purple'}
    
    start_year = 1972.75
    middle_year = 1985.75
    end_year = 2019.5
    dotted_months = int((middle_year - start_year) * 12) #Start Sept. 1972, end June 2019
    lined_months = int((end_year - middle_year) * 12) #Start Sept. 1972, end June 2019
    start_time = datefunc('1972-09-01')
    middle_time = datefunc('1985-01-01')
    end_time = datefunc('2019-07-01')

    interp_dates_dotted = np.linspace(start_time, middle_time, dotted_months)
    interp_dates_lined = np.linspace(middle_time, end_time, lined_months)
    ax.plot_date(interp_dates_dotted[:-2], interp_changes_dotted[:-2], ls='--', linewidth=4, marker='', c=region_colors[region[0:2]])
    ax.plot_date(interp_dates_lined[:-2], interp_changes_lined[:-2], ls='-', linewidth=4, marker='', c=region_colors[region[0:2]])
    
    plt.xlim(start_time, end_time)
    min_loc = YearLocator(1)
    maj_loc = YearLocator(5)
    min_formatter = mdates.DateFormatter('')
    maj_formatter = mdates.DateFormatter('%Y')
    ax.xaxis.set_minor_locator(min_loc)
    ax.xaxis.set_minor_formatter(min_formatter)
    ax.xaxis.set_major_locator(maj_loc)
    ax.xaxis.set_major_formatter(maj_formatter)
    ax.xaxis.set_tick_params(labelsize=22)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")   
    ax.set_xticklabels([], minor=True)
    ax.grid(True, which='both', linewidth=1.5)
    ax.legend()
    plt.show()
    
    fig.savefig(os.path.join(dest_path, region.replace(' ', '_') +'_mean_relative_change.png'), bbox_inches='tight', pad_inches=0, frameon=False)
    
if __name__ == "__main__":
    #Graph results
    #Set figure size for 1600x900 resolution, tight layout
    plt.close('all')
    plt.rcParams["figure.figsize"] = (22,9)
    plt.rcParams["font.size"] = "20"
    plt.subplots_adjust(top=0.925, bottom=0.1, right=0.95, left=0.05, hspace=0.25, wspace=0.25)
    
    # #['CALFIN', 'ESACCI', 'PROMICE', 'MEaSUREs GlacierID']
    mappings = {'Akullersuup-Sermia': 			{'esa':[], 											'promice':['Akullersuup_Sermia'], 						'measures':[208]},
                  'Docker-Smith-Gletsjer': 			{'esa':[], 											'promice':['Døcker_Smith'], 							'measures':[58, 59, 60]},
                  'Fenris-Gletsjer':  				{'esa':[], 											'promice':['Fenris'], 									'measures':[174]},
                  'Hayes-Gletsjer': 				{'esa':[], 											'promice':['Hayes'], 									'measures':[40]},
                  'Helheim-Gletsjer': 				{'esa':['Helheim_Gletsjer_G321627E66422N'], 		'promice':['Helheim'], 									'measures':[175]},
                  'Inngia-Isbrae':					{'esa':['Inngia_Isbrae_G307495E72082N'], 			'promice':['Ingia'], 									'measures':[19]},
                  'Jakobshavn-Isbrae': 				{'esa':['Jakobshavn_Isbrae_G310846E69083N'], 		'promice':['Jakobshavn'], 								'measures':[3]},
                  'Kangerluarsuup-Sermia': 			{'esa':[], 											'promice':['Kangerdluarssup_Sermia'], 					'measures':[15]},
                  'Kangerlussuaq-Gletsjer': 		{'esa':['Kangerlussuaq_Gletsjer_G326914E68667N'], 	'promice':['Kangerdlugssuaq'], 							'measures':[153]},
                  'Kangerlussuup-Sermia': 			{'esa':[], 											'promice':['Kangerdlugssup_Sermerssua'], 				'measures':[16]},
                  'Kangiata-Nunaata-Sermia': 		{'esa':['Kangiata_Nunaata_Sermia_G310427E64274N'], 	'promice':['KNS'], 										'measures':[207]},
                  'Kangilleq-Kangigdleq-Isbrae': 	{'esa':['Kangilleq_G309415E70752N'], 				'promice':['Kangigdleq'], 								'measures':[12]},
                  'Kong-Oscar-Gletsjer': 			{'esa':['Kong_Oscar_Gletsjer_G300347E76024N'], 		'promice':['Kong_Oscars'], 								'measures':[51]},
                  'Lille-Gletsjer': 				{'esa':[], 											'promice':['Lille'], 									'measures':[10]},
                  'Midgard-Gletsjer': 				{'esa':[], 											'promice':['Midgaard'], 								'measures':[173]},
                  'Alison-Gletsjer': 				{'esa':['Nunatakassaap_Sermia_G304089E74641N'], 	'promice':['Nunatakassaap'], 							'measures':[35]},
                  'Naajarsuit-Sermiat': 			{'esa':[], 											'promice':['Nunatakavsaup'], 							'measures':[25]},
                  'Perlerfiup-Sermia': 				{'esa':['Perlerfiup_Sermia_G309225E70985N'], 		'promice':['Perdlerfiup'], 								'measures':[14]},
                  'Petermann-Gletsjer': 			{'esa':['Petermann_Gletsjer_G299936E80548N'], 		'promice':['Petermann'], 								'measures':[93]},
                  'Rink-Isbrae': 					{'esa':['Rink_Isbrae_G308503E71781N'], 				'promice':['Rink'], 									'measures':[17]},
                  'Sermeq-Avannarleq-69N': 			{'esa':['Sermeq_Avannarleq_G309720E69381N'], 		'promice':['Sermeq_Avannarleq'], 						'measures':[4]},
                  'Sermeq-Silarleq': 				{'esa':[], 											'promice':['Sermeq_Silardleq'], 						'measures':[13]},
                  'Sermilik-Isbrae': 				{'esa':['Sermilik_Brae_G313080E61025N'], 			'promice':['Sermilik'], 								'measures':[11]},
                  'Steenstrup-Gletsjer': 			{'esa':['Steenstrup_Gletsjer_G302212E75326N'], 		'promice':['Steenstrup'], 								'measures':[44]},
                  'Store-Gletsjer': 				{'esa':['Store_Gletsjer_G309511E70416N'], 			'promice':['Store'], 									'measures':[9]},
                  'Upernavik-Isstrom-S': 			{'esa':['Upernavik_Isstroem_G305731E72859N'], 		'promice':['Upernavik*'], 			                   	'measures':[20, 21]},
                  'Upernavik-Isstrom-N-C': 			{'esa':['Upernavik_Isstroem_G305731E72859N'], 		'promice':['Upernavik*'],                               'measures':[22, 23]},
                  'Upernavik-Isstrom-NW': 			{'esa':['Upernavik_Isstroem_G305731E72859N'], 		'promice':['Upernavik*'], 								'measures':[24]}}
        
    path = r"../postprocessing/centerlines/GreenlandRegions.shp"
    regions = region_read(path)
    
    domains = ['Hayes-Gletsjer',  'Jakobshavn-Isbrae', 'Rink-Isbrae', 'Upernavik-Isstrom-N-C', 'Upernavik-Isstrom-S', 
               'Petermann-Gletsjer', 'Kong-Oscar-Gletsjer', 'Kangerlussuaq-Gletsjer', 'Helheim-Gletsjer', 'Kangiata-Nunaata-Sermia']
    # domains = ['Kong-Oscar-Gletsjer']
    paper_path = r"../paper"
    if not os.path.exists(paper_path):
        os.mkdir(paper_path)
    dest_path = r"../paper/length_change"
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
        
    change_dict = dict()
    glaciers = glob.glob('../outputs/upload_production/v1.0/level-1_shapefiles-domain-termini/*_v1.0.shp')
    interp_results_dotted = defaultdict(list)
    interp_results_lined = defaultdict(list)
    for i in range(0, len(glaciers)):
        path = glaciers[i]
        if "_closed_v" not in path:
            generate_centerline_graphs(path, mappings, regions, domains, interp_results_dotted, interp_results_lined, dest_path)
    
    for region in interp_results_dotted.keys():
        if region != 'NO':
            mean_interp_results_dotted = np.nanmean(np.array(interp_results_dotted[region]), axis=0)
            mean_interp_results_lined = np.nanmean(np.array(interp_results_lined[region]), axis=0)
            plot_mean_results(mean_interp_results_dotted, mean_interp_results_lined, region, len(interp_results_dotted[region]), dest_path)
            print(len(mean_interp_results_dotted))
    



