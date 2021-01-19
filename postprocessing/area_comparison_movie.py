# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:38:07 2019

@author: Daniel
"""
import numpy as np
import cv2
import os, shutil, glob, copy, sys
from datetime import datetime
from PIL import Image
os.environ['GDAL_DATA'] = r'D://ProgramData//Anaconda3//envs//cfm//Library//share//gdal' #Ensure crs are exported correctly by gdal/osr/fiona

from osgeo import gdal, osr
import rasterio
from rasterio import features
from rasterio.windows import from_bounds
from scipy.ndimage.morphology import distance_transform_edt
from scipy.spatial import KDTree
from scipy.ndimage import median_filter, gaussian_filter1d
from skimage.io import imsave, imread
from skimage import measure
from skimage.transform import resize
from skimage.morphology import skeletonize
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

from matplotlib.ticker import FormatStrFormatter
from matplotlib.dates import MonthLocator, YearLocator
from pyproj import Proj, transform
from shapely.geometry import mapping, Polygon, LineString, Point
from collections import defaultdict
import fiona
from fiona.crs import from_epsg
from ordered_line_from_unordered_points import ordered_line_from_unordered_points_tree, is_outlier


def get_tif_path_from_shp_feature(domain, feature):
    quality = feature['properties']['QualFlag']
    date = feature['properties']['Date']
    basename = feature['properties']['ImageID']
    # print(date, basename)
    basename_parts = basename.split('_')
    
    satellite = basename_parts[0]
    level = basename_parts[1]
    path_row = basename_parts[2]
    tier = basename_parts[6]
    
    path = path_row[0:3]
    row = path_row[3:6]
    pathrow = path + '-' + row
    
    if satellite in ['LC08']:
        band = 'B5'
    elif satellite in ['LE07', 'LT05', 'LT04', 'LM05', 'LM04']:
        band = 'B4'    
    elif satellite in ['LM03', 'LM02', 'LM01']:
        band = 'B7'
    else:
        band = 'B5'
    
    if quality in [0, 3]:
        source_path = r'../outputs/mask_extractor/domain'
    elif quality in [10, 13]:
        source_path = r'../outputs/production_staging/domain'
       
    suffix = '.tif'
    file_name = '_'.join([domain, satellite, level, date, pathrow, tier, '*']) + suffix
    file_path = os.path.join(source_path, domain, file_name)
    file_path = glob.glob(file_path)[0]
    return file_path


def calfin_read(domain, calfin_path):
    annual_fronts = defaultdict(list)
    all_fronts = defaultdict(lambda: defaultdict(list))
    with fiona.open(calfin_path, 'r', encoding='utf-8') as shp:
        for feature in shp:
            try:
                line_coords = feature['geometry']['coordinates']
                date = feature['properties']['Date']
                datetuple = datetime.strptime(date,'%Y-%m-%d').timetuple()
                year = datetuple.tm_year
                year_days = datetuple.tm_yday
                year_days_fraction = year_days / 365
                path = get_tif_path_from_shp_feature(domain, feature)
                
                geometry = {'type': 'LineString'}
                geometry['coordinates'] = line_coords
                all_fronts[year][date].append({'geometry': geometry, 'fraction': year_days_fraction, 'year':year, 'path': path})
            except AttributeError as e:
                print(e)
                print('Missing data near:', domain, date)
        #Get annual fronts closest to midyear
        midyear = 0.5
        for year in all_fronts.keys():
            first_key = list(all_fronts[year].keys())[0] 
            annual_fronts[year] = all_fronts[year][first_key]
            for date in all_fronts[year].keys():
                #for an arbitray front (the 0th), check to see if this date is closer than the first one to midyear
                if midyear - all_fronts[year][date][0]['fraction'] < midyear - annual_fronts[year][0]['fraction']:
                    annual_fronts[year] = all_fronts[year][date]
    # print(all_fronts, annual_fronts) 
    return all_fronts, annual_fronts


def generate_movies():
    fjord_boundary_path = r'../training/data/fjord_boundaries_tif'
    domain_path = r'../preprocessing/domains'
    movie_path = r'../paper/movies'
    
    official_domains = ['Hayes-Gletsjer', 'Helheim-Gletsjer', 'Jakobshavn-Isbrae', 'Kangerlussuaq-Gletsjer', 
           'Kangiata-Nunaata-Sermia', 'Kong-Oscar-Gletsjer', 'Petermann-Gletsjer', 'Rink-Isbrae', 
           'Upernavik-Isstrom-N-C', 'Upernavik-Isstrom-S']
    domains = ['Hayes', 'Helheim', 'Jakobshavn', 'Kangerlussuaq', 
           'Kangiata-Nunaata', 'Kong-Oscar', 'Petermann', 'Rink-Isbrae', 
           'Upernavik-NE', 'Upernavik-SE']
    colormap_annual = matplotlib.cm.get_cmap('viridis')
    colormap_seasonal = matplotlib.cm.get_cmap('plasma')
    # for i in range(3, len(domains)):
    #     official_domain = official_domains[i]
    #     domain = domains[i]
    #     fjord_boundary_file_path = os.path.join(fjord_boundary_path, domain + '_fjord_boundaries.tif')
    #     fjord_boundary_overrides_file_path = os.path.join(fjord_boundary_path, domain + '_fjord_boundaries_overrides.tif')
    #     calfin_path = '../outputs/upload_production/v1.0/level-1_shapefiles-domain-termini/termini_1972-2019_' + official_domain + '_v1.0.shp'
        
    #     #Read in all fronts
    #     all_fronts, annual_fronts = calfin_read(domain, calfin_path)
        
    #     #Get projection
    #     domain_prj_path = os.path.join(domain_path, domain + '.prj')
    #     prj_txt = open(domain_prj_path, 'r').read()
       
    #     #Must project epsg:3413 points to polygon's projection to allow for easy bounding box point-in-polygon testing
    #     #otherwise the boxes will be skewed
    #     srs = osr.SpatialReference()
    #     srs.ImportFromESRI([prj_txt])
    #     srs.AutoIdentifyEPSG()
    #     code = srs.GetAuthorityCode(None)
        
    #     in_proj = Proj(init='epsg:3413')
    #     out_proj = Proj(init='epsg:' + code)
    
    #     print('Now starting:', domain)
    #     #Generate frames per year
    #     with rasterio.open(fjord_boundary_file_path) as fjord_boundary_tif:
    #         fjord_boundary = fjord_boundary_tif.read(1)
    #         fjord_bounds = rasterio.transform.array_bounds(fjord_boundary.shape[0], fjord_boundary.shape[1], fjord_boundary_tif.transform)
    #         annual_frame = np.zeros((fjord_boundary.shape[0], fjord_boundary.shape[1], 3))
            
    #         output_domain_path = os.path.join(movie_path, domain)
    #         if not os.path.exists(output_domain_path):
    #             os.mkdir(output_domain_path)
                        
    #         for year in all_fronts.keys():
    #             raw_tif_path = annual_fronts[year][0]['path']
                
    #             with rasterio.open(raw_tif_path) as raw_tif:
    #                 # Read croped array
    #                 x_min_fjord = fjord_bounds[0]
    #                 y_min_fjord = fjord_bounds[1]
    #                 x_max_fjord = fjord_bounds[2]
    #                 y_max_fjord = fjord_bounds[3]
    #                 window = from_bounds(x_min_fjord, y_min_fjord, x_max_fjord, y_max_fjord, raw_tif.transform)
    #                 raw_img = raw_tif.read(1, window=window, out_shape=fjord_boundary.shape, boundless=True, fill_value=0)
    #                 raw_img = raw_img / raw_img.max() * 255.0  * 0.75
                    
    #                 annual_frame = np.stack((raw_img, raw_img, raw_img), axis=2)
    #                 fig = plt.figure()
    #                 gs1 = gridspec.GridSpec(1, 2)
    #                 gs1.update(wspace=0.05, hspace=0.05) # set the spacing between axes. 
    #                 ax1 = plt.subplot(gs1[0])
    #                 ax2 = plt.subplot(gs1[1])
                    
    #                 ax1.imshow(annual_frame.astype(np.uint8), extent=(x_min_fjord, x_max_fjord, y_min_fjord, y_max_fjord))
    #                 ax2.imshow(annual_frame.astype(np.uint8), extent=(x_min_fjord, x_max_fjord, y_min_fjord, y_max_fjord))
                    
    #                 for annual_year, fronts in annual_fronts.items():
    #                     for front in fronts:
    #                         if annual_year <= year:
    #                             year_fraction = (front['year'] - 1972) / (2019 - 1972)
    #                             rgba_weights = colormap_annual(year_fraction)
    #                             geometry = front['geometry']
    #                             coords = np.array(geometry['coordinates'])
                                
    #                             x = coords[:,0]
    #                             y = coords[:,1]
    #                             x, y = transform(in_proj, out_proj, x, y)
    #                             ax1.plot(x, y, c=rgba_weights, label=str(annual_year))
                                
    #                 #Then, print seasonal fronts
    #                 for date, fronts in all_fronts[year].items():
    #                     for front in fronts:
    #                         year_days_fraction = front['fraction']
    #                         rgba_weights = colormap_seasonal(year_days_fraction)
    #                         geometry = front['geometry']
    #                         coords = np.array(geometry['coordinates'])
                            
    #                         x = coords[:,0]
    #                         y = coords[:,1]
    #                         x, y = transform(in_proj, out_proj, x, y)
    #                         ax2.plot(x, y, c=rgba_weights)
                    
                                
    #                 #Annual colorbar
    #                 norm = matplotlib.colors.Normalize(vmin=1972,vmax=2019)
    #                 sm = plt.cm.ScalarMappable(cmap=colormap_annual, norm=norm)
    #                 sm.set_array([])
    #                 N = int((2018-1972) / 2) + 1
    #                 cbb = fig.colorbar(sm, ticks=np.linspace(1972,2018,N), pad=.025, ax=ax1)
    #                 cbb.ax.tick_params(labelsize=16) 
    #                 cbb.ax.yaxis.set_major_formatter(FormatStrFormatter('%i'))
                    
    #                 #Seasonal colorbar
    #                 norm = matplotlib.colors.Normalize(vmin=1,vmax=12)
    #                 sm = plt.cm.ScalarMappable(cmap=colormap_seasonal, norm=norm)
    #                 sm.set_array([])
    #                 N = 12
    #                 cba = fig.colorbar(sm, ticks=np.linspace(1,12,N), pad=.025, ax=ax2)
    #                 cba.ax.tick_params(labelsize=16) 
                    
    #                 ax1.axis('off')
    #                 ax1.set_title(official_domain + ' Annual Positions, 1972-' + str(year), fontdict={'fontsize': 20})
                    
    #                 ax2.axis('off')
    #                 ax2.set_title(official_domain + ' Seasonal Positions, ' + str(year), fontdict={'fontsize': 20})
                    
    #                 cba.set_label('Month',size=18)
    #                 cbb.set_label('Year',size=18)
                    
    #                 fig.tight_layout()
    #                 print(domain, year)
                    
                    
    #                 plt.savefig(os.path.join(output_domain_path, str(year) + '.png'), dpi=50, bbox_inches='tight')
    #                 plt.close()
                    
    for i in range(len(domains)):
    # for i in range(1):
        official_domain = official_domains[i]
        domain = domains[i]
        output_domain_path = os.path.join(movie_path, domain)
        #save movies/gifs
        save_gif(output_domain_path, movie_path, domain)

def save_gif(output_domain_path, movie_path, domain):
    # filepaths
    fp_in = output_domain_path + "/*.png"
    fp_out = os.path.join(movie_path, 'gifs', domain + '.gif')
    
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=500, loop=0)
    
                    
                    
# for each domain in domains,
#     annual_frame = None
#     get fjord_boundary
#     for each calving front in greenland,
#         get date
#         if date.year != prev_date.year:
#             save current_frame
#             current_frame = get new image tif subset 
#         rasterize front to current_frame
#     compile frames into movie
        
            
if __name__ == "__main__":
    #Set figure size for 1600x900 resolution, tight layout
    plt.close('all')
    plt.rcParams["figure.figsize"] = (22,9)
    plt.rcParams["font.size"] = "20"
    
    all_fronts, annual_fronts = generate_movies()
    plt.show()