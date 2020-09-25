# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 03:15:59 2020

@author: Daniel
"""
import os, shutil
import matplotlib.pyplot as plt
import numpy as np

os.environ['GDAL_DATA'] = 'D:\\ProgramData\\Anaconda3\\envs\\cfm\\Library\\share\\gdal' #Ensure crs are exported correctly by gdal/osr/fiona

from osgeo import gdal, osr
from shapely.geometry import mapping, Polygon, LineString
import fiona
from fiona.crs import from_epsg
from scipy.special import comb
from scipy.signal import savgol_filter
from skimage.io import imread, imsave
#from rasterio.plot import show
        
def mask_to_shp(settings):
    """Saves a post-processed prediction mask to a calving front shapefile."""
    #write out shp file
    date_index = settings['date_index']
    image_settings = settings['image_settings']
    index = str(image_settings['box_counter'])
    image_name_base = image_settings['image_name_base']
    image_name_base_parts = image_name_base.split('_')
    if len(image_name_base_parts) < 3:
        return
    domain = image_name_base_parts[0]
    date = image_name_base_parts[date_index]
    year = date.split('-')[0]
    shp_name = image_name_base + '_' + index + '_cf.shp'
    tif_name = image_name_base + '.tif'
    source_tif_path = os.path.join(settings['tif_source_path'], domain, year, tif_name)

    #Collate all together
    dest_domain_root_folder = os.path.join(settings['dest_root_path'], 'domain')
    dest_domain_folder = os.path.join(dest_domain_root_folder, domain)
    dest_all_folder = os.path.join(settings['dest_root_path'], 'all')

#    if not os.path.exists(source_tif_path):
#        return
    if not os.path.exists(dest_domain_root_folder):
        os.mkdir(dest_domain_root_folder)
    if not os.path.exists(dest_domain_folder):
        os.mkdir(dest_domain_folder)
    if not os.path.exists(dest_all_folder):
        os.mkdir(dest_all_folder)

    dest_shp_domain_path = os.path.join(dest_domain_folder, shp_name)
    dest_shp_all_path = os.path.join(dest_all_folder, shp_name)
    dest_tif_domain_path = os.path.join(dest_domain_folder, tif_name)
    dest_tif_all_path = os.path.join(dest_all_folder, tif_name)

    front_line = image_settings['polyline_coords']
    
    curve = np.transpose(list(zip(savgol_filter(front_line[1], 9, 3), savgol_filter(front_line[0], 9, 3))))
    vertices = np.array(list(zip(curve[0], curve[1]))).astype(np.float32)
    
    # Load geotiff and get domain layer/bounding box of area to mask
    geotiff = gdal.Open(source_tif_path)

    #Get bounds
    geotransform = geotiff.GetGeoTransform()
    x_min = geotransform[0]
    y_max = geotransform[3]
    x_max = x_min + geotransform[1] * geotiff.RasterXSize
    y_min = y_max + geotransform[5] * geotiff.RasterYSize

    #Get projection
    prj = geotiff.GetProjection()
    srs = osr.SpatialReference(wkt=prj)
    if srs.GetAttrValue("PROJCS|AUTHORITY", 1) is not None:
        rasterCRS = int(srs.GetAttrValue("PROJCS|AUTHORITY", 1))
    elif srs.GetAttrValue("AUTHORITY", 1) is not None:
        rasterCRS = int(srs.GetAttrValue("AUTHORITY", 1))
    else:
        rasterCRS = 32621

    #Transform from scaled pixel coordaintes to fractional scaled fractional original to original image to geotiff coordinates
    full_size = settings['full_size']
    bounding_box = image_settings['actual_bounding_box']
    fractional_bounding_box = np.array(bounding_box) / full_size

    #Transform vertices from scaled subset pixel space to original subset fractional space
    top_left = np.array([fractional_bounding_box[1], fractional_bounding_box[0]])
    scale = np.array([fractional_bounding_box[3] / full_size, fractional_bounding_box[2] / full_size])
    vertices_transformed = (vertices * scale) + top_left

    #Transform vertices from original subset fractional space to geolcated meters space
    top_left = np.array([x_min, y_max])
    scale = np.array([x_max - x_min, y_min - y_max])
    vertices_geolocated = (vertices_transformed * scale) + top_left

    # Define a polygon feature geometry with one attribute
    polyline = LineString(vertices_geolocated)
    schema = {
        'geometry': 'LineString',
        'properties': {'id': 'int'},
    }

    # Write a new Shapefile
    shp_save_paths = [dest_shp_domain_path, dest_shp_all_path]
    tif_save_paths = [dest_tif_domain_path, dest_tif_all_path]
    for i in range(len(shp_save_paths)):
        dest_shp_path = shp_save_paths[i]
        dest_tif_path = tif_save_paths[i]
        with fiona.open(
                dest_shp_path,
                'w',
                driver='ESRI Shapefile',
                crs=from_epsg(rasterCRS),
                schema=schema) as out_shp:
            out_shp.write({
                'geometry': mapping(polyline),
                'properties': {'id': 0},
            })
        shutil.copy2(source_tif_path, dest_tif_path)

#, raw_tif_path
def mask_to_polygon_shp(image_name_base, id_str, front_lines, fjord_boundary_tif, source_tif_path, dest_root_path, domain):
    """Saves a post-processed prediction mask to a calving front shapefile."""
    from rasterio import features
    shp_name = image_name_base + '_cf_closed.shp'

    #Collate all together
    dest_domain_root_folder = os.path.join(dest_root_path, 'domain')
    dest_domain_folder = os.path.join(dest_domain_root_folder, domain)
    dest_all_folder = os.path.join(dest_root_path, 'all')

    if not os.path.exists(source_tif_path):
        return
    if not os.path.exists(dest_domain_root_folder):
        os.mkdir(dest_domain_root_folder)
    if not os.path.exists(dest_domain_folder):
        os.mkdir(dest_domain_folder)
    if not os.path.exists(dest_all_folder):
        os.mkdir(dest_all_folder)

    dest_shp_domain_path = os.path.join(dest_domain_folder, shp_name)
    dest_shp_all_path = os.path.join(dest_all_folder, shp_name)
    
    # vertices =  np.array(list(zip(front_line[1], front_line[0]))).astype(np.float32)
    # curve = bezier_curve(vertices, nTimes=len(front_line[0]) * 5) #switch to savgol to avoid large depth issues
    # interpolated_vertices = np.array(list(zip(curve[0], curve[1])))
    # interpolated_vertices = interpolated_vertices.astype(np.float32)
    # interpolated_vertices = vertices
    
#    Show the line interp diff
#     plt.figure(4000)
#     plt.plot(front_line[0], front_line[1], linewidth=1, color='r')
#     plt.plot(interpolated_vertices[:, 1], interpolated_vertices[:, 0], linewidth=1, color='g')
#     plt.show()
    # Load geotiff and get domain layer/bounding box of area to mask
    geotiff = gdal.Open(source_tif_path)
    band = geotiff.GetRasterBand(1)
    arr = band.ReadAsArray()
    [cols, rows] = arr.shape

    #Get bounds
    geotransform = geotiff.GetGeoTransform()
    x_min = geotransform[0]
    y_max = geotransform[3]
    x_max = x_min + geotransform[1] * geotiff.RasterXSize
    y_min = y_max + geotransform[5] * geotiff.RasterYSize

    #Get projection
    prj = geotiff.GetProjection()
    srs = osr.SpatialReference(wkt=prj)
    if srs.GetAttrValue("PROJCS|AUTHORITY", 1) is not None:
        rasterCRS = int(srs.GetAttrValue("PROJCS|AUTHORITY", 1))
    elif srs.GetAttrValue("AUTHORITY", 1) is not None:
        rasterCRS = int(srs.GetAttrValue("AUTHORITY", 1))
    else:
        rasterCRS = 32621
        
    vertices_geolocated_list = []
    for front_line in front_lines:
        # vertices = list(zip(front_line[1], front_line[0]))
    #     curve = bezier_curve(vertices, nTimes=len(front_line[0]) * 2)
        curve = np.transpose(list(zip(savgol_filter(front_line[1], 9, 3), savgol_filter(front_line[0], 9, 3))))
        interpolated_vertices = np.transpose(np.array(curve))
        interpolated_vertices = interpolated_vertices.astype(np.float32)
        #Transform from scaled pixel coordaintes to fractional scaled fractional original to original image to geotiff coordinates
        #Transform vertices from scaled subset pixel space to original subset fractional space
        top_left = np.array([0, 0])
        scale = np.array([1 / rows, 1 / cols])
        vertices_transformed = (interpolated_vertices * scale) + top_left
    
        #Transform vertices from original subset fractional space to geolcated meters space
        top_left = np.array([x_min, y_max])
        scale = np.array([x_max - x_min, y_min - y_max])
        vertices_geolocated = (vertices_transformed * scale) + top_left
        vertices_geolocated_list.append(vertices_geolocated)
#     #Transform vertices from scaled subset pixel space to original subset fractional space
#     vertices_transformed = interpolated_vertices

#     #Transform vertices from original subset fractional space to geolcated meters space
#     top_left = np.array([x_min, y_max])
#     scale = np.array([(x_max - x_min) / cols, (y_min - y_max) / rows])
#     vertices_geolocated = (vertices_transformed * scale) + top_left
    
#     show(src.read(), transform=src.transform)

    # Define a polygon feature geometry with one attribute
    
    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int'},
    }

    #Save overlay image for quality assurance     
    raw_image_path = os.path.join(dest_root_path, 'quality_assurance', domain, image_name_base + '_' + id_str + '_large_processed_raw.png')
    raw_image = imread(raw_image_path)
    #TODO: Save geotiff overlay_polygon + png for easy filtering?
    # Write a new Shapefile
    # shp_save_paths = [dest_shp_domain_path, dest_shp_all_path]
    shp_save_paths = [dest_shp_domain_path]
#    tif_save_paths = [dest_tif_domain_path, dest_tif_all_path]
    for i in range(len(shp_save_paths)):
        dest_shp_path = shp_save_paths[i]
#        dest_tif_path = tif_save_paths[i]
        with fiona.open(
                dest_shp_path,
                'w',
                driver='ESRI Shapefile',
                crs=from_epsg(rasterCRS),
                schema=schema) as out_shp:
            for i in range(len(vertices_geolocated_list)):
                #Write the polygon out
                vertices_geolocated = vertices_geolocated_list[i]
                polygon = mapping(Polygon(vertices_geolocated))
                out_shp.write({
                    'geometry': polygon,
                    'properties': {'id': i},
                })
                
                #Save overlay image for quality assurance   
                raw_image[:,:,0] = features.rasterize([(polygon, 255)],
                    # all_touched=True,
                    out=raw_image[:,:,0],
                    out_shape=fjord_boundary_tif.shape,
                    transform=fjord_boundary_tif.transform)
                
#        shutil.copy2(source_tif_path, dest_tif_path)
    new_image_name_base = image_name_base
    overlay_polygon_path = os.path.join(dest_root_path, 'quality_assurance', domain, new_image_name_base + '_overlay_polygon.png')
    imsave(overlay_polygon_path, raw_image)


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, num_time=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        n_times is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
        
        Modified through recursion/divide and conquer to allow for very large 
        curves to be smoothed together.
    """
    multiplier = 5
    binomial_coef_overflow_limiter = 128
    num_points = len(points)
    points = np.array(points)
    mixer = 5
    mix_mult = mixer * multiplier
    if num_points > binomial_coef_overflow_limiter:
        thirds_index = num_points//3
        num_time = num_time // 3
        points_a = points[:thirds_index + mixer]
        points_b = points[thirds_index:thirds_index*2 + mixer]
        points_c = points[thirds_index*2:]
        curve_a = bezier_curve(points_a, num_time)
        curve_b = bezier_curve(points_b, num_time)
        curve_c = bezier_curve(points_c, num_time)
        
        curve_c_trunc = curve_c[:, :-mix_mult]
        curve_c_end = curve_c[:, -mix_mult:]
        
        curve_b_start = curve_b[:, :mix_mult]
        curve_b_trunc = curve_b[:, mix_mult:-mix_mult]
        curve_b_end = curve_b[:, -mix_mult:]
        
        curve_a_start = curve_a[:, :mix_mult]
        curve_a_trunc = curve_a[:, mix_mult:]
        
        weights = np.array(list(range(mix_mult)))/mix_mult + 1/(mix_mult*2)
        curve_cb_mixed = curve_c_end * (1 - weights) + curve_b_start * weights
        curve_ba_mixed = curve_b_end * (1 - weights) + curve_a_start * weights
        
        final_curve_x = np.concatenate((curve_c_trunc[0], curve_cb_mixed[0], curve_b_trunc[0], curve_ba_mixed[0], curve_a_trunc[0]))
        final_curve_y = np.concatenate((curve_c_trunc[1], curve_cb_mixed[1], curve_b_trunc[1], curve_ba_mixed[1], curve_a_trunc[1]))
        
        return np.array([final_curve_x, final_curve_y])
    else:
        xPoints = points[:, 0]
        yPoints = points[:, 1]

        t = np.linspace(0.0, 1.0, num_time)

        polynomial_array = np.array([bernstein_poly(i, num_points-1, t) for i in range(0, num_points)])

        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)

        return np.array([xvals, yvals])
