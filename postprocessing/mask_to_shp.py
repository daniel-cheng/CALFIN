# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:28:22 2019

@author: Daniel
"""
import numpy as np
import os, shutil
from osgeo import gdal, osr
from shapely.geometry import mapping, Polygon, LineString
import fiona
from fiona.crs import from_epsg
from error_analysis import extract_front_indicators

def mask_to_shp(settings, metrics):
	#write out shp file
	image_settings = settings['image_settings']
	index = str(image_settings['box_counter'])
	image_name_base = image_settings['image_name_base']
	image_name_base_parts = image_name_base.split('_')
	if len(image_name_base_parts) < 3:
		return
	domain = image_name_base_parts[0]
	date =  image_name_base_parts[3]
	year = date.split('-')[0]
	shp_name = image_name_base + '_' + index + '.shp'
	tif_name = image_name_base + '.tif'
	source_tif_path = os.path.join(settings['tif_source_path'], domain, year, tif_name)
	dest_shp_folder = os.path.join(settings['dest_root_path'], 'shp', domain)
	dest_tif_folder = os.path.join(settings['dest_root_path'], 'tif', domain)
	dest_all_folder = os.path.join(settings['dest_root_path'], 'all', domain)
	dest_shp_year_folder = os.path.join(dest_shp_folder, year)
	dest_tif_year_folder = os.path.join(dest_tif_folder, year)
		
	if not os.path.exists(source_tif_path):
		return
	if not os.path.exists(dest_shp_folder):
		os.mkdir(dest_shp_folder)
	if not os.path.exists(dest_tif_folder):
		os.mkdir(dest_tif_folder)
	if not os.path.exists(dest_all_folder):
		os.mkdir(dest_all_folder)
	if not os.path.exists(dest_shp_year_folder):
		os.mkdir(dest_shp_year_folder)
	if not os.path.exists(dest_tif_year_folder):
		os.mkdir(dest_tif_year_folder)
			
	dest_shp_path = os.path.join(dest_shp_year_folder, shp_name)
	dest_tif_path = os.path.join(dest_tif_year_folder, tif_name)
	dest_shp_all_path = os.path.join(dest_all_folder, shp_name)
	dest_tif_all_path = os.path.join(dest_all_folder, tif_name)

	
	# Here's an example Shapely geometry
	polyline_image = image_settings['polyline_image']
	results_polyline = extract_front_indicators(polyline_image[:,:,0])
	front_line = results_polyline[1]
	vertices = list(zip(front_line[1], front_line[0]))
#	vertex_list.append(vertex_list[0])
	vertices = np.array(vertices)
	vertices = vertices.astype(np.float32)
	
	# Load geotiff and get domain layer/bounding box of area to mask
	geotiff = gdal.Open(source_tif_path)
	print(source_tif_path)
	prj = geotiff.GetProjection()
	srs = osr.SpatialReference(wkt=prj)
	rasterCRS = srs.GetAttrValue("PROJCS|AUTHORITY", 1)
	if (rasterCRS is None):
		rasterCRS = srs.GetAttrValue("AUTHORITY", 1)
	rasterCRS = int(rasterCRS)
	
	#Get bounds
	geoTransform = geotiff.GetGeoTransform()
	xMin = geoTransform[0]
	yMax = geoTransform[3]
	xMax = xMin + geoTransform[1] * geotiff.RasterXSize
	yMin = yMax + geoTransform[5] * geotiff.RasterYSize
	
	#Transform from scaled pixel coordaintes to fractional scaled fractional original to original image to geotiff coordinates
	full_size = settings['full_size']
	bounding_box = image_settings['actual_bounding_box']
	fractional_bounding_box = np.array(bounding_box) / full_size
	
	#Transform vertices from scaled subset pixel space to original subset fractional space
	top_left = np.array([fractional_bounding_box[0], fractional_bounding_box[1]])
	scale = np.array([fractional_bounding_box[2] / full_size, fractional_bounding_box[3] / full_size])
	vertices_transformed = (vertices * scale) + top_left
	
	#Transform vertices from original subset fractional space to geolcated meters space
	top_left = np.array([xMin, yMax])
	scale = np.array([xMax - xMin, yMin - yMax])
	vertices_geolocated = (vertices_transformed * scale) + top_left
	
	# Define a polygon feature geometry with one attribute
	polyline = LineString(vertices_geolocated)
	schema = {
		'geometry': 'LineString',
		'properties': {'id': 'int'},
	}
	
	# Write a new Shapefile
	with fiona.open(
			dest_shp_all_path,
			'w',
			driver='ESRI Shapefile',
			crs=from_epsg(rasterCRS),
			schema=schema) as c:
		
		## If there are multiple geometries, put the "for" loop here
		c.write({
			'geometry': mapping(polyline),
			'properties': {'id': 0},
		})
		shutil.copy2(dest_shp_all_path, dest_shp_path)
		shutil.copy2(source_tif_path, dest_tif_path)
		shutil.copy2(source_tif_path, dest_tif_all_path)