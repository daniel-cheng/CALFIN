# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:28:22 2019

@author: Daniel
"""
import matplotlib.pyplot as plt
import numpy as np
import os, shutil
from osgeo import gdal, osr
from shapely.geometry import mapping, Polygon, LineString
import fiona
from fiona.crs import from_epsg
from scipy.special import comb

def mask_to_shp(settings, metrics):
	#write out shp file
	date_index = settings['date_index']
	image_settings = settings['image_settings']
	index = str(image_settings['box_counter'])
	image_name_base = image_settings['image_name_base']
	image_name_base_parts = image_name_base.split('_')
	if len(image_name_base_parts) < 3:
		return
	domain = image_name_base_parts[0]
	date =  image_name_base_parts[date_index]
	year = date.split('-')[0]
	shp_name = image_name_base + '_' + index + '_cf.shp'
	tif_name = image_name_base + '.tif'
	source_tif_path = os.path.join(settings['tif_source_path'], domain, year, tif_name)
	
	#Collate all together
	dest_domain_root_folder = os.path.join(settings['dest_root_path'], 'domain')
	dest_domain_folder = os.path.join(dest_domain_root_folder, domain)
	dest_all_folder = os.path.join(settings['dest_root_path'], 'all')
		
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
	dest_tif_domain_path = os.path.join(dest_domain_folder, tif_name)
	dest_tif_all_path = os.path.join(dest_all_folder, tif_name)
	
	front_line = image_settings['polyline_coords']
	
	vertices = list(zip(front_line[1], front_line[0]))
	curve = bezier_curve(vertices, nTimes=len(front_line[0]) * 5)
	interpolated_vertices = np.array(list(zip(curve[0], curve[1])))
	interpolated_vertices = interpolated_vertices.astype(np.float32)
	
	# Load geotiff and get domain layer/bounding box of area to mask
	geotiff = gdal.Open(source_tif_path)
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
	top_left = np.array([fractional_bounding_box[1], fractional_bounding_box[0]])
	scale = np.array([fractional_bounding_box[3] / full_size, fractional_bounding_box[2] / full_size])
	vertices_transformed = (interpolated_vertices * scale) + top_left
	
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
	for dest_shp_path in [dest_shp_domain_path, dest_shp_all_path]:
		with fiona.open(
			dest_shp_path,
			'w',
			driver='ESRI Shapefile',
			crs=from_epsg(rasterCRS),
			schema=schema) as c:
			
			## If there are multiple geometries, put the "for" loop here
			c.write({
				'geometry': mapping(polyline),
				'properties': {'id': 0},
			})
	shutil.copy2(source_tif_path, dest_tif_domain_path)
	shutil.copy2(source_tif_path, dest_tif_all_path)


#beizer curves taken from: https://stackoverflow.com/a/12644499/1905613
def bernstein_poly(i, n, t):
	"""
	 The Bernstein polynomial of n, i as a function of t
	"""
	
	return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
	"""
	   Given a set of control points, return the
	   bezier curve defined by the control points.
	   
	   points should be a list of lists, or list of tuples
	   such as [ [1,1], 
				 [2,3], 
				 [4,5], ..[Xn, Yn] ]
		nTimes is the number of time steps, defaults to 1000
		
		See http://processingjs.nihongoresources.com/bezierinfo/
	"""
	
	nPoints = len(points)
	xPoints = np.array([p[0] for p in points])
	yPoints = np.array([p[1] for p in points])
	
	t = np.linspace(0.0, 1.0, nTimes)
	
	polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
	
	xvals = np.dot(xPoints, polynomial_array)
	yvals = np.dot(yPoints, polynomial_array)
	
	return xvals, yvals