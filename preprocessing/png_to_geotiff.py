import os, glob
import numpy as np
os.environ['GDAL_DATA'] = r'D:\\ProgramData\\Anaconda3\\envs\\cfm\\Library\\share\\gdal' #Ensure crs are exported correctly by gdal/osr/fiona
import fiona

from collections import defaultdict
from osgeo import gdal, osr
from pyproj import Proj, transform
from skimage.io import imsave, imread
from skimage.transform import resize

def boundsFromShp(domain_shp_path):
	"""Returns QgsRectangle representing bounds of geotiff in projection coordinates
	
	:domain_shp_path: str
	:bounds: QgsRectangle
	"""
	try:
		with fiona.open(domain_shp_path, 'r', encoding='utf-8') as source_shp:
			coords = np.array(list(source_shp)[0]['geometry']['coordinates'])
			coords
			
			x = coords[0,:,0]
			y = coords[0,:,1]
			return {"xMin":float(np.min(x)), "yMin":float(np.min(y)), "xMax":float(np.max(x)), "yMax":float(np.max(y))}
	except Exception as e:
		print(e)
		return None

def pngToGeotiff(array:np.ndarray, bounds:dict, dest_path:str) -> ('Driver', 'Dataset'):
	"""Array > Raster
	Save a raster from a C order array.
	
	:param array: ndarray
	"""
	
	# TODO: Fix X/Y coordinate mismatch and use ns/ew labels to reduce confusion. Also, general cleanup and refactoring.
# 	array = np.flip(array, axis=0)
	h, w = array.shape[:2]
	x_pixels = w  # number of pixels in x
	y_pixels = h  # number of pixels in y
	x_pixel_size = (bounds['xMax'] - bounds['xMin']) / x_pixels  # size of the pixel...		
	y_pixel_size = (bounds['yMax']  - bounds['yMin']) / y_pixels  # size of the pixel...		
	x_min = bounds['xMin'] 
	y_max = bounds['yMax']   # x_min & y_max are like the "top left" corner.
	
	
	driver = gdal.GetDriverByName('GTiff')
	
	dataset = driver.Create(
		dest_path,
		x_pixels,
		y_pixels,
		1,
		gdal.GDT_Float32, )
	
	dataset.SetGeoTransform((
		x_min,	# 0
		x_pixel_size,  # 1
		0,					  # 2
		y_max,	# 3
		0,					  # 4
		-y_pixel_size))  #6
	
	srs = osr.SpatialReference()
	srs.ImportFromEPSG(3413)
	
	dataset.SetProjection(srs.ExportToWkt())
	dataset.GetRasterBand(1).WriteArray(array)
	dataset.FlushCache()  # Write to disk.
	return dataset, dataset.GetRasterBand(1)  #If you need to return, remenber to return  also the dataset because the band don`t live without dataset.


fjord_boundary_source_path = r'D:\Daniel\Documents\Github\CALFIN Repo\training\data\fjord_boundaries'
domain_path = r'D:\Daniel\Documents\Github\CALFIN Repo\preprocessing\domains'
fjord_boundary_tif_path = r'D:\Daniel\Downloads'

#Use domain shapefiles
for fjord_boundary_path in [os.path.join(fjord_boundary_tif_path, "123BedDomainClippedBinarySmoothed.tif")]:
	basename = os.path.basename(fjord_boundary_path)
	stripped_basename = os.path.splitext(basename)[0]
	domain_shp_path = r"D:\Daniel\Downloads\123BedDomain.shp"
	fjord_boundary_img = imread(fjord_boundary_path, as_gray = True).astype(np.float32)
	shpBounds = boundsFromShp(domain_shp_path)
	if shpBounds is not None:
		dest_path = os.path.join(fjord_boundary_tif_path, '123BedDomainClippedBinarySmoothedGeoref.tif')
		print(dest_path)
		pngToGeotiff(fjord_boundary_img, shpBounds, dest_path)
# 		break