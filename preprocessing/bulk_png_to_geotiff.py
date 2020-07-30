import os, glob
import numpy as np
os.environ['GDAL_DATA'] = r'D:\\ProgramData\\Anaconda3\\envs\\cfm\\Library\\share\\gdal' #Ensure crs are exported correctly by gdal/osr/fiona
import fiona

from collections import defaultdict
from osgeo import gdal, osr
from pyproj import Proj, transform
from skimage.io import imsave, imread
from skimage.transform import resize


name_id_dict = dict()
name_id_dict['Akullikassaap'] = 32621
name_id_dict['Alangorssup'] = 32621
name_id_dict['Alanngorliup'] = 32621
name_id_dict['Brückner'] = 32624
name_id_dict['Christian-IV'] = 32624
name_id_dict['Cornell'] = 32621
name_id_dict['Courtauld'] = 32624
name_id_dict['Dietrichson'] = 32621
name_id_dict['Docker-Smith'] = 32621
name_id_dict['Eqip'] = 32621
name_id_dict['Fenris'] = 32624
name_id_dict['Frederiksborg'] = 32624
name_id_dict['Gade'] = 32621
name_id_dict['Glacier-de-France'] = 32624
name_id_dict['Hayes'] = 32621
name_id_dict['Heim'] = 32624
name_id_dict['Helheim'] = 32624
name_id_dict['Hutchinson'] = 32624
name_id_dict['Illullip'] = 32621
name_id_dict['Inngia'] = 32621
name_id_dict['Issuusarsuit'] = 32621
name_id_dict['Jakobshavn'] = 32621
name_id_dict['Kakiffaat'] = 32621
name_id_dict['Kangerdluarssup'] = 32621
name_id_dict['Kangerlussuaq'] = 32624
name_id_dict['Kangerlussuup'] = 32621
name_id_dict['Kangiata-Nunaata'] = 32621
name_id_dict['Kangilerngata'] = 32621
name_id_dict['Kangilinnguata'] = 32621
name_id_dict['Kangilleq'] = 32621
name_id_dict['Kjer'] = 32621
name_id_dict['Kong-Oscar'] = 32621
name_id_dict['Kælvegletscher'] = 32624
name_id_dict['Lille'] = 32621
name_id_dict['Midgård'] = 32624
name_id_dict['Morell'] = 32621
name_id_dict['Nansen'] = 32621
name_id_dict['Narsap'] = 32621
name_id_dict['Nordenskiold'] = 32621
name_id_dict['Nordfjord'] = 32624
name_id_dict['Nordre-Parallelgletsjer'] = 32624
name_id_dict['Nunatakassaap'] = 32621
name_id_dict['Nunatakavsaup'] = 32621
name_id_dict['Perlerfiup'] = 32621
name_id_dict['Polaric'] = 32624
name_id_dict['Qeqertarsuup'] = 32621
name_id_dict['Rink-Gletsjer'] = 32621
name_id_dict['Rink-Isbrae'] = 32621
name_id_dict['Rosenborg'] = 32624
name_id_dict['Saqqarliup'] = 32621
name_id_dict['Sermeq-Avannarleq-69'] = 32621
name_id_dict['Sermeq-Avannarleq-70'] = 32621
name_id_dict['Sermeq-Avannarleq-73'] = 32621
name_id_dict['Sermeq-Kujalleq-70'] = 32621
name_id_dict['Sermeq-Kujalleq-73'] = 32621
name_id_dict['Sermeq-Silarleq'] = 32621
name_id_dict['Sermilik'] = 32621
name_id_dict['Sorgenfri'] = 32624
name_id_dict['Steenstrup'] = 32621
name_id_dict['Store'] = 32621
name_id_dict['Styrtegletsjer'] = 32624
name_id_dict['Sverdrup'] = 32621
name_id_dict['Søndre-Parallelgletsjer'] = 32624
name_id_dict['Umiammakku'] = 32621
name_id_dict['Upernavik-NE'] = 32621
name_id_dict['Upernavik-SE'] =  32621


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

def pngToGeotiff(array:np.ndarray, domain:str, bounds:dict, dest_path:str) -> ('Driver', 'Dataset'):
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
	if domain in name_id_dict:
		srs.ImportFromEPSG(name_id_dict[domain])
	else:
		srs.ImportFromEPSG(3413)
	
	dataset.SetProjection(srs.ExportToWkt())
	dataset.GetRasterBand(1).WriteArray(array)
	dataset.FlushCache()  # Write to disk.
	return dataset, dataset.GetRasterBand(1)  #If you need to return, remenber to return  also the dataset because the band don`t live without dataset.


fjord_boundary_source_path = r'D:\Daniel\Documents\Github\CALFIN Repo\training\data\fjord_boundaries'
domain_path = r'D:\Daniel\Documents\Github\CALFIN Repo\preprocessing\domains'
fjord_boundary_tif_path = r'D:\Daniel\Documents\Github\CALFIN Repo\training\data\fjord_boundaries_tif'

for fjord_boundary_path in glob.glob(os.path.join(fjord_boundary_source_path, '*')):
	basename = os.path.basename(fjord_boundary_path)
	stripped_basename = os.path.splitext(basename)[0]
	domain = basename.split('_')[0]
	domain_shp_path = os.path.join(domain_path, domain + '.shp')
	fjord_boundary_img = imread(fjord_boundary_path, as_gray = True).astype(np.float32)
	shpBounds = boundsFromShp(domain_shp_path)
	if shpBounds is not None:
		dest_path = os.path.join(fjord_boundary_tif_path, stripped_basename + '.tif')
		print(dest_path)
		pngToGeotiff(fjord_boundary_img, domain, shpBounds, dest_path)
# 		break