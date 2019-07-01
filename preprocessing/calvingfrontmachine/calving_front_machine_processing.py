import numpy as np
import cv2, processing, os, glob, shutil, re
from osgeo import gdal, ogr, osr
from gdalconst import *
from pyproj import Proj, transform
from qgis.core import *

from PyQt5 import QtCore
from PyQt5.QtCore import QFile, QFileInfo
from qgis import core, gui, utils
from qgis.utils import iface
from skimage.io import imsave, imread
from skimage.transform import resize

def geotiffBounds(geotiff) -> QgsRectangle:
	"""Returns QgsRectangle representing bounds of geotiff in projection coordinates
	
	:geotiff: geotiff
	:bounds: QgsRectangle
	"""
	geoTransform = geotiff.GetGeoTransform()
	
	xMin = geoTransform[0]
	yMax = geoTransform[3]
	xMax = xMin + geoTransform[1] * geotiff.RasterXSize
	yMin = yMax + geoTransform[5] * geotiff.RasterYSize
	
	return QgsRectangle(float(xMin), float(yMin), float(xMax), float(yMax))

def geotiffWorldToPixelCoords(geotiff, rectDomain:QgsRectangle, rasterCRS:str, domainCRS:str) -> QgsRectangle:
	"""Transforms QgsRectangle coordinates into geotiff image pixel coordinates

	:geotiff: geotiff
	:rect: QgsRectangle
	"""
	
	# Transform and scale rect by width/height to obtain normalized image coordiantes
	rectRef = geotiffBounds(geotiff)
	rectRefCenter = rectRef.center()
	
	rectRefWidth = rectRef.width()
	rectRefHeight = rectRef.height()
	
	domainX = [rectDomain.xMinimum(), rectDomain.xMaximum()]
	domainY = [rectDomain.yMinimum(), rectDomain.yMaximum()]
	inProj = Proj(init=domainCRS)
	outProj = Proj(init=rasterCRS)
	#print(inProj, outProj, domainCRS, rasterCRS)
	rasterCRSDomainX, rasterCRSDomainY = transform(inProj, outProj, domainX, domainY)
	#print(rasterCRSDomainX, rasterCRSDomainY)
	
	xMin = (rasterCRSDomainX[0] - rectRef.xMinimum()) / rectRefWidth
	xMax = (rasterCRSDomainX[1] - rectRef.xMinimum()) / rectRefWidth
	yMin = (rasterCRSDomainY[0] - rectRef.yMinimum()) / rectRefHeight
	yMax = (rasterCRSDomainY[1] - rectRef.yMinimum()) / rectRefHeight
	
	# Scale by image dimensions to obtain pixel coordinates
	xMin = xMin * geotiff.RasterXSize
	xMax = xMax * geotiff.RasterXSize
	yMin = (1.0 - yMin) * geotiff.RasterYSize
	yMax = (1.0 - yMax) * geotiff.RasterYSize
	
	#print(rasterCRS, domainCRS)
	
	#Return pixel coordinates
	rectOut = QgsRectangle(xMin, yMin, xMax, yMax)
	return rectOut

def arrayToRaster(array:np.ndarray, geotiff, subset:QgsRectangle, destinationPath:str) -> ('Driver', 'Dataset'):
	"""Array > Raster
	Save a raster from a C order array.
	
	:param array: ndarray
	"""
	geoBounds = geotiffBounds(geotiff)
	geoTransform = geotiff.GetGeoTransform()
	
	# TODO: Fix X/Y coordinate mismatch and use ns/ew labels to reduce confusion. Also, general cleanup and refactoring.
	h, w = array.shape[:2]
	x_pixels = w  # number of pixels in x
	y_pixels = h  # number of pixels in y
	x_pixel_size = geoTransform[1]  # size of the pixel...		
	y_pixel_size = geoTransform[5]  # size of the pixel...		
	x_min = geoTransform[0] 
	y_max = geoTransform[3]  # x_min & y_max are like the "top left" corner.
	
	x_subset_percentage = 1.0 - (float(subset.yMinimum()) / float(geotiff.RasterYSize))
	y_subset_percentage = (float(subset.xMinimum()) / float(geotiff.RasterXSize))
	
	y_coordinate_range = geoBounds.width()
	x_coordinate_range = geoBounds.height()
	
	x_offset = x_subset_percentage * x_coordinate_range
	y_offset = y_subset_percentage * y_coordinate_range
	
	x_min = geoBounds.xMinimum() + int(y_offset)
	y_max = geoBounds.yMinimum() + int(x_offset)
	
	driver = gdal.GetDriverByName('GTiff')
	
	dataset = driver.Create(
		destinationPath,
		x_pixels,
		y_pixels,
		1,
		gdal.GDT_Float32, )
	
	dataset.SetGeoTransform((
		x_min,	# 0
		x_pixel_size,  # 1
		geoTransform[2],					  # 2
		y_max,	# 3
		geoTransform[4],					  # 4
		y_pixel_size))  #6
	
	dataset.SetProjection(geotiff.GetProjection())
	dataset.GetRasterBand(1).WriteArray(array)
	dataset.FlushCache()  # Write to disk.
	return dataset, dataset.GetRasterBand(1)  #If you need to return, remenber to return  also the dataset because the band don`t live without dataset.

def resolve(name:str, basepath=None):
	if not basepath:
		basepath = os.path.dirname(os.path.realpath(__file__))
	return os.path.join(basepath, name)

def vectorizeRaster(rasterPath:str, outLineShp:str, lineName:str, outPolygonShp:str, polygonName:str) -> (QgsVectorLayer, QgsVectorLayer):
	"""Description: Creates a vector layer from a raster using processing:polygonize.
		Make sure to save the shapefile, as it will be deleted otherwise! 
		Input:  string rasterPath - path to raster image to polygonize
				string outLineShp - file name to give new line shapefile
				string lineName - layer name to give new line vector layer
				string outLineShp - file name to give new closed polygon shapefile
				string polygonName - layer name to give new closed polygon vector layer
		Output: QgsVectorLayer, QgsVectorLayer - object referencing the new line and polygon vector layers
	"""
	# this allows GDAL to throw Python Exceptions
	gdal.UseExceptions()
	
	# Get raster datasource
	src_ds = gdal.Open(rasterPath)
	srcband = src_ds.GetRasterBand(1)
	prj = src_ds.GetProjection()
	raster_srs = osr.SpatialReference(wkt = prj)
	
	# Create output datasource
	drv = ogr.GetDriverByName("ESRI Shapefile")
	# Remove output shapefile if it already exists)
	if os.path.exists(outLineShp):
		outShapefileBase = outLineShp[0:-4]
		for filename in glob.glob(outShapefileBase + "*.shp"):
			os.remove(filename)
		for filename in glob.glob(outShapefileBase + "*.dbf"):
			os.remove(filename)
		for filename in glob.glob(outShapefileBase + "*.prj"):
			os.remove(filename)
		for filename in glob.glob(outShapefileBase + "*.qpj"):
			os.remove(filename)
		for filename in glob.glob(outShapefileBase + "*.shx"):
			os.remove(filename)
	
	drv = None
	drv = ogr.GetDriverByName("ESRI Shapefile")
	
	processing.run("gdal:contour",
		{"INPUT":rasterPath,
		"BAND":1,
		"INTERVAL":255,
		"FIELD_NAME":"ELEV",
		"CREATE_3D":False, #Bilinear
		"IGNORE_NODATA":False,
		"NODATA":None,
		"OFFSET":0,
		"OUTPUT":outLineShp[0:-4] + '_tmp.shp'
		})
	
	processing.run("native:simplifygeometries",
		{"INPUT":outLineShp[0:-4] + '_tmp.shp',
		"METHOD": 0, #Distance (Douglas-Peucker)
		"TOLERANCE": 20, #20 meter tolerance (Landsat B5 resolution: 30m
		"OUTPUT":outLineShp
		})
	
	processing.run("qgis:linestopolygons",
		{"INPUT":outLineShp,
		"OUTPUT":outPolygonShp
		})
	
	for filename in glob.glob(outLineShp[0:-4] + '_tmp*'):
		os.remove(filename)
	#src_ds = None
	#srcband = None
	#dst_ds = None
	#dst_layer = None
	#drv = None
	#
	## If area is less than inMinSize or if it isn't forest, remove polygon 
	#ioShpFile = ogr.Open(outShapefile, update = 1)
	#layer = ioShpFile.GetLayerByIndex(0)
	#		
	#layer.ResetReading()
	#for feature in layer:
	#	print('feature', feature.GetFID(), feature.GetField('Class'))
	#	layer.SetFeature(feature)
	#	if feature.GetField('Class')==0:
	#		layer.DeleteFeature(feature.GetFID())		
	#ioShpFile.Destroy()
	#ioShpFile = None
	
	return QgsVectorLayer(outLineShp, lineName, 'ogr'), QgsVectorLayer(outPolygonShp, polygonName, 'ogr')

def layerSubsetSave(rasterLayer:QgsRasterLayer, domainLayer:QgsVectorLayer, rootGroup:QgsLayerTreeGroup, name:str) -> None:
	"""Description: Processes a raster image into a vector polygon ocean/land mask.
		Make sure to save the shapefile, as it will be deleted otherwise! 
		Input:  QgsRasterLayer rasterLayer - layer that contains the raster image to process
				QgsVectorLayer domainLayer - layer that contains a polygon specifying the bounds of the raster image to process
				QgsLayerTreeGroup rootGroup - layer that contains the root name of the glacier
		Output: QgsRasterLayer, QgsVectorLayer - objects referencing the new mask layers
	"""
	
	# Get basic file name information on geotiff, raster image, masked raster subset image, and masked vector subset shp file
	fileSource = rasterLayer.source()
	fileInfo = QFileInfo(fileSource)
	filePath = fileInfo.absolutePath()
	fileName = fileInfo.baseName()
	subsetName = fileName + '_' + domainLayer.name()
	subsetPath = getSavePaths(fileSource, domainLayer, 'tif') + '/' + subsetName + '.tif'
	
	# Load geotiff and get domain layer/bounding box of area to mask
	geotiff = gdal.Open(fileSource)
	feature = domainLayer.getFeature(0)
	domain = feature.geometry().boundingBox()
	prj = geotiff.GetProjection()
	srs = osr.SpatialReference(wkt=prj)
	rasterCRS = srs.GetAttrValue("PROJCS|AUTHORITY", 0) + ":" + srs.GetAttrValue("PROJCS|AUTHORITY", 1)
	if (rasterCRS is None):
		rasterCRS = srs.GetAttrValue("AUTHORITY", 0) + ":" + srs.GetAttrValue("AUTHORITY", 1)
	
	crs = rasterLayer.crs()
	crs.createFromId(int(srs.GetAttrValue("AUTHORITY", 1)))
	rasterLayer.setCrs(crs)
	rasterLayer.triggerRepaint()
	
	#rasterCRS = rasterLayer.crs().authid()
	domainCRS = domainLayer.crs().authid()
	bounds = geotiffWorldToPixelCoords(geotiff, domain, rasterCRS, domainCRS)
	
	img = geotiff.GetRasterBand(1)
	img = img.ReadAsArray(0,0,geotiff.RasterXSize,geotiff.RasterYSize).astype(np.uint16)
	img = img[int(round(bounds.yMinimum())):int(round(bounds.yMaximum())), int(round(bounds.xMinimum())):int(round(bounds.xMaximum()))]
	
	print('Save subset:', subsetPath)
	arrayToRaster(img, geotiff, bounds, subsetPath)
	imsave(resolve('landsat_raw/' + domainLayer.name() + '/' + name + '.png'), img)
	
	return img.shape

def layerResize(rasterLayer:QgsRasterLayer, domainLayer:QgsVectorLayer, name:str, resolution:(int, int)) -> None:
	"""Description: Processes a raster image into a vector polygon ocean/land mask.
		Make sure to save the shapefile, as it will be deleted otherwise! 
		Input:  QgsRasterLayer rasterLayer - layer that contains the raster image to process
				QgsVectorLayer domainLayer - layer that contains a polygon specifying the bounds of the raster image to process
				QgsVectorLayer outputLayer - layer to save vector layer in. Warning: not supported yet. 
		Output: QgsRasterLayer, QgsVectorLayer - objects referencing the new mask layers
	"""
	
	path = resolve('landsat_raw/' + domainLayer.name() + '/' + name + '.png')
	img = imread(path)
	img = resize(img, resolution)
	print(img.shape, path)
	imsave(path, img)

def layerWarp(rasterLayer:QgsRasterLayer, domainLayer:QgsVectorLayer) -> None:
	"""Description: Reprojects a raster if not already in the desired CRS.
		Make sure to save the shapefile, as it will be deleted otherwise! 
		Input:  QgsRasterLayer rasterLayer - layer that contains the raster image to process
				QgsVectorLayer domainLayer - layer that contains a polygon specifying the bounds of the raster image to process
	"""
	
	fileSource = rasterLayer.layer().source()
	geotiff = gdal.Open(fileSource)
	prj = geotiff.GetProjection()
	srs = osr.SpatialReference(wkt=prj)
	rasterEpsgCode = srs.GetAttrValue("PROJCS|AUTHORITY", 0) + ":" + srs.GetAttrValue("PROJCS|AUTHORITY", 1)
	domainEpsgCode = domainCRS = domainLayer.crs().authid()
	if (rasterEpsgCode is None):
		rasterEpsgCode = srs.GetAttrValue("AUTHORITY", 0) + ":" + srs.GetAttrValue("AUTHORITY", 1)
	if (rasterEpsgCode != domainEpsgCode):
		parent = rasterLayer.parent()
		rasterName = rasterLayer.layer().name()
		
		QgsProject.instance().removeMapLayer(rasterLayer.layer().id())
		print('warping...', rasterEpsgCode, domainEpsgCode)
		outSource = fileSource[0:-4] + "_" + domainEpsgCode[5:] + ".tif"
		#processing.algorithmHelp("gdal:warpreproject")
		processing.run("gdal:warpreproject",
			{"INPUT":fileSource,
			"SOURCE_SRS":rasterEpsgCode,
			"TARGET_CRS":domainEpsgCode,
			"RESAMPLING":1, #Bilinear
			"NO_DATA":0,
			"DATA_TYPE":5, #Float32
			"MULTITHREADING":True,
			"OUTPUT":outSource})
		geotiff = None
		shutil.copy2(outSource, fileSource)
		os.remove(outSource)
		rasterLayer = QgsRasterLayer(fileSource, rasterName)
		QgsProject.instance().addMapLayer(rasterLayer, False)
		parent.insertLayer(0, rasterLayer)
		return rasterLayer
	return rasterLayer.layer()

def getVectorNames(fileName:str, domainName:str):
	"""Description: Processes a raster image into a vector polygon ocean/land mask.
		Get a standardized vector name from a Landsat raster file name.
		Input:  str fileName - Landsat raster file name.
				str domainName - Domain vector file name.
		Output: str lineName, str polygonName - vector file names.
	"""
	
	date = fileName.split('_')[2]
	lineName = '_'.join(['cf', domainName, date, 'closed'])
	polygonName = '_'.join([lineName, 'polygon'])
	return lineName, polygonName

def getSavePaths(fileName:str, domainLayer:QgsLayerTreeGroup, typeDir:str):
	"""Description: Processes a raster image into a vector polygon ocean/land mask.
		Get a standardized vector name from a Landsat raster file name.
		Input:  str fileName - Landsat raster file name.
				str glacierName - Root vector file name.
				str typeDir - name of the type subdirectory.
		Output: str path - file save paths.
	"""
	
	date = fileName.split('_')[2]
	year = date.split('-')[0]
	
	# if (rootGroup.parent().name().startswith('2') or rootGroup.parent().name().startswith('1')):
		# rootGroup = rootGroup.parent()
	
	path = resolve('CalvingFronts')
	if (not os.path.exists(path)):
		os.mkdir(path)
	path = os.path.join(path, typeDir)
	if (not os.path.exists(path)):
		os.mkdir(path)
	path = os.path.join(path, domainLayer.name())
	if (not os.path.exists(path)):
		os.mkdir(path)
	path = os.path.join(path, year)
	if (not os.path.exists(path)):
		os.mkdir(path)
	
	return path
	
def layerSubsetLoad(rasterLayer:QgsRasterLayer, domainLayer:QgsVectorLayer, rootGroup:QgsLayerTreeGroup, name:str) -> (QgsRasterLayer, QgsVectorLayer):
	"""Description: Processes a raster image into a vector polygon ocean/land mask.
		Make sure to save the shapefile, as it will be deleted otherwise! 
		Input:  QgsRasterLayer rasterLayer - layer that contains the raster image to process
				QgsVectorLayer domainLayer - layer that contains a polygon specifying the bounds of the raster image to process
				QgsLayerTreeGroup rootGroup - layer that contains the root name of the glacier
				str name - name of file.
		Output: QgsRasterLayer, QgsVectorLayer - objects referencing the new mask layers
	"""
	
	# Get basic file name information on geotiff, raster image, masked raster subset image, and masked vector subset shp file
	#print('Get basic file name information on geotiff, raster image, masked raster subset image, and masked vector subset shp file')
	fileSource = rasterLayer.source()
	fileInfo = QFileInfo(fileSource)
	filePath = fileInfo.absolutePath()
	fileName = fileInfo.baseName()
	fileQASource = filePath + '/' + fileName[:fileName.rfind('_B')] + '_BQA.TIFF'
	maskName = fileName + '_masked'
	maskPath = filePath + '/' + maskName + '.tif'
	lineMaskName, polyMaskName = getVectorNames(fileName, domainLayer.name())
	vectorPath = getSavePaths(fileSource, domainLayer, 'shp')
	lineMaskPath = vectorPath + '/' + lineMaskName + '.shp'
	polyMaskPath = vectorPath + '/' + polyMaskName + '.shp'
	maskIceName = fileName + '_masked_ice'
	maskIcePath = filePath + '/' + maskIceName + '.tif'
	maskCloudName = fileName + '_masked_cloud'
	maskCloudPath = filePath + '/' + maskCloudName + '.tif'
	
	if os.path.exists(lineMaskPath):
		layers = QgsProject.instance().mapLayersByName(lineMaskName)
		if (len(layers) > 0):
			for layer in layers:
				QgsProject.instance().removeMapLayer(layer.id())
	if os.path.exists(polyMaskPath):
		layers = QgsProject.instance().mapLayersByName(polyMaskName)
		if (len(layers) > 0):
			for layer in layers:
				QgsProject.instance().removeMapLayer(layer.id())
	
	# Load geotiff and get domain layer/bounding box of area to mask
	#print('Load geotiff and get domain layer/bounding box of area to mask')
	geotiff = gdal.Open(fileSource)
	feature = domainLayer.getFeature(0)
	domain = feature.geometry().boundingBox()
	rasterCRS = rasterLayer.crs().authid()
	domainCRS = domainLayer.crs().authid()
	bounds = geotiffWorldToPixelCoords(geotiff, domain, rasterCRS, domainCRS)
	
	#print("mask = imread(resolve('landsat_preds/' + name + '_pred.png'))")
	mask = imread(resolve('landsat_preds/' + domainLayer.name() + '/' + name + '_mask.png'))
	
	# Save results to files and layers
	#print('Save results to files and layers')
	arrayToRaster(mask, geotiff, bounds, maskPath)
	rasterLayer = QgsRasterLayer(maskPath, maskName)
	lineLayer, polygonLayer = vectorizeRaster(maskPath, lineMaskPath, lineMaskName, polyMaskPath, polyMaskName)
	
	geotiff = None
	return lineLayer, polygonLayer

#https://www.commonlounge.com/discussion/e4362e6c0f094ed5af381b39e29a9d8a
#Multi-scale evaluation: At test time, Q={S-32,S,S+32} for fixed S and Q={S_min, 0.5(S_min + S_max), S_max} for jittered S. 
#Resulting class posteriors are averaged. This performs the best.
#Dense v/s multi-crop evaluation
#In dense evaluation, the fully connected layers are converted to convolutional layers at test time, 
#and the uncropped image is passed through the fully convolutional net to get dense class scores. 
#Scores are averaged for the uncropped image and its flip to obtain the final fixed-width class posteriors.
#This is compared against taking multiple crops of the test image and averaging scores obtained by passing each of these through the CNN.
#Multi-crop evaluation works slightly better than dense evaluation, 
#but the methods are somewhat complementary as averaging scores from both did better than each of them individually.
# The authors hypothesize that this is probably because of the different boundary conditions: 
#when applying a ConvNet to a crop, the convolved feature maps are padded with zeros, 
#while in the case of dense evaluation the padding for the same crop naturally comes from the neighbouring parts of an image 
#(due to both the convolutions and spatial pooling), which substantially increases the overall network receptive field, so more context is captured.