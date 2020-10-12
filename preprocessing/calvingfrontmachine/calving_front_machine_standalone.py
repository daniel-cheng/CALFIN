from PyQt5.QtCore import QSettings, QTranslator, qVersion, QCoreApplication, QFile, QFileInfo
from PyQt5.QtGui import QIcon, QColor
from PyQt5.QtWidgets import QAction
from qgis.core import *
from qgis.utils import iface

import os, shutil, glob, subprocess, fnmatch
import numpy as np

from collections import defaultdict
from osgeo import gdal, ogr, osr
from pyproj import Proj, transform
from skimage.io import imsave, imread
from skimage.transform import resize

DRY_RUN = 0
nodata_threshold = 0.25
cloud_threshold = 0.15
#Clouds are 5th bit in 16 bit BQA image
maskClouds = 0b0000000000010000

def domainInRaster(rasterLayer: QgsRasterLayer, domainLayer: QgsVectorLayer) -> bool:
	"""Returns bool if domain is within bounds of geotiff in rasterLayer
	:param rasterLayer: QgsRasterLayer
	:param domainLayer: QgsVectorLayer
	"""
	# Get basic file name information on geotiff, raster image, masked raster subset image, and masked vector subset shp file
	fileSource = rasterLayer.source()
	fileInfo = QFileInfo(fileSource)
	fileName = fileInfo.baseName()
	rowPath = fileName.split('_')[3]
	# Load geotiff and get domain layer/bounding box of area to mask
	geotiff = gdal.Open(fileSource)
	feature = domainLayer.getFeature(0)
	domain = feature.geometry().boundingBox()
	prj = geotiff.GetProjection()
	srs = osr.SpatialReference(wkt=prj)
	if srs.GetAttrValue("PROJCS|AUTHORITY", 1) is not None:
		epsgCode = srs.GetAttrValue("PROJCS|AUTHORITY", 1)
	elif srs.GetAttrValue("AUTHORITY", 1) is not None:
		epsgCode = srs.GetAttrValue("AUTHORITY", 1)
	else:
		epsgCode = str(32621)
	rasterCRS = "EPSG:" + epsgCode
	
	crs = rasterLayer.crs()
	crs.createFromId(int(epsgCode))
	
	domainCRS = domainLayer.crs().authid()
	bounds = geotiffWorldToPixelCoords(geotiff, domain, rasterCRS, domainCRS)
	
	#Gather BQA info
	fileSourceBQA = fileSource[:-7] + '_BQA.TIF'
	#Save BQA subset
	geotiffBQA = gdal.Open(fileSourceBQA)
	
	minX = int(round(bounds.yMinimum()))
	maxX = int(round(bounds.yMaximum()))
	minY = int(round(bounds.xMinimum()))
	maxY = int(round(bounds.xMaximum())) 
	
	if minX < 0 or maxX > geotiff.RasterXSize or maxX > geotiffBQA.RasterXSize or minY < 0 or maxY > geotiff.RasterYSize or maxY > geotiffBQA.RasterYSize:
		return False
	else:
		#Check image is above Nodata percentage threshold
		band = geotiff.GetRasterBand(1)
		noDataValue = 0.0
		img = band.ReadAsArray(minX, minY, maxX - minX, maxY - minY).astype(np.uint16)
		noDataCount = np.sum(img == noDataValue)
		percentNoData = noDataCount / img.size
		
		if percentNoData > nodata_threshold:
			geotiff = None
			geotiffBQA = None
			print('Skipping: Nodata percentage above threshold:', percentNoData, ' > ', nodata_threshold)
			return False
		
		#Check image is above cloud percentage threshold
		bandBQA = geotiffBQA.GetRasterBand(1)
		imgBQA = bandBQA.ReadAsArray(minX, minY, maxX - minX, maxY - minY).astype(np.uint16)
		masked = imgBQA & maskClouds
		cloudCount = np.sum(masked)
		percentCloud = cloudCount / 16.0 / imgBQA.size
		if percentCloud > cloud_threshold:
			geotiff = None
			geotiffBQA = None
			print('Skipping: Cloud percentage above threshold:', percentCloud, ' > ', cloud_threshold)
			return False
	geotiff = None
	geotiffBQA = None
	return True

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
		for filename in glob.glob(outShapefileBase + "*"):
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
	if not DRY_RUN:
		imsave(path, img)

def layerWarp(rasterNode:QgsLayerTreeGroup, domainLayer:QgsVectorLayer) -> None:
	"""Description: Reprojects a raster if not already in the desired CRS.
		Make sure to save the shapefile, as it will be deleted otherwise! 
		Input:  QgsLayerTreeGroup rasterNode - node that contains the raster image to process
				QgsVectorLayer domainLayer - layer that contains a polygon specifying the bounds of the raster image to process
	"""
	
	rasterLayer = rasterNode.layer()
	fileSource = rasterLayer.source()
	geotiff = gdal.Open(fileSource)
	prj = geotiff.GetProjection()
	srs = osr.SpatialReference(wkt=prj)
	domainCRS = domainLayer.crs().authid()
	if srs.GetAttrValue("PROJCS|AUTHORITY", 1) is not None:
		epsgCode = srs.GetAttrValue("PROJCS|AUTHORITY", 1)
	elif srs.GetAttrValue("AUTHORITY", 1) is not None:
		epsgCode = srs.GetAttrValue("AUTHORITY", 1)
	else:
		epsgCode = str(32621)
	rasterCRS = "EPSG:" + epsgCode
	if (rasterCRS != domainCRS):
		print('warping...', rasterCRS, domainCRS)
		if not DRY_RUN:
			parent = rasterNode.parent()
			rasterName = rasterLayer.name()
			
			outSource = fileSource[0:-4] + "_" + domainCRS[5:] + ".tif"
			#processing.algorithmHelp("gdal:warpreproject")
			print(fileSource, outSource)
			processing.run("gdal:warpreproject",
				# {"INPUT":fileSource,
				# "SOURCE_SRS":rasterCRS,
				# "TARGET_CRS":domainCRS,
				# "RESAMPLING":1, #Bilinear
				# "NO_DATA":0,
				# "DATA_TYPE":5, #Float32
				# "MULTITHREADING":True,
				# "OUTPUT":outSource})
				{'DATA_TYPE': 5,#Float32
				'INPUT': rasterName,
				'MULTITHREADING': True,
				'NODATA': 0.0,
				'OPTIONS': '',
				'OUTPUT': outSource,
				'RESAMPLING': 1, #Bilinear
				'SOURCE_CRS': rasterCRS,
				'TARGET_CRS': domainCRS,
				'TARGET_EXTENT': None,
				'TARGET_EXTENT_CRS': None,
				'TARGET_RESOLUTION': None})
			print('hello')
			QgsProject.instance().removeMapLayer(rasterLayer.id())
			geotiff = None
			shutil.copy2(outSource, fileSource)
			os.remove(outSource)
			rasterLayer = QgsRasterLayer(fileSource, rasterName)
			QgsProject.instance().addMapLayer(rasterLayer, False)
			rasterNode = parent.insertLayer(0, rasterLayer)
			return rasterNode
	return rasterNode

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

def getSavePaths(fileName:str, domainLayer:QgsVectorLayer, typeDir:str):
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

def postprocess(image:np.ndarray) -> np.ndarray:
	"""Description: Postprocesses mask layer to remove small features.
	"""
	image = np.where(image > 127, 255, 0)
	# Close edges to join them and dilate them before removing small components
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
	closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
	dilated = cv2.dilate(closing, kernel, iterations = 1)
	largeComponents = removeSmallComponents(dilated, 0, 255)
	# largeComponents = dilated
	
	# Remove small components inside floodfill area
	largeComponentsInverted = 255 - largeComponents
	largeComponentsInverted = removeSmallComponents(largeComponentsInverted, 0, 255)
	largeComponentsInverted2 = 255 - largeComponentsInverted
	
	# Reverse initial morphological operators to retrieve original edge mask
	eroded = cv2.erode(largeComponentsInverted2, kernel, iterations = 1)
	opening = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
	
	result = removeSmallComponents(opening, 0, 255)
	
	return result

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
	tifPath = getSavePaths(fileSource, domainLayer, 'tif')
	lineMaskPath = vectorPath + '/' + lineMaskName + '.shp'
	polyMaskPath = vectorPath + '/' + polyMaskName + '.shp'
	rawTifPath = tifPath + '/' + name + '.tif'
	
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
	try:
		# raw = imread(resolve('landsat_preds/' + domainLayer.name() + '/' + name + '_raw.png'))
		# arrayToRaster(raw, geotiff, bounds, rawTifPath)
		
		mask = imread(resolve('landsat_preds/' + domainLayer.name() + '/' + name + '_mask.png'))
		# mask = postprocess(mask)
		# Save results to files and layers
		#print('Save results to files and layers')
		arrayToRaster(mask, geotiff, bounds, maskPath)
		rasterLayer = QgsRasterLayer(maskPath, maskName)
		lineLayer, polygonLayer = vectorizeRaster(maskPath, lineMaskPath, lineMaskName, polyMaskPath, polyMaskName)
		
		geotiff = None
		return lineLayer, polygonLayer
	except:
		print(resolve('landsat_preds/' + domainLayer.name() + '/' + name + '_mask.png'), 'not found - skipping.')
		return None, None

def layerSubsetSave(rasterLayer:QgsRasterLayer, domainLayer:QgsVectorLayer, subsetName:str) -> None:
	"""Description: Processes a raster image into a vector polygon ocean/land mask.
		Make sure to save the shapefile, as it will be deleted otherwise! 
		Input:  QgsRasterLayer rasterLayer - layer that contains the raster image to process
				QgsVectorLayer domainLayer - layer that contains a polygon specifying the bounds of the raster image to process
				string name - output file name.
		Output: QgsRasterLayer, QgsVectorLayer - objects referencing the new mask layers
	"""
	
	# Get basic file name information on geotiff, raster image, masked raster subset image, and masked vector subset shp file
	fileSource = rasterLayer.source()
	fileInfo = QFileInfo(fileSource)
	fileName = fileInfo.baseName()	
	savePaths = getSavePaths(fileSource, domainLayer, 'tif') 
	subsetPath = savePaths + '/' + subsetName + '.tif'
	
	# Load geotiff and get domain layer/bounding box of area to mask
	geotiff = gdal.Open(fileSource)
	feature = domainLayer.getFeature(0)
	domain = feature.geometry().boundingBox()
	prj = geotiff.GetProjection()
	srs = osr.SpatialReference(wkt=prj)
	if srs.GetAttrValue("PROJCS|AUTHORITY", 1) is not None:
		epsgCode = srs.GetAttrValue("PROJCS|AUTHORITY", 1)
	elif srs.GetAttrValue("AUTHORITY", 1) is not None:
		epsgCode = srs.GetAttrValue("AUTHORITY", 1)
	else:
		epsgCode = str(32621)
	rasterCRS = "EPSG:" + epsgCode
	
	crs = rasterLayer.crs()
	crs.createFromId(int(epsgCode))
	rasterLayer.setCrs(crs)
	rasterLayer.triggerRepaint()
	
	#rasterCRS = rasterLayer.crs().authid()
	domainCRS = domainLayer.crs().authid()
	bounds = geotiffWorldToPixelCoords(geotiff, domain, rasterCRS, domainCRS)
	
	img = geotiff.GetRasterBand(1)
	img = img.ReadAsArray(0,0,geotiff.RasterXSize,geotiff.RasterYSize)
	img = img[int(round(bounds.yMinimum())):int(round(bounds.yMaximum())), int(round(bounds.xMinimum())):int(round(bounds.xMaximum()))]
	img = (img.astype(np.float32) / img.max() * 65535).astype(np.uint16)
	
	# print('Save subset:', subsetPath, resolve('landsat_raw/' + domainLayer.name() + '/' + subsetName + '.png'))
	if not DRY_RUN:
		arrayToRaster(img, geotiff, bounds, subsetPath)
		imsave(resolve('landsat_raw/' + domainLayer.name() + '/' + subsetName + '.png'), img)
		# imsave(resolve('small/' + domainLayer.name() + '/' + subsetName + '.png'), img)
		# imsave(os.path.join(r'../reprocessing\images_1024', domainLayer.name(), subsetName + '.png'), img)
	try:
		#Gather BQA info
		fileSourceBQA = fileSource[:-7] + '_BQA.TIF'
		fileInfoBQA = QFileInfo(fileSourceBQA)
		fileNameBQA = fileInfoBQA.baseName()
		subsetNameBQA = fileNameBQA + '_' + domainLayer.name()
		subsetPathBQA = savePaths + '/' + subsetNameBQA + '.tif'
		#Save BQA subset
		geotiffBQA = gdal.Open(fileSourceBQA)
		imgBQA = geotiffBQA.GetRasterBand(1)
		imgBQA = imgBQA.ReadAsArray(0,0,geotiffBQA.RasterXSize,geotiffBQA.RasterYSize).astype(np.uint16)
		imgBQA = imgBQA[int(round(bounds.yMinimum())):int(round(bounds.yMaximum())), int(round(bounds.xMinimum())):int(round(bounds.xMaximum()))]
		# print('Save BQA subset:', subsetPathBQA, resolve('landsat_raw/' + domainLayer.name() + '/' + subsetName + '_bqa.png'))
		if not DRY_RUN:
			# arrayToRaster(imgBQA, geotiffBQA, bounds, subsetPathBQA)
			# print(fileSourceBQA, geotiffBQA.RasterXSize, geotiffBQA.RasterYSize)
			# print(int(round(bounds.yMinimum())), int(round(bounds.yMaximum())), int(round(bounds.xMinimum())), int(round(bounds.xMaximum())))
			imsave(resolve('landsat_raw/' + domainLayer.name() + '/' + subsetName + '_bqa.png'), imgBQA)
		
		#Gather MTL info
		fileSourceMTL = fileSource[:-7] + '_MTL.txt'
		fileInfoMTL = QFileInfo(fileSourceMTL)
		fileNameMTL = fileInfoMTL.baseName()
		subsetNameMTL = fileNameMTL + '_' + domainLayer.name()
		subsetPathMTL = savePaths + '/' + subsetNameMTL + '.txt'
		#Save MTL subset	
		if not DRY_RUN:
			image_feats = ['']*6
			with open(fileSourceMTL, 'r') as image_feats_source_file:	
				lines = image_feats_source_file.readlines()
				for line in lines:
					if 'SUN_AZIMUTH =' in line:
						image_feats[0] = line.strip()
					elif 'SUN_ELEVATION =' in line:
						image_feats[1] = line.strip()
					elif 'CLOUD_COVER ' in line:
						image_feats[2] = line.strip()
					elif 'CLOUD_COVER_LAND ' in line:
						image_feats[3] = line.strip()
					elif 'DATE_ACQUIRED =' in line:
						image_feats[4] = line.strip()
					elif 'GRID_CELL_SIZE_REFLECTIVE =' in line:
						image_feats[5] = line.strip()
			savePath = resolve('landsat_raw/' + domainLayer.name() + '/' + subsetName + '_mtl.txt')
			with open(savePath, 'w') as image_feats_dest_file:
				for line in image_feats:
					image_feats_dest_file.write(str(line) + '\n')
	except:
		print('No BQA/MTL found for:', subsetName)
	
	return img.shape

def perform_subsetting(rasterLayers, rasterPrefix, domainLayers):
	print('Performing subsetting...')
	
	#Make directories if not already existing
	for domainLayer in domainLayers:
		domainLayer = domainLayer.layer()
		raw_path = resolve('landsat_raw/' + domainLayer.name())
		mask_path = resolve('landsat_preds/' + domainLayer.name())
		if not os.path.exists(raw_path):
			os.mkdir(raw_path)
		if not os.path.exists(mask_path):
			os.mkdir(mask_path)
	
		# Clear data from any previous runs
		# files = glob.glob(raw_path + '/*')
		# for f in files:
			# if os.path.isfile(f):
				# os.remove(f)
		# files = glob.glob(mask_path + '/*')
		# for f in files:
			# if os.path.isfile(f):
				# os.remove(f)
	
	resolutions = warpAndSaveSubsets(rasterLayers, rasterPrefix, domainLayers)
	resizeandSaveSubsets(rasterLayers, rasterPrefix, domainLayers, resolutions)

def warpAndSaveSubsets(rasterLayers, rasterPrefix, domainLayers) -> list:
	# Perform subsetting for each layer, extracting each subset for every domain in domainGroup
	resolutions = defaultdict(list)
	rasterLen = len(rasterLayers)
	domainLen = len(domainLayers)
	total = rasterLen * domainLen
	for i in range(rasterLen):
		rasterLayerNode = rasterLayers[i]
		rasterLayer = rasterLayerNode.layer()
		print('Raster (number):', rasterLayer.source())#, ' Progress:', str(i) + '/' + str(total), '(' + str((i * rasterLen) / total) + '%)')
		for j in range(domainLen):
			domainLayer = domainLayers[j]
			domainLayer = domainLayer.layer()
			rasterLayers[i] = layerWarp(rasterLayerNode, domainLayer)
			rasterLayerNode = rasterLayers[i]
			rasterLayer = rasterLayerNode.layer() #Reload layer in case of warping
			subset_name = domainLayer.name() + "_" + rasterLayer.name()
			print('Domain: ', domainLayer.name())#, ' Progress:', str((i * rasterLen + j) / total) + '%')
			resolution = layerSubsetSave(rasterLayer, domainLayer, subset_name)
			resolutions[domainLayer.name()].append(resolution)
	
	# Calculate the median resolutions for each domain	
	median_resolutions = []
	for i in range(len(domainLayers)):
		domainLayer = domainLayers[i]
		resolution = np.median(resolutions[domainLayer.name()], axis=0)
		print('domainLayer (#):', i, domainLayer.name(), ' resolution:', resolution)
		median_resolutions.append(resolution)
	
	return median_resolutions

def resizeandSaveSubsets(rasterLayers, rasterPrefix, domainLayers, resolutions) -> list:
	# Resize the images to the median size to account for reprojection differences
	rasterLen = len(rasterLayers)
	domainLen = len(domainLayers)
	total = rasterLen * domainLen
	for i in range(rasterLen):
		rasterLayerNode = rasterLayers[i]
		rasterLayer = rasterLayerNode.layer()
		if rasterLayer.name()[-2:] in rasterPrefix:
			for j in range(domainLen):
				domainLayer = domainLayers[j]
				resolution = resolutions[j]
				domainLayer = domainLayer.layer()
				print('Resizing subsets to', resolution)#, ' Progress:', str((i * rasterLen + j) / total) + '%')
				if domainInRaster(rasterLayer, domainLayer):
					subset_name = domainLayer.name() + "_" + rasterLayer.name()
					layerResize(rasterLayer, domainLayer, subset_name, resolution)

def perform_saving(rasterLayers, rasterPrefix, domainLayers):
	#Save for training
	for domainNode in domainLayers:
		domainLayer = domainNode.layer()
		source_path_base = resolve('landsat_raw/' + domainLayer.name())
		dest_path_base = r'D:/Daniel/Documents/GitHub/ultrasound-nerve-segmentation/landsat_raw/train_full/' + domainLayer.name()
		if not os.path.exists(dest_path_base):
			os.mkdir(dest_path_base)
		for rasterLayer in rasterLayers:
			rasterLayer = rasterLayer.layer()
			if rasterLayer.name()[-2:] in rasterPrefix:
				name = domainLayer.name() + "_" + rasterLayer.name() + '.png'
				source_path = os.path.join(source_path_base, name)
				dest_path = os.path.join(dest_path_base, name)
				if not DRY_RUN:
					shutil.copy2(source_path, dest_path)

def processLayers(domainLayers, check_masking, check_saving, check_postprocessing):
	print('Performing masking...')
	for domainNode in domainLayers:
		launchcommand = r'C:\Users\Daniel\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\calvingfrontmachine\cfm.bat'
		check_masking = str(int(check_masking))
		check_postprocessing = str(int(check_postprocessing))
		check_saving = str(int(check_saving))
		arguments = [launchcommand, domainNode.name(), check_masking, check_saving, check_postprocessing]
		print(arguments)
		p = subprocess.Popen(arguments, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_CONSOLE)
		while True:
			line = p.stdout.readline()
			print(str(line))
			if p.poll() != None:
				print('exit code: ', p.poll())
				break
		p.kill()

def perform_vectorization(rasterLayers, domainLayers, rasterPrefix, check_vectorization, check_adding):
	for rasterLayer in rasterLayers:
		for domainLayer in domainLayers:
			try:
				if rasterLayer.name()[-2:] in rasterPrefix:
					if check_vectorization:
						lineLayer, polygonLayer = layerSubsetLoad(rasterLayer.layer(), domainLayer.layer(), rasterGroup, domainLayer.name() + "_" + rasterLayer.name())
					if check_adding:
						# step 1: add the layer to the registry, False indicates not to add to the layer tree
						QgsProject.instance().addMapLayer(lineLayer, False)
						QgsProject.instance().addMapLayer(polygonLayer, False)
						# step 2: append layer to the root group node
						rasterLayer.parent().insertLayer(0, lineLayer)
						rasterLayer.parent().insertLayer(0, polygonLayer)
						# step 3: Add transparency slider to polygon layers
						#polygonLayer.setCustomProperty("embeddedWidgets/count", 1)
						#polygonLayer.setCustomProperty("embeddedWidgets/0/id", "transparency")
						# Alter fill style for vector layers
						polygonSymbol = polygonLayer.renderer().symbol()
						lineSymbol = lineLayer.renderer().symbol()
						polygonSymbol.setColor(lineSymbol.color())
						polygonSymbol.setOpacity(0.25)
						# Redraw canvas and save variable to global context
						iface.layerTreeView().refreshLayerSymbology(lineLayer.id())
						iface.layerTreeView().refreshLayerSymbology(polygonLayer.id())

			except Exception as e:
				print(e)

def resolve(name, basepath='C:/Users/Daniel/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/calvingfrontmachine'):
	if not os.path.exists(basepath):
		basepath = os.path.dirname(os.path.realpath(__file__))
	return os.path.join(basepath, name)

def findGroups(root:QgsLayerTree):
	"""Return a string list of groups."""
	result = []
	for child in root.children():
		if isinstance(child, QgsLayerTreeGroup):
			result.append(child.name())
			result.extend(findGroups(child))
	return result	

def findChildren(root:QgsLayerTree, matchString:str):
	"""Return a string list of groups."""
	result = []
	matchStringParts = matchString.split('/', 1)
	for child in root.children():
		if fnmatch.fnmatch(child.name(), matchStringParts[0]):
			if isinstance(child, QgsLayerTreeGroup):
				if child.name().startswith(('1', '2')): 
					
					if int(child.name()) < 1985:
						result.extend(findChildren(child, matchStringParts[1]))
				else:
					print(child.name())
					result.extend(findChildren(child, matchStringParts[1]))
			else:
				result.append(child)
	return result


project = QgsProject.instance()
root = project.layerTreeRoot()
rasterPrefix = ['B5', 'B4', 'B7']
groups = findGroups(root)

# Get layer objects based on selection string values
rasterGroupName = 'CalvingFronts/Rasters/*/*/*'
rasterGroup = root.findGroup(rasterGroupName)
rasterLayers = findChildren(root, rasterGroupName)
domainGroupName = 'CalvingFronts/Domains/*'
domainGroup = root.findGroup(domainGroupName)
domainLayers = findChildren(root, domainGroupName)

# Get layer objects based on selection string values
# rasterGroupName = 'CalvingFronts/Rasters/Upernavik/*/*'
# rasterGroup = root.findGroup(rasterGroupName)
# rasterLayers = findChildren(root, rasterGroupName)
# domainGroupName = 'CalvingFronts/Domains/Upernavik*'
# domainGroup = root.findGroup(domainGroupName)
# domainLayers = findChildren(root, domainGroupName)

rasterLayers
check_subsetting = True
check_saving = False
check_masking = False
check_postprocessing = False
check_vectorization = False
check_adding = False

#Save subsets of raster source files using clipping domain
if check_subsetting:
	perform_subsetting(rasterLayers, rasterPrefix, domainLayers)
