from PyQt5.QtCore import  QFileInfo
from qgis.core import QgsVectorLayer, QgsRasterLayer, QgsRectangle, QgsLayerTreeGroup, QgsProject, QgsLayerTree

import os, shutil, fnmatch
import numpy as np

from collections import defaultdict
from osgeo import gdal, osr
from pyproj import Proj, transform
from skimage.io import imsave, imread
from skimage.transform import resize
import traceback

DRY_RUN = 0
nodata_threshold = 0.25
cloud_threshold = 0.15 #.25
#Clouds are 5th bit in 16 bit BQA image
maskClouds = 0b0000000000001000 #https://www.usgs.gov/land-resources/nli/landsat/landsat-collection-1-level-1-quality-assessment-band?qt-science_support_page_related_con=0#qt-science_support_page_related_con


def domainInRaster(rasterLayer: QgsRasterLayer, domainLayer: QgsVectorLayer) -> bool:
	"""Returns bool if domain is within bounds of geotiff in rasterLayer
	:param rasterLayer: QgsRasterLayer
	:param domainLayer: QgsVectorLayer
	"""
	# Get basic file name information on geotiff, raster image, masked raster subset image, and masked vector subset shp file
	fileSource = rasterLayer.source()
	
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
	
	minX = int(round(bounds.yMinimum()))
	maxX = int(round(bounds.yMaximum()))
	minY = int(round(bounds.xMinimum()))
	maxY = int(round(bounds.xMaximum())) 
	
	if minX < 0 or maxX > geotiff.RasterXSize or minY < 0 or maxY > geotiff.RasterYSize:
		return False
	else:
		#Check image is above Nodata percentage threshold
		band = geotiff.GetRasterBand(1)
		# Get raster statistics
		stats = band.GetStatistics(True, True)
		min_value, max_value = stats[0], stats[1]
		
		img_full = band.ReadAsArray(0,0,geotiff.RasterXSize,geotiff.RasterYSize)
		img = img_full[int(round(bounds.yMinimum())):int(round(bounds.yMaximum())), int(round(bounds.xMinimum())):int(round(bounds.xMaximum()))]
		if img.shape[0] == 0 or img.shape[1] == 0:
			geotiff = None
			print('Skipping: not in domain')
			return False
		noDataValue = 0.0
#		print(min_value, max_value, img.shape)
		noDataValueThreshold = noDataValue + (max_value - min_value) * 0.006
		noDataCount = np.sum(img < noDataValue + noDataValueThreshold)
		percentNoData = noDataCount / img.size
#		print('percentNoData', percentNoData, 'noDataCount', noDataCount, 'noDataValueThreshold', noDataValueThreshold)
		if percentNoData > nodata_threshold:
			geotiff = None
			print('Skipping: Nodata percentage above threshold:', percentNoData, ' > ', nodata_threshold)
			return False
	geotiff = None
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
	
	rectRefWidth = rectRef.width()
	rectRefHeight = rectRef.height()
	
	domainX = [rectDomain.xMinimum(), rectDomain.xMaximum()]
	domainY = [rectDomain.yMinimum(), rectDomain.yMaximum()]
	inProj = Proj(init=domainCRS)
	outProj = Proj(init=rasterCRS)
	#print(inProj, outProj, domainCRS, rasterCRS)
#	print(domainX, domainY)
	rasterCRSDomainX, rasterCRSDomainY = transform(inProj, outProj, domainX, domainY)
#	print(rasterCRSDomainX, rasterCRSDomainY)
	
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
#	print(xMin, yMin, xMax, yMax)
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


def layerResize(rasterLayer:QgsRasterLayer, domainLayer:QgsVectorLayer, name:str, resolution:(int, int)) -> None:
	"""Description: Processes a raster image into a vector polygon ocean/land mask.
		Make sure to save the shapefile, as it will be deleted otherwise! 
		Input:  QgsRasterLayer rasterLayer - layer that contains the raster image to process
				QgsVectorLayer domainLayer - layer that contains a polygon specifying the bounds of the raster image to process
				QgsVectorLayer outputLayer - layer to save vector layer in. Warning: not supported yet. 
		Output: QgsRasterLayer, QgsVectorLayer - objects referencing the new mask layers
	"""
	
	path = resolve('sentinel_raw/' + domainLayer.name() + '/' + name + '.png')
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
		epsgCode = str(3412)
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
			QgsProject.instance().removeMapLayer(rasterLayer.id())
			geotiff = None
			shutil.copy2(outSource, fileSource)
			os.remove(outSource)
			rasterLayer = QgsRasterLayer(fileSource, rasterName)
			QgsProject.instance().addMapLayer(rasterLayer, False)
			rasterNode = parent.insertLayer(0, rasterLayer)
			return rasterNode
	return rasterNode


def getSavePaths(fileName:str, domainLayer:QgsVectorLayer, typeDir:str):
	"""Description: Processes a raster image into a vector polygon ocean/land mask.
		Get a standardized vector name from a Landsat raster file name.
		Input:  str fileName - Landsat raster file name.
				str glacierName - Root vector file name.
				str typeDir - name of the type subdirectory.
		Output: str path - file save paths.
	"""
	
	date = fileName.split('_')[5]
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
		epsgCode = str(3412)
	rasterCRS = "EPSG:" + epsgCode
	
	crs = rasterLayer.crs()
	crs.createFromId(int(epsgCode))
	rasterLayer.setCrs(crs)
	rasterLayer.triggerRepaint()
	
	#rasterCRS = rasterLayer.crs().authid()
	domainCRS = domainLayer.crs().authid()
	bounds = geotiffWorldToPixelCoords(geotiff, domain, rasterCRS, domainCRS)
	
	band = geotiff.GetRasterBand(1)
	img_full = band.ReadAsArray(0,0,geotiff.RasterXSize,geotiff.RasterYSize)
	img = img_full[int(round(bounds.yMinimum())):int(round(bounds.yMaximum())), int(round(bounds.xMinimum())):int(round(bounds.xMaximum()))]
	print('bounds', bounds.yMinimum(), bounds.yMaximum(), bounds.xMinimum(), bounds.xMaximum())
	print('img.shape', img.shape, 'img_full.shape', img_full.shape)
	print("img min/max/mean:", img.min(), img.max(), np.mean(img, axis=(0, 1)))
	img = (img.astype(np.float32) / img.max() * 65535).astype(np.uint16)
	print("after img min/max/mean:", img.min(), img.max(), np.mean(img, axis=(0, 1)))
	
	# print('Save subset:', subsetPath, resolve('sentinel_raw/' + domainLayer.name() + '/' + subsetName + '.png'))
	if not DRY_RUN:
		arrayToRaster(img, geotiff, bounds, subsetPath)
		imsave(resolve('sentinel_raw/' + domainLayer.name() + '/' + subsetName + '.png'), img)
		# imsave(resolve('small/' + domainLayer.name() + '/' + subsetName + '.png'), img)
		# imsave(os.path.join(r'D:\Daniel\Documents\Github\CALFIN Repo\reprocessing\images_1024', domainLayer.name(), subsetName + '.png'), img)
	
	return img.shape


def resolve(name, basepath=r'D:\Daniel\Documents\Github\CALFIN Repo\preprocessing\calvingfrontmachine'):
	if not os.path.exists(basepath):
		basepath = os.path.dirname(os.path.realpath(__file__))
	return os.path.join(basepath, name)


def findChildren(root:QgsLayerTree, matchString:str):
	"""Return a list of groups in the root that match a regex string argument."""
	result = []
	matchStringParts = matchString.split('/', 1)
	for child in root.children():
		if fnmatch.fnmatch(child.name(), matchStringParts[0]):
			if isinstance(child, QgsLayerTreeGroup):
				result.extend(findChildren(child, matchStringParts[1]))
			else:
				result.append(child)
	return result

class TestTask( QgsTask ):

	def __init__(self, desc):
		QgsTask.__init__(self, desc )
	
	def perform_subsetting(self, rasterLayers, domainLayers):
		print('Performing subsetting...')
		
		#Make directories if not already existing
		base_path = resolve('sentinel_raw/')
		if not os.path.exists(base_path):
			os.mkdir(base_path)
		for domainLayer in domainLayers:
			domainLayer = domainLayer.layer()
			raw_path = resolve('sentinel_raw/' + domainLayer.name())
			if not os.path.exists(raw_path):
				os.mkdir(raw_path)
		
		resolutions = self.warpAndSaveSubsets(rasterLayers, domainLayers)
		self.resizeandSaveSubsets(rasterLayers, domainLayers, resolutions)
	
	def resizeandSaveSubsets(self, rasterLayers, domainLayers, resolutions) -> list:
		# Resize the images to the median size to account for reprojection differences
		rasterLen = len(rasterLayers)
		domainLen = len(domainLayers)
		for i in range(rasterLen):
			self.setProgress((i + 1) / rasterLen * 100)
			rasterLayerNode = rasterLayers[i]
			rasterLayer = rasterLayerNode.layer()
			for j in range(domainLen):
				domainLayer = domainLayers[j]
				resolution = resolutions[j]
				domainLayer = domainLayer.layer()
				print('Resizing subsets to', resolution)#, ' Progress:', str((i * rasterLen + j) / total) + '%')
				if domainInRaster(rasterLayer, domainLayer):
					subset_name = domainLayer.name() + "_" + rasterLayer.name()
					layerResize(rasterLayer, domainLayer, subset_name, resolution)
	
	def warpAndSaveSubsets(self, rasterLayers, domainLayers) -> list:
		# Perform subsetting for each layer, extracting each subset for every domain in domainGroup
		resolutions = defaultdict(list)
		rasterLen = len(rasterLayers)
		domainLen = len(domainLayers)
		for i in range(rasterLen):
			self.setProgress((i + 1) / rasterLen * 100)
			rasterLayerNode = rasterLayers[i]
			rasterLayer = rasterLayerNode.layer()
			print('Raster (' + str(i) + '/' + str(rasterLen) + '):', rasterLayer.source())#, ' Progress:', str(i) + '/' + str(total), '(' + str((i * rasterLen) / total) + '%)')
			for j in range(domainLen):
				domainLayer = domainLayers[j]
				domainLayer = domainLayer.layer()
				rasterLayers[i] = layerWarp(rasterLayerNode, domainLayer)
				rasterLayerNode = rasterLayers[i]
				rasterLayer = rasterLayerNode.layer() #Reload layer in case of warping
				subset_name = domainLayer.name() + "_" + rasterLayer.name()
				print('Domain: ', domainLayer.name())#, ' Progress:', str((i * rasterLen + j) / total) + '%')
				if domainInRaster(rasterLayer, domainLayer):
					resolution = layerSubsetSave(rasterLayer, domainLayer, subset_name)
					resolutions[domainLayer.name()].append(resolution)
		
		# Calculate the median resolutions for each domain	
		median_resolutions = []
		for i in range(len(domainLayers)):
			domainLayer = domainLayers[i]
			resolution = np.ceil(np.median(resolutions[domainLayer.name()], axis=0))
			print('domainLayer (#):', i, domainLayer.name(), ' resolution:', resolution)
			median_resolutions.append(resolution)
		
		return median_resolutions
	
	def run(self):
		project = QgsProject.instance()
		root = project.layerTreeRoot()
		
		# Get layer objects based on selection string values
		rasterGroupName = 'CalvingFronts/Rasters/*/*/*'
		rasterLayers = findChildren(root, rasterGroupName)
		domainGroupName = 'CalvingFronts/Domains/*/*'
		domainLayers = findChildren(root, domainGroupName)
		
		# Get layer objects based on selection string values
		#rasterGroupName = 'CalvingFronts/Rasters/Helheim/1991/*'
		#rasterLayers = findChildren(root, rasterGroupName, 1992)
		#domainGroupName = 'CalvingFronts/Domains/Br*'
		#domainLayers = findChildren(root, domainGroupName, 1992)
		
		#Save subsets of raster source files using clipping domain
		try:
			self.perform_subsetting(rasterLayers, domainLayers)
		except Exception as e:
			traceback.print_exc()
		
		self.completed()

task = TestTask('Warp and Subsetting...') 
QgsApplication.taskManager().addTask(task)
	

