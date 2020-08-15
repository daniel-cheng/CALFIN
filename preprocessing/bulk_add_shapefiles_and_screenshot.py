# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:15:18 2020
@author: Daniel
"""
from qgis.core import QgsLayerTreeGroup, QgsVectorLayer, QgsProject, QgsTask, QgsApplication, QgsRendererRange, QgsStyle, QgsGraduatedSymbolRenderer
import os, glob, traceback, fnmatch, time, qgis
from osgeo import ogr
from qgis.utils import iface
#add vectors

def findGroup(root:QgsLayerTreeGroup, name:str) -> QgsLayerTreeGroup:
	"""Recursively finds first group that matches name."""
	#Search immediate children
	for child in root.children():
		if isinstance(child, QgsLayerTreeGroup):
			if name == child.name():
				return child
	#Search subchildren
	for child in root.children():
		if isinstance(child, QgsLayerTreeGroup):
			result = findGroup(child, name)
			if result != None:
				return result
	#Found nothing
	return None

def findChildren(root:QgsLayerTreeGroup, matchString:str):
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

def layerFromPath(lineFilePath:str, rootGroup:QgsLayerTreeGroup,  project:QgsLayerTreeGroup) -> None:
	lineFileBasename = os.path.splitext(os.path.basename(lineFilePath))[0]
	lineLayer = QgsVectorLayer(lineFilePath, lineFileBasename, 'ogr')
	
	# Get number of features (range of Sequence#, number of renderer color classes)
	driver = ogr.GetDriverByName('ESRI Shapefile')
	dataSource = driver.Open(lineFilePath, 0) # 0 means read-only. 1 means writeable.
	layer = dataSource.GetLayer()
	dataSource = None
	
	#Setup graduated color renderer based on year
	targetField = 'Year'
	renderer = QgsGraduatedSymbolRenderer('', [QgsRendererRange()])
	renderer.setClassAttribute(targetField)
	lineLayer.setRenderer(renderer)
	
	#Get viridis color ramp
	style = QgsStyle().defaultStyle()
	defaultColorRampNames = style.colorRampNames()
	viridisIndex = defaultColorRampNames.index('Viridis')
	viridisColorRamp = style.colorRamp(defaultColorRampNames[viridisIndex]) #Spectral color ramp
	
	#Dynamically recalculate number of classes and colors
	renderer.updateColorRamp(viridisColorRamp)
	yearsRange = list(range(1972, 2020))
	classCount = len(yearsRange)
	renderer.updateClasses(lineLayer, QgsGraduatedSymbolRenderer.EqualInterval, classCount)
	
	#Set graduated color renderer based on Sequence#
	for i in range(classCount): #[1972-2019], 2020 not included
		targetField = 'DateUnix'
		year = yearsRange[i]
		renderer.updateRangeLowerValue(i, year)
		renderer.updateRangeUpperValue(i, year)
		
	project.addMapLayer(lineLayer, False)
	layer = rootGroup.insertLayer(0, lineLayer)

class TestTask( QgsTask ):
	def __init__(self, desc):
		QgsTask.__init__(self, desc )
		
	def bulkAdd(self):
		rootGroupName = 'Shapefiles'
		sourcePath = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\upload_production\v1.0\level-1_shapefiles-domain-termini'
		project = QgsProject.instance()
		root = project.layerTreeRoot()
		rootGroup = findGroup(root, rootGroupName)
		
		# For each domain in CalvingFronts...
		shapefilesPathList = glob.glob(os.path.join(sourcePath, '*.shp'))
		numShapefiles = len(shapefilesPathList)
		for i in reversed(range(numShapefiles)):
			self.setProgress((i) / numShapefiles * 50)
			lineFilePath = shapefilesPathList[i]
			layerFromPath(lineFilePath, rootGroup, project)
			self.setProgress((i + 1) / numShapefiles * 50)
			
	def bulkScreenshot(self):
		project = QgsProject.instance()
		root = project.layerTreeRoot()
		domainGroupName = 'CalvingFronts/Domains/*/*'
		domainLayers = findChildren(root, domainGroupName)
		
		# For each domain in CalvingFronts...
#		canvas = iface.mapCanvas()
		numDomain = len(domainLayers)
		
		view = iface.layerTreeView()
		saveDir = r'D:\Daniel\Documents\Github\CALFIN Repo\paper\qgis_screnshots'
		for i in range(numDomain):
			self.setProgress((i) / numDomain * 50)
			domainLayer = domainLayers[i].layer()
			layerName = domainLayer.name()
			view.setCurrentLayer(domainLayer)
#			extent = domainLayer.extent()
#			print(extent)
#			canvas.setExtent(extent)
			iface.actionZoomToLayer()
#			iface.mapCanvas().refresh()
			time.sleep(3)
			savePath = os.path.join(saveDir, 'termini_1972-2019_' + layerName + '_overlay.png')
			iface.mapCanvas().saveAsImage(savePath)
			self.setProgress((i + 1) / numDomain * 50)
			
	def run(self):
		try:
			self.bulkAdd()
#			self.bulkScreenshot()
		except:
			traceback.print_exc()
		self.completed()

task = TestTask('Adding Shapefiles...') 
QgsApplication.taskManager().addTask(task)
def manual_screenshot():
	domainLayer = domainLayers.pop()
	layerName = domainLayer.name()
	savePath = os.path.join(saveDir, 'termini_1972-2019_' + layerName + '_overlay.png')
	qgis.utils.iface.mapCanvas().saveAsImage(savePath)
	if len(domainLayers) > 0:
		domainLayer = domainLayers[-1].layer()
		view.setCurrentLayer(domainLayer)
		qgis.utils.iface.zoomToActiveLayer()
#		timer = threading.Timer(1.0, lambda: save_screenshot(domainLayers))
#		timer.start()

project = QgsProject.instance()
root = project.layerTreeRoot()
domainGroupName = 'CalvingFronts/Domains/*'
domainLayers = findChildren(root, domainGroupName)
# For each domain in CalvingFronts...
numDomain = len(domainLayers)
view = qgis.utils.iface.layerTreeView()
saveDir = r'D:\Daniel\Documents\Github\CALFIN Repo\paper\qgis_screenshots'
domainLayer = domainLayers[-1].layer()
view.setCurrentLayer(domainLayer)
qgis.utils.iface.zoomToActiveLayer()
#for i in range(numDomain):
#	manual_screenshot()
manual_screenshot()

