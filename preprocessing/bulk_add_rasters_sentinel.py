from qgis.core import *
import os, re, glob, time


def sentinel_sort(file_path):
	"""Sorting key function derives date from landsat file path."""
	return file_path.split(os.path.sep)[-1].split('_')[4]

def landsat_sort(file_path):
	"""Sorting key function derives date from landsat file path."""
	return file_path.split(os.path.sep)[-1].split('_')[3]

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
			# If we found something, return it.
			if result is not None:
				return result
	#Found nothing
	return None

def layerFromPath(file_path:str, group:QgsLayerTreeGroup) -> (QgsRasterLayer):
	name = os.path.splitext(os.path.basename(file_path))[0]
	rasterLayer = QgsRasterLayer(file_path, name)
	provider = rasterLayer.dataProvider()
	provider.setNoDataValue(1, 0)
	
	QgsProject.instance().addMapLayer(rasterLayer, False)
	# step 2: append layer to the root group node
	layerGroup = group.insertLayer(0, rasterLayer)
	layerGroup.setItemVisibilityChecked(False)
	layerGroup.setExpanded(False)

def bulkAdd():
	root_group_name = 'Rasters'
	source_path = r'../downloader/rasters/Sentinel-1/Antarctica'
	dry_run = 1

	project = QgsProject.instance()
	root = project.layerTreeRoot()
#	layers = [node.layer() for node in root.findLayers()]
#	groups = root.findGroups()
	root_group = findGroup(root, root_group_name)
	
	domain_groups = []
	# For each domain in CalvingFronts...
	for domain in os.listdir(source_path):
		if domain == 'zipped':
			continue
		domain_groups.append(domain)
		root_group.addGroup(domain)
		domain_group = findGroup(root_group, domain)
		domain_path = os.path.join(source_path, domain)
		# For each year group in CalvingFronts...
		for year in sorted(os.listdir(domain_path)):
			domain_group.addGroup(year)
			year_group = findGroup(domain_group, year)
			year_path = os.path.join(domain_path, year)
			# For each shapefile in the year group...
			for file in sorted(glob.glob(os.path.join(year_path, '*.tiff')), key=sentinel_sort, reverse=True):
				file_path = os.path.join(year_path, file)
				#Add the layers to the project
				print(domain, year, file_path)
				if dry_run != 0:
					layerFromPath(file_path, year_group)
		domain_group.setExpanded(False)
	project.write()

bulkAdd()
