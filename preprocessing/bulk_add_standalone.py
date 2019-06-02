from qgis.core import *
import os, re, glob

root_group_name = 'CalvingFronts'
source_path = './Greenland/CalvingFronts'
dry_run = 0

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
				
def layersFromPath(line_file_path:str, poly_file_path:str, group:QgsLayerTreeGroup) -> (QgsVectorLayer, QgsVectorLayer):
	lineLayer = QgsVectorLayer(line_file_path, line_file_path[0:-4], 'ogr')
	polygonLayer = QgsVectorLayer(poly_file_path, poly_file_path[0:-4], 'ogr')

	QgsProject.instance().addMapLayer(lineLayer, False)
	QgsProject.instance().addMapLayer(polygonLayer, False)
	# step 2: append layer to the root group node
	group.insertLayer(0, lineLayer)
	group.insertLayer(0, polygonLayer)
	# step 3: Add transparency slider to polygon layers
	# Alter fill style for vector layers
	polygonSymbol = polygonLayer.renderer().symbol()
	lineSymbol = lineLayer.renderer().symbol()
	polygonSymbol.setColor(lineSymbol.color())
	polygonSymbol.setOpacity(0.25)
	# Redraw canvas and save variable to global context
	self.iface.layerTreeView().refreshLayerSymbology(lineLayer.id())
	self.iface.layerTreeView().refreshLayerSymbology(polygonLayer.id())

def bulkAdd()
	project = QgsProject.instance()
	root = project.layerTreeRoot()
	layers = [node.layer() for node in root.findLayers()]
	groups = self.findGroups(root)
	root_group = findGroup(root, root_group_name)

	domain_groups = []
	# For each domain in CalvingFronts...
	for domain in os.listdir(source_path):
		domain_groups.append(domain)
		root_group.addGroup(domain)
		domain_group = findGroup(root_group, domain)
		domain_path = os.path.join(source_path, domain)
		year_groups = []
		# For each year group in CalvingFronts...
		for year in os.listdir(domain_path)
			year_groups.append(domain)
			domain_group.addGroup(year)
			year_group = findGroup(domain_group, year)
			year_path = os.path.join(domain_path, year, 'shp')
			# For each shapefile in the year group...
			for file in glob.glob(os.path.join(year_path, '*_closed.shp'):
				line_file_path = os.path.join(year_path, file)
				poly_file_path = line_file_path[0:-4] + '_polygon.shp'
				#Add the layers to the project
				print(domain, year, line_file_path, poly_file_path)
				if dry_run != 0:
					layersFromPath(line_file_path, poly_file_path, year_group)
	

if __name__ == "__main__":
	bulkAdd()