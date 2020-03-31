# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 01:41:16 2020

@author: Daniel
"""
import os
print('GDAL_DATA' in os.environ)
from osgeo import osr
srs = osr.SpatialReference() # Declare a new SpatialReference 
srs.ImportFromEPSG(3413) # Import the EPSG code into the new object srs
print(srs.ExportToWkt()) # Print the result before transformation to ESRI WKT (prints nothing)


import os
os.environ['GDAL_DATA'] = 'D:\\ProgramData\\Anaconda3\\envs\\cfm\\Library\\share\\gdal'
print('GDAL_DATA' in os.environ)
from osgeo import , osr
srs = osr.SpatialReference() # Declare a new SpatialReference 
srs.ImportFromEPSG(3413) # Import the EPSG code into the new object srs
print(srs.ExportToWkt()) # Print the result before transformation to ESRI WKT (prints nothing)
