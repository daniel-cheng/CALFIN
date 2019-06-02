from qgis.core import *
import os, re, glob, sys
import numpy as np
from skimage.io import imsave, imread

in_path = r"C:\Users\Daniel\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\calvingfrontmachine\train"
try:
    domain = sys.argv[1] #'Upernavik'
    scale = float(sys.argv[2])
    dry_run = int(sys.argv[3])
except:
    domain = 'Hayes-train'
    dry_run = 0
source_path = os.path.join(in_path, domain)

# Generate mask confidence from masks
averages = []    
for name in os.listdir(source_path):        
    file_path = os.path.join(source_path, name)
    if '_mask.png' not in name or not os.path.isfile(file_path):
        continue
    result = imread(file_path, as_gray = True)
    averages.append(result)
    resolutions.append(result.shape)
resolution = np.median(resolutions, axis=0)

averages = np.array(averages)

#Calculate confidence image
confidence = calculate_confidence(averages)

#Duplicate mask confidence for each input image
for name in os.listdir(source_path):
    if '_mask.png' not in name or not os.path.isfile(file_path):
        continue
    save_path = os.path.join(source_path, name[0:-4] + '_confidence.png')
    print('Saving mask confidence to:', save_path)
    if (dry_run == 0):
        imsave(save_path, confidence)
