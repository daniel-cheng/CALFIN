import os, re, glob, sys
import numpy as np
from skimage.io import imsave, imread
import skimage
import numpngw
#skimage.io.use_plugin('freeimage')

dry_run = False
#raw_path = r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\train_raw"
#temp_path = r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\train_temp"
#dest_path = r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\train_processed"

raw_path = r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\validation_raw"
temp_path = r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\validation_temp"
dest_path = r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\validation_processed"

raw_path = r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\temp_raw"
temp_path = r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\temp_temp"
dest_path = r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\temp_processed"

raw_path = r"D:\Daniel\Documents\Github\CALFIN Repo Intercomp\processing\raw"
temp_path = r"D:\Daniel\Documents\Github\CALFIN Repo Intercomp\processing\temp"
dest_path = r"D:\Daniel\Documents\Github\CALFIN Repo Intercomp\processing\processed"

raw_path = r"D:\Daniel\Documents\Github\CALFIN Repo\preprocessing\calvingfrontmachine\landsat_raw"
temp_path = r"D:\Daniel\Documents\Github\CALFIN Repo\processing\landsat_raw_temp"
dest_path = r"D:\Daniel\Documents\Github\CALFIN Repo\processing\landsat_raw_processed"

raw_path = r"D:\Daniel\Documents\Github\CALFIN Repo\processing\landsat_raw"
temp_path = r"D:\Daniel\Documents\Github\CALFIN Repo\processing\landsat_raw_temp"
dest_path = r"D:\Daniel\Documents\Github\CALFIN Repo\processing\landsat_raw_processed"

for file_path in glob.glob(os.path.join(raw_path, '**', '*')):
	base_name = file_path.split(os.path.sep)[-1]
	raw_path = os.path.join(file_path)
	hdr_path = os.path.join(temp_path, base_name[0:-4] + '_hdr.png')
	sh_path = os.path.join(temp_path, base_name[0:-4] + '_sh.png')
	
	raw_img = imread(raw_path, as_gray = True)
	hdr_img = imread(hdr_path, as_gray = True)
	sh_img = imread(sh_path, as_gray = True)
	
	#Convert greyscale to RGB greyscale
	raw_max = raw_img.max()
	hdr_max = hdr_img.max()
	sh_max = sh_img.max()
	print(raw_max, hdr_max, sh_max, raw_img.dtype)
	if (raw_img.dtype != np.uint16):
		raw_img = np.round(raw_img / raw_max * 65535.0).astype(np.uint16) #np.uint16 [0, 65535]
	if (hdr_img.dtype != np.uint16):
		hdr_img = np.round(hdr_img / hdr_max * 65535.0).astype(np.uint16) #np.uint16 [0, 65535]
	if (sh_img.dtype != np.uint16):
		sh_img = np.round(sh_img / sh_max * 65535.0).astype(np.uint16) #np.uint16 [0, 65535]
#	if (mask_max != 0.0):
#		mask_uint8 = np.floor(mask_f64 / mask_max * 255.0).astype(np.uint8) #np.uint8 [0, 255]
#	mask_3_uint8 = np.stack((mask_uint8,)*3, axis=-1)
#	

	rgb_img = np.stack((raw_img, hdr_img, sh_img), axis=-1)
	save_path = os.path.join(dest_path, base_name)
	print('Saving processed raw to:', save_path)
	if (dry_run == False):
		numpngw.write_png(save_path, rgb_img)