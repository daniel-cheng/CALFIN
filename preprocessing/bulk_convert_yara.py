import os, re, glob, sys, cv2
import numpy as np
from skimage.io import imsave, imread
import skimage
import numpngw
from skimage.transform import resize
#skimage.io.use_plugin('freeimage')

def convert(raw_path, temp_path, dest_path, domains, dry_run = True):
	total = 0
	for image_path in glob.glob(os.path.join(raw_path, "*B[0-9].png")):
		image_name = image_path.split(os.path.sep)[-1]
		image_name_base = image_name.split('.')[0]
		
		domain = image_name_base.split('_')[0]
		if domain in domains:
			image_name_raw = image_name_base + '.png'
			image_name_mask = image_name_base + '_mask.png'
			
			img_uint16 = imread(os.path.join(raw_path, image_name_raw), as_gray=True) #np.uint16 [0, 65535]
			mask_uint16 = imread(os.path.join(raw_path, image_name_mask), as_gray=True) #np.uint16 [0, 65535]
			img_f64 = resize(img_uint16, (224, 224), preserve_range=True)  #np.float64 [0.0, 65535.0]
			mask_f64 = resize(mask_uint16, (224, 224), order=0, preserve_range=True) #np.float64 [0.0, 65535.0]
			
			#Convert greyscale to RGB greyscale
			img_max = img_f64.max()
			img_min = img_f64.min()
			img_range = img_max - img_min
			mask_max = mask_f64.max()
			if (img_max != 0.0 and img_range > 255.0):
				img_uint8 = np.round(img_f64 / img_max * 255.0).astype(np.uint8) #np.float32 [0, 65535.0]
			else:
				img_uint8 = img_f64.astype(np.uint8)
			if (mask_max != 0.0):
				mask_uint8 = np.floor(mask_f64 / mask_max * 255.0).astype(np.uint8) #np.uint8 [0, 255]
			else:
				mask_uint8 = mask_f64.astype(np.uint8)
			
			print(image_name_base)
			thickness = 3
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
			mask_edge = cv2.Canny(mask_uint8, 250, 255 * 2) #thresholds = Use diagonals to detect strong edges, then connect anything with at least a single edge
			mask_edge_f32 = cv2.dilate(mask_edge.astype('float64'), kernel, iterations = 1).astype(np.float32) #np.float32 [0.0, 255.0]
			mask_edge_f32 = np.where(mask_edge_f32 > 127.0, 1.0, 0.0)
			
			save_path_raw = os.path.join(dest_path, image_name_raw)
			save_path_mask = os.path.join(dest_path, image_name_mask)
			total += 1
			print('Saving #' + str(total) + ' processed raw to:', save_path_raw)
			if (dry_run == False):
				numpngw.write_png(save_path_raw, img_uint8)
				imsave(save_path_mask, mask_edge_f32)
			
	
if __name__ == "__main__":
	domains = ["Jakobshavn", "Helheim", "Sverdrup", "Kangerlussuaq"]
	raw_path = r"../training/data/train_original"
	temp_path = r"../training/data/train_temp"
	dest_path = r"../training/data/train_yara"
	convert(raw_path, temp_path, dest_path, domains, dry_run = False)
		 
	raw_path = r"../training/data/validation_original"
	temp_path = r"../training/data/validation_temp"
	dest_path = r"../training/data/validation_yara"
	convert(raw_path, temp_path, dest_path, domains, dry_run = False)