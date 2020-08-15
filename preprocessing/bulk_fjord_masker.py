import os, re, glob, sys
import numpy as np
from skimage.io import imsave, imread
import skimage
import numpngw
#skimage.io.use_plugin('freeimage')
sys.path.insert(0, '../training')
from aug_generators import aug_resize

augs = aug_resize(img_size=1024)


dry_run = False

root_path = r"D:\Daniel\Documents\Github\CALFIN Repo\processing\landsat_raw"
dest_path = r"..\reprocessing\fjord_boundaries" 

for domain in os.listdir(root_path):		
	print(domain)
#	if domain != 'Petermann':
#		continue
	source_domain_path = os.path.join(root_path, domain)
	dest_domain_path = os.path.join(dest_path)
	if not os.path.exists(dest_domain_path):
		os.mkdir(dest_domain_path)
	
	domain_mask_img = None
	domain_raw_img = None
	counter = 0
#	for mask_path in glob.glob(os.path.join(source_domain_path, '*_mask.png')):
#		mask_img = imread(mask_path, as_gray = True).astype(np.float32)
#		
#		#Add to batches
#		if domain_mask_img is not None:
#			domain_mask_img += mask_img #np.float32 [-1.0, 1.0], imagenet mean (~0.45)
#			counter += 1
#		else:
#			domain_mask_img = mask_img
#	domain_mask_img = 1.0 - domain_mask_img / domain_mask_img.max()
	
	file_list = glob.glob(os.path.join(source_domain_path, '*.png'))
	file_list = reversed(list(filter(lambda x: '_mask' not in x, file_list)))
	for raw_path in file_list:
#		print(raw_path)
		raw_img = imread(raw_path, as_gray = True).astype(np.float32)
		
		#Add to batches
		if domain_raw_img is not None:
			domain_raw_img += raw_img #np.float32 [-1.0, 1.0], imagenet mean (~0.45)
			counter += 1
		else:
			domain_raw_img = raw_img
		if counter > 3:
			break
	domain_raw_img = 1.0 - domain_raw_img / domain_raw_img.max()
					
	
#	mask_3_uint8 = np.stack((mask_uint8,)*3, axis=-1)
#	

	save_mask_path = os.path.join(dest_domain_path, domain + "_fjord_boundaries.png")
	save_raw_path = os.path.join(dest_domain_path, domain + "_raw.png")
	print('Saving processed fjord boundary mask to:', save_mask_path)
	if (dry_run == False):
#		print(domain_mask_img.shape)
		
		img = domain_raw_img
		if img.dtype == np.uint8:
			img = img.astype(np.uint16) * 257
		elif img.dtype == np.float64:
			img = (img * 65535).astype(np.uint16)
			
		dat = augs(image=img)
		img_aug = dat['image'] #np.uint15 [0, 65535]
						
		imsave(save_mask_path, img_aug)
	
#		imsave(save_raw_path, domain_raw_img)