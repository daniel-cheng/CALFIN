import os, sys
import numpy as np
from skimage.io import imsave, imread
from skimage.transform import resize
sys.path.insert(0, '../training')
from aug_generators import aug_resize
import numpngw

source_path = r"../training\data\validation"
dest_path = source_path
# Generate mask confidence from masks
augs = aug_resize(img_size=1024)

for file_name in os.listdir(source_path):
	source_file_path = os.path.join(source_path, file_name)
	if '_mask' in file_name:
		mask_img = imread(source_file_path, as_gray = True)
		
		if mask_img.shape[0] > 1100 or mask_img.shape[1] > 1110:
			print(file_name)
			raw_file_name = file_name[0:-9] + '.png'
			mask_file_name = file_name
			
			source_raw_file_path = os.path.join(source_path, raw_file_name)
			raw_img = imread(source_raw_file_path)
			if raw_img.dtype == np.uint8:
				raw_img = raw_img.astype(np.uint16) * 257
			elif raw_img.dtype == np.float64:
				raw_img = (raw_img * 65535).astype(np.uint16)
			
			dat = augs(image=raw_img, mask=mask_img)
			img_aug = dat['image'] #np.uint15 [0, 65535]
			mask_aug = dat['mask'] #np.uint15 [0, 65535]
			
			raw_dest_path = os.path.join(dest_path, raw_file_name)
			mask_dest_path = os.path.join(dest_path, mask_file_name)
			numpngw.write_png(raw_dest_path, img_aug)
			imsave(mask_dest_path, mask_aug)