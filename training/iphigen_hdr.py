import sys
sys.path.insert(0, 'iphigen')
from iphigen import core, utils
import matplotlib.pyplot as plt
import numpy as np

def iphigen_hdr(data):
	data = np.asarray(data, dtype=float)
	# Compute intensity
	inten = np.sum(data, axis=-1)
	# Compute barycentic coordinates
	bary = data / (inten[..., np.newaxis] + 1e-6)
	scales = [15, 80, 250]
	int_bal_perc = [1.0, 99.0]

	inten = utils.truncate_range(inten, pmin=int_bal_perc[0], pmax=int_bal_perc[1])
	inten = utils.set_range(inten, zero_to=255*data.shape[-1])
	
	data = bary * inten[..., None]
	# Update barycentic coordinates
	bary = data / (inten[..., None] + 1e-6)
			
	new_inten = core.multi_scale_retinex(inten, scales=scales, verbose=False)
	# Scale back to the approximage original intensity range
	inten = core.scale_approx(new_inten, inten)
	
	# Insert back the processed intensity image
	
	data = bary * inten[..., None]
	#print(data.max(), inten.max(), new_inten.max(), bary.max())
	
	#clip values
	
	data = np.round(np.clip(data, 0.0, 255.0)).astype(np.uint8)
	
	return data

if __name__ == "__main__":
	test2 = iphigen_hdr(img_3_uint8)
