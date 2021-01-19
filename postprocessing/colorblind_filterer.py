import os, re, glob, sys
import numpy as np
from skimage.io import imsave, imread
import skimage
# import numpngw
#skimage.io.use_plugin('freeimage')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

dry_run = False
#raw_path = r"../training/data/train_raw"
#temp_path = r"../training/data/train_temp"
#dest_path = r"../training/data/train_processed"

# raw_path = r"../training/data/validation_raw"
# temp_path = r"../training/data/validation_temp"
# dest_path = r"../training/data/validation_processed"

# raw_path = r"../training/data/temp_raw"
# temp_path = r"../training/data/temp_temp"
# dest_path = r"../training/data/temp_processed"

# raw_path = r"../../CALFIN Repo Intercomp/processing/raw"
# temp_path = r"../../CALFIN Repo Intercomp/processing/temp"
# dest_path = r"../../CALFIN Repo Intercomp/processing/processed"

# raw_path = r"../preprocessing/calvingfrontmachine/landsat_raw"
# temp_path = r"../processing/landsat_raw_temp"
# dest_path = r"../processing/landsat_raw_processed"

# raw_path = r"../processing/landsat_raw"
# temp_path = r"../processing/landsat_raw_temp"
# dest_path = r"../processing/landsat_raw_processed"

raw_path = r"D:\Daniel\Documents\Github\CALFIN Repo\paper"
dest_path = r"D:\Daniel\Documents\Github\CALFIN Repo\paper"

# file_names = ["validation_calfin_tabled.png",
#              "validation_baumhoer_tabled.png", 
#              "validation_mohajerani_tabled.png", 
#              "validation_zhang_tabled.png",
#              "pipeline-postprocess.png"]
# file_names = ["mask_to_polyline.png"]
# file_names = ["pipeline-postprocess.png"]
for file_name in file_names:
    file_path = os.path.join(raw_path, file_name)
    out_name = file_name[0:-4] + '_cb_temp.png'
    out_path = os.path.join(dest_path, out_name)
    
    raw_img = imread(file_path)
    raw_img = raw_img[:,:,0:3]
    
    red_mask = raw_img[:,:,0] >= 255 * 0.5
    green_mask = raw_img[:,:,1] >= 255 * 0.5
    blue_mask = raw_img[:,:,2] >= 255 * 0.5
    
    red_mask = median_filter(red_mask, size=3)
    green_mask = median_filter(green_mask, size=3)
    blue_mask = median_filter(blue_mask, size=3)
    yellow_mask = np.logical_and(red_mask, green_mask)
    white_mask = np.logical_and(yellow_mask, blue_mask)
    nonwhite_mask = np.logical_not(white_mask)
    
    #Tune down colors in raw image to increase contrast 
    raw_gray_img = np.mean(raw_img, axis=2, keepdims=True)
    grayness = 0.25
    raw_img = raw_img * (1 - grayness) + raw_gray_img * grayness
    
    #determine indices of pixels to be changed
    red_nonwhite_indices = np.logical_and(red_mask, nonwhite_mask).nonzero()
    green_nonwhite_indices = np.logical_and(green_mask, nonwhite_mask).nonzero()
    yellow_nonwhite_indices = np.logical_and(yellow_mask, nonwhite_mask).nonzero()
    blue_nonwhite_indices = np.logical_and(blue_mask, nonwhite_mask).nonzero()
    
    #Red blue yellow scheme    
    red_replace_color = [216, 27, 96] #red
    green_replace_color = [79, 86, 255] #blue
    yellow_replace_color = [255, 218, 106]
    blue_replace_color = [216, 27, 96] #red
    blue_replace_color = [86, 235, 243] #cyan
    
    #replace the colors
    # raw_img[red_nonwhite_indices] = red_replace_color
    # raw_img[green_nonwhite_indices] = green_replace_color
    # raw_img[yellow_nonwhite_indices] = yellow_replace_color
    raw_img[blue_nonwhite_indices] = blue_replace_color
    raw_img = raw_img.astype(np.uint8)
    
    plt.close('all')
    plt.figure()
    plt.imshow(raw_img)
    plt.figure()
    plt.imshow(np.logical_and(blue_mask, nonwhite_mask))
    # plt.figure()
    # plt.imshow(nonwhite_mask)
    # plt.figure()
    # plt.imshow(green_mask)
    # plt.figure()
    # plt.imshow(yellow_mask)
    # plt.figure()
    # plt.imshow(np.logical_and(yellow_mask, nonwhite_mask))
    # plt.figure()
    # plt.imshow(np.logical_and(red_green_yellow_mask, nonwhite_mask))
    plt.show()
    
    imsave(out_path, raw_img)