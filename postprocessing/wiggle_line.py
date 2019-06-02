# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:28:22 2019

@author: Daniel
"""
import numpy as np
import os, glob, cv2
import matplotlib.pyplot as plt
from scipy.misc import imresize
from scipy.ndimage import distance_transform_edt
from redistribute_points import redistribute_points

steps = ['1', '2', '3']
steps = ['3']

#Load domain to process
if '1' in steps:
    def landsat_sort(file_path):
        return file_path.split(os.path.sep)[-1].split('_')[3]
    
    raw_paths = glob.glob(r'D:\Daniel\Documents\Github\CALFIN Repo\postprocessing\testing\*[0-9].png')
    raw_paths_calfin = glob.glob(r'D:\Daniel\Documents\Github\CALFIN Repo\postprocessing\testing_calfin\*_raw.png')
    raw_paths_all = raw_paths + raw_paths_calfin
    raw_paths.sort(key=landsat_sort)
    raw_paths_calfin.sort(key=landsat_sort)
    raw_paths_all.sort(key=landsat_sort)
    
    mask_paths = glob.glob(r'D:\Daniel\Documents\Github\CALFIN Repo\postprocessing\testing\*_mask.png')
    mask_paths_calfin = glob.glob(r'D:\Daniel\Documents\Github\CALFIN Repo\postprocessing\testing_calfin\*_pred.png')
    mask_paths_all = mask_paths + mask_paths_calfin
    mask_paths.sort(key=landsat_sort)
    mask_paths_calfin.sort(key=landsat_sort)
    mask_paths_all.sort(key=landsat_sort)
    
    first_mask_path = mask_paths[0]
    first_mask_img = cv2.imread(first_mask_path, 0)

#Extract known masks
if '2' in steps:
    width, height = (512, 512)
    def extract_front(raw_path, mask_path):
        raw_img = imresize(cv2.imread(raw_path, 0), (width, height), interp='bicubic').astype('uint8')
        mask_img = imresize(cv2.imread(mask_path, 0), (width, height), interp='bicubic').astype('uint8')
        raw_rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
        
        cimg, contours, hierarchy = cv2.findContours(mask_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        
        front = contours[0]
        cv2.drawContours(raw_rgb_img, [front], -1, (0,255,0), 3)
        return raw_rgb_img, contours
    
    fronts = []
    for i in range(0, len(mask_paths)):
        raw_path = raw_paths[i]
        mask_path = mask_paths[i]
        result = extract_front(raw_path, mask_path)
        
        overlay = result[0]
        front = result[1]
        fronts.append(front)
        
        plt.figure()
        plt.imshow(overlay)
        plt.show()
        
if '3' in steps:
    def evolve_front(previous_front, raw_path, pred_path):
        raw_img = imresize(cv2.imread(raw_path, 0), (width, height), interp='bicubic').astype('uint8')
        pred_img = imresize(cv2.imread(pred_path, 0), (width, height), interp='bicubic').astype('uint8')
        raw_rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
        
#        cimg, contours, hierarchy = cv2.findContours(mask_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
#        contour_img = np.zeros((width, height, 3))
#        cv2.drawContours(contour_img, [previous_front], -1, (255,255,255), 3)
        pred_img[pred_img < 127] = 0
        pred_img[pred_img >= 127] = 255
        np.logical_not(pred_img)
        
        distance_map = distance_transform_edt(np.logical_not(pred_img))
        gradient_map = np.gradient(distance_map)
        
        new_front = previous_front
        
        epsilon = 0.25
        alpha = 2
        resolution = len(previous_front)
#        beta = 0.5
        
        #Track number of discontinious events
        #Bayesian model of size/frequency of calving events vs frequency, and use tha input for 
        #second derivative information
        #judge objective function over entire time series
        # events are uniform, but size is a distribution (model many different poisson processes)
        
        while True:
            #Fix boundary points
            for i in range(1, len(new_front) - 2):
                x = new_front[i][0][0]
                y = new_front[i][0][1]
                #derivative information
                dx = gradient_map[0][np.clip(int(x), 0, 511)][np.clip(int(y), 0, 511)]
                dy = gradient_map[1][np.clip(int(x), 0, 511)][np.clip(int(y), 0, 511)]
                x += dx * alpha
                y += dy * alpha
                new_front[i][0][0] = x
                new_front[i][0][1] = y
            new_front_x, new_front_y = redistribute_points(new_front[:,0,0], new_front[:,0,1], resolution)
            new_front[:,0,0] = new_front_x
            new_front[:,0,1] = new_front_y
            changes = new_front - previous_front
            normalized_changes = changes / len(new_front)
            if np.sum(normalized_changes) < epsilon:
                break
            
        new_front = new_front.astype(int)
        cv2.drawContours(raw_rgb_img, [new_front], -1, (255,255,255), 3)
        cv2.drawContours(pred_img, [new_front], -1, (255,255,255), 3)
        return raw_rgb_img, new_front, pred_img, gradient_map
    
    first_front = fronts[0][0]
    
    estimated_fronts = [first_front]
#    for i in range(1, len(mask_paths_all)):
    for i in range(2, 15):
        raw_path = raw_paths_all[i]
        mask_path = mask_paths_all[i]
        
        result = evolve_front(estimated_fronts[-1], raw_path, mask_path)
        
        overlay = result[0]
        new_front = result[1]
        distance_map = result[2]
        gradient_map = result[3]
        estimated_fronts.append(new_front)
        
#        plt.figure()
#        plt.imshow(overlay)
        plt.figure()
        plt.imshow(distance_map)
        plt.show()