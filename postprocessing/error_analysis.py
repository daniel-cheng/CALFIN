# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:28:22 2019

@author: Daniel
"""
import numpy as np
import os, glob, cv2, shutil
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import skimage
import meshcut
from ordered_line_from_unordered_points import ordered_line_from_unordered_points_tree
from PIL import Image
from cv2 import VideoWriter, VideoWriter_fourcc, resize
from collections import defaultdict

"""Front Extraction"""
def extract_front_indicators(mask_img, z_score_cutoff=2.0):
    """Extracts an ordered polyline from the processed mask. Also returns an overlay of the extracted polyline and the raw image. Draws to the indexed figure and specified resolution."""
#    print('/t/t/t' + 'extract_front_indicators')
    minimum_points = 4
    
    #Extract known masks
    if mask_img.dtype == np.uint8:
        mask_img = (mask_img / 255.0).astype(np.float32)
    edge_binary = np.where(mask_img > 0.33, 1, 0)
    skeleton = skeletonize(edge_binary)
    front_pixels = np.nonzero(skeleton)
#    empty_image = np.zeros(skeleton.shape)
#    plt.imshow(np.stack((skeleton, empty_image, empty_image), axis=2))
#    plt.show()
    
    #Require a minimum number of points
    if len(front_pixels[0]) < minimum_points:
        return None
    
    #Perform mask to polyline extraction.
    results = ordered_line_from_unordered_points_tree(front_pixels, mask_img.shape, minimum_points, z_score_cutoff)
    overlay = results[2]
    front_line = np.array((results[0], results[1]))
    number_of_points = front_line.shape[1]
    front_normals = np.zeros((2, number_of_points))
    
    #Require a minimum number of points
    if len(front_line[0]) < minimum_points:
        return None
    
    #Calculate normals for endpoints.
    i = 0
    p1 = front_line[:, i]
    p2 = front_line[:, i + 1]
    d21 = p2 - p1
    n21 = np.array([-d21[1], d21[0]])
    n1 = n21
    n1 = n1 / np.linalg.norm(n1)
    front_normals[:,i] = n1
    
    i = number_of_points - 1
    p0 = front_line[:, i - 1]
    p1 = front_line[:, i]
    d10 = p1 - p0
    n10 = np.array([-d10[1], d10[0]])
    n1 = n10
    n1 = n1 / np.linalg.norm(n1)
    front_normals[:,i] = n1
    
    #Calculate normals for all other points.
    for i in range(1, number_of_points - 1):
        p0 = front_line[:, i - 1]
        p1 = front_line[:, i]
        p2 = front_line[:, i + 1]
        d10 = p1 - p0
        d21 = p2 - p1
        n10 = np.array([d10[1], -d10[0]])
        n21 = np.array([d21[1], -d21[0]])
        n1 = n10 + n21
        n1 = n1 / np.linalg.norm(n1)
        front_normals[:,i] = n1
    
    #Draw normals over raw image.
#        raw_rgb_img = np.zeros((512, 512, 3)) + 0.5
#        for i in range(len(front_line[0])):
#            raw_rgb_img[front_line[0, i], front_line[1, i]] = [front_normals[0,i] / 2 + 0.5, front_normals[1,i] / 2 + 0.5, 0.5]
    
#    pts = np.vstack((fitx,ploty)).astype(np.int32).T
    
    return overlay, front_line, front_normals

