# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 01:55:14 2019

@author: Daniel
@author: Imanol Luengo (https://stackoverflow.com/a/37744549/1905613)
"""
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, pause
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import math
from collections import defaultdict
from skimage.transform import rescale
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_order
from dfs import longest_undirected_weighted_path
import cv2
from skimage.morphology import skeletonize
from sklearn.neighbors import kneighbors_graph
import random


def is_outlier(data, z_score_cutoff = 2.0):
    """Uses Robust/Modified Z-score method with Median Absolute Deviation for robust univariate outlier estimation.
        Returns a list of booleans that indicate if element is an outlier.
        z_score_cutoff is the number of MAD deviations the point must exceed to be considered an outlier.
        Fallback to MeanAD in case MAD == 0. 1.486 and 1.253314 approximate standard deviation. Why? Unknown.
        See https://www.ibm.com/support/knowledgecenter/SSEP7J_11.1.0/com.ibm.swg.ba.cognos.ug_ca_dshb.doc/modified_z.html"""
    diff = np.abs(data - np.median(data))
    MAD = np.median(diff)
    MeanAD = np.mean(diff)
    is_outlier = [False] * len(data)
    
    if MAD != 0:
        z_score = diff / (1.486 * MAD )
    else:
        z_score = diff / (1.253314 * MeanAD)
        
    is_outlier = z_score >= z_score_cutoff
    return is_outlier

#disallow edges between points on boundary
def ordered_line_from_unordered_points_tree(points_tuple, dimensions, minimum_points, settings):
    """ Algorithm for extracting a single "correct" polyline from pixel edge probablity mask.
    """
#    print('\t\t\t\t' + 'ordered_line_from_unordered_points_tree')
    x = points_tuple[0]
    y = points_tuple[1]
    points = np.c_[x, y]
    k = max(minimum_points - 1, int(np.floor(len(points) * .75)))
    distances = distance_matrix(points, points)
    adjacency_matrix = np.zeros((len(points), len(points)))
#    distances -= 1.0
    mean_cluster_distances = []
    k_closest_indices_list = []
#    min_neighbor_distance = 3
    for row in range(len(distances)):
        k_closest_indices = np.argpartition(distances[row,:], k)[:k]
#        k_nearest_distances = distances[row, k_closest_indices]
#        nearest_distance = distances[row, k_closest_indices[1]] #ensure at least 2 nearby vertices
        adjacency_matrix[row, k_closest_indices] = 1
#        mean_cluster_distances.append(np.mean(k_nearest_distances))
#        k_closest_indices_list.append(k_closest_indices)
        
    indices = list(range(len(x)))
    positions = list(zip(-y, x))
    node_positions = dict(zip(indices, positions))
#    dense_graph_nx = nx.from_numpy_matrix(adjacency_matrix)
#    plt.figure(300 + random.randint(1,500))
#    nx.draw_networkx(dense_graph_nx, pos=node_positions, with_labels=False, node_size = 15)
#    plt.show()
    
    #Eliminate small seperated clusters (outliers/noise)
#    outlier_mask = is_outlier(mean_cluster_distances, z_score_cutoff)
#    print(mean_cluster_distances)
#    for row in range(len(distances)):
#        if outlier_mask[row]:
#            k_closest_indices = k_closest_indices_list[row]
#            adjacency_matrix[row, k_closest_indices] = 0
    
    #use sqrt to penalize large jumps (shorter distances like 1 are given more weight,
    #longer distances reduced more, but longer overall distances are still preserved)
    distances = np.log(distances + 1) * adjacency_matrix
    
    
    mst = minimum_spanning_tree(distances)
    
    
    
    
    #plot intermediate
    # mst_nx = nx.from_scipy_sparse_matrix(mst)
    # plt.figure(800 + random.randint(1,250))
    # nx.draw_networkx(mst_nx, pos=node_positions, with_labels=False, node_size = 15)
    # plt.show()
    
    
    rows, cols = mst.nonzero()
    #penalize long distances after the mst creation
    #this ensures that jumps can still be made, but must connect a reasonable number of new edge
    #in order to account for negative weighting
    #Note, this can't be done before the mst calculation since this would prevent the actual 
    #mst from being effectively found
    #zero point is when the distance weighting starts being negative.
    #power is the exponential penalty for longer distances.
    zero_point = 5 if 'polyline_zero_point' not in settings else settings['polyline_zero_point']
    power = 1.5 if 'polyline_distance_power' not in settings else settings['polyline_distance_power']
    new_distances = np.power(zero_point, power) - np.power(np.power(np.e, mst[rows, cols]) - 1, power)
    mst[rows, cols] = np.ravel(new_distances)
    
#    mst[rows, cols] = mst[rows, cols] - np.min([np.min(mst[rows, cols]), 0])
    
    #mst_nx = nx.from_scipy_sparse_matrix(mst)
    #plt.figure(1050 + random.randint(1,250))
    #nx.draw_networkx(mst_nx, pos=node_positions, with_labels=False, node_size = 15)
    #plt.show()
    
    #Symmetrize matrix to make undriected
    mst = mst + mst.T - np.diag(mst.diagonal())
    mst_array = np.squeeze(np.asarray(mst))
    #Find longest path
#    print(mst_array)
    length, path_indices = longest_undirected_weighted_path(mst_array)
    
    xx = x[path_indices]
    yy = y[path_indices]
    
    # Create an image to draw the lines on
    image = np.zeros((dimensions[0], dimensions[1], 3))
        
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts = np.vstack((yy,xx)).astype(np.int32).T
    
    # Draw the lane onto the warped blank image
#    plt.plot(left_fitx, ploty, color='yellow')
    cv2.polylines(image,  [pts],  False,  (255, 0, 0),  1)#higher thickness may be necessary, but ddecreases accuracy...
#    image = image[:,:,0]
#    print(pts)
#    image_rescaled = rescale(image, 2, anti_aliasing=False)
#    plt.figure(1300 + random.randint(1,500))
#    plt.imshow(image_rescaled)
#    plt.show()
     
#    edge_bianry = np.where(image > 127.0, 1.0, 0.0)
#    skeleton = skeletonize(edge_bianry)
#    skeleton = np.where(skeleton > 0.5, 255.0, 0.0)
#    edge_bianry[:,:,0] = skeleton
#    plt.figure(1800 + random.randint(1,500))
#    plt.imshow(image)
#    plt.show()
    # image = edge_bianry
#    pause(1)
#    plt.show()
    
    #If start point is further from 0, 0 than end point, reverse.
    start_dist = np.linalg.norm([xx[0], yy[0]])
    end_dist = np.linalg.norm([xx[-1], yy[-1]])
    if start_dist > end_dist:
        xx = np.flip(xx)
        yy = np.flip(yy)
    
    return xx, yy, image
