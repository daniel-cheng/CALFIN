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
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_order
from dfs import longest_undirected_weighted_path

def ordered_line_from_unordered_points(x, y):
	points = np.c_[x, y]
	clf = NearestNeighbors(math.floor(2)).fit(points)
	G = clf.kneighbors_graph()
	T = nx.from_scipy_sparse_matrix(G)
	
	paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(points))]
	paths_segments = defaultdict(list)
	for i in range(len(paths)):
		paths_segments[len(paths[i])].append(paths[i])
	
	
#	plt.figure()
	for length, paths_segment in paths_segments.items():
		mindist = np.inf
		minidx = 0
		
		#one if len(p) == len(points)
		for i in range(length):
			p = paths_segment[i]		   # order of nodes
			ordered = points[p]	# ordered nodes
			# find cost of that order by the sum of euclidean distances between points (i) and (i+1)
			cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
			if cost < mindist:
				mindist = cost
				minidx = i
				
		#one to combine or handle broken pieces
		opt_order = paths_segment[minidx]
		
		xx = x[opt_order]
		yy = y[opt_order]
#		plt.plot(xx, yy)
#		pause(1)
#	plt.show()
	
	#If start point is further from 0, 0 than end point, reverse.
	start_dist = np.linalg.norm([xx[0], yy[0]])
	end_dist = np.linalg.norm([xx[-1], yy[-1]])
	if start_dist > end_dist:
		xx = np.flip(xx)
		yy = np.flip(yy)
	
	return xx, yy

def is_outlier(data, m = 18.0):
	diff = np.abs(data - np.median(data))
	mdev = np.median(diff)
	is_outlier = [False] * len(data)
	for i in range(len(diff)):
		if mdev != 0:
			sigma = diff[i] / mdev 
		else:
			sigma = 0.0
		is_outlier[i] = sigma >= m
	return is_outlier

#disallow edges between points on boundary
def ordered_line_from_unordered_points_tree(points_tuple, dimensions, minimum_points):
	x = points_tuple[0]
	y = points_tuple[1]
	points = np.c_[x, y]
	k = max(minimum_points - 1, int(np.floor(len(points) /25)))
	distances = distance_matrix(points, points)
	mean_cluster_distances = []
	for row in range(len(distances)):
		k_closest_indices = np.argpartition(distances[row,:], k)[:k]
		k_nearest_distances = distances[row, k_closest_indices[:k]]
		mean_cluster_distances.append(np.mean(k_nearest_distances))
		
	#Eliminate small seperated clusters (outliers/noise)
	outlier_mask = is_outlier(mean_cluster_distances, 25)
	for row in range(len(distances)):
		try:
			if outlier_mask[row]:
				for col in range(row, len(distances)):
					distances[row, col] = 0
					distances[col, row] = 0
		except TypeError as e:
			print(e)
	
	mst = minimum_spanning_tree(distances)
	#Symmetrize matrix to make undriected
	rows, cols = mst.nonzero()
	mst[cols, rows] = mst[rows, cols]
	mst_array = mst.toarray()
	#Find longest path
	length, path_indices = longest_undirected_weighted_path(mst_array)
	
	
	xx = x[path_indices]
	yy = y[path_indices]
	plt.plot(yy, xx, 'r')
#	pause(1)
#	plt.show()
	
	#If start point is further from 0, 0 than end point, reverse.
	start_dist = np.linalg.norm([xx[0], yy[0]])
	end_dist = np.linalg.norm([xx[-1], yy[-1]])
	if start_dist > end_dist:
		xx = np.flip(xx)
		yy = np.flip(yy)
	
	return xx, yy