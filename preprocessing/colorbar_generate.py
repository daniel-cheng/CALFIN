# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 19:39:44 2020

@author: Daniel
"""


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

fig, ax = plt.subplots(figsize=(1, 6))
fig.subplots_adjust(right=0.5)

cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=1972, vmax=2019)
num_ticks = 2019-1972+1
ticks = np.linspace(0, 1, num_ticks)
labels = np.linspace(1972, 2019, num_ticks)
ticks_int = ticks[::2].astype(int)
labels_int = labels[::2].astype(int)

for file_path in glob.glob(r'D:\Daniel\Documents\Github\CALFIN Repo\paper\qgis_screenshots\*'):
	

	cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
	                                norm=norm,
	                                orientation='vertical')
	
	
	cb1.set_ticks(labels_int)
	cb1.set_ticklabels(labels_int)
	imread(file_path)
	fig.imshow(
	fig.show()