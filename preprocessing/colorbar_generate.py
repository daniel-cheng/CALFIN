# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 19:39:44 2020

@author: Daniel
"""


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.dates import YearLocator
import matplotlib.dates as mdates
from datetime import datetime

#Initialize colorbar shape
def horizontal():
    plt.close('all')
    plt.rcParams["font.size"] = "20"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    # plt.subplots_adjust(top=0.925, bottom=0.5, right=0.95, left=0.05, hspace=0.25, wspace=0.25)
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=1972, vmax=2019)
    
    #Setup colorbar range
    datefunc = lambda x: mdates.date2num(datetime.strptime(x, '%Y-%m-%d'))
    start_year = 1972.75
    middle_year = 1985.75
    end_year = 2020
    dotted_months = int((middle_year - start_year) * 12) #Start Sept. 1972, end June 2019
    lined_months = int((end_year - middle_year) * 12) #Start Sept. 1972, end June 2019
    start_time = datefunc('1972-09-01')
    middle_time = datefunc('1985-01-01')
    end_time = datefunc('2019-07-01')
    
    #Draw dummy image and hide it to get colorbar to show up
    a = np.array([[start_time,end_time]])
    plt.figure(figsize=(16, 1.0))
    img = plt.imshow(a, cmap=cmap)
    plt.gca().set_visible(False)
    cax = plt.axes([0.05, 0.35, 0.90, 0.3])
    cb1 = plt.colorbar(orientation="horizontal", cax=cax, cmap=cmap, norm=norm)
    
    #Perform colorbar label formatting
    ax = cb1.ax
    min_loc = YearLocator(1)
    maj_loc = YearLocator(5)
    min_formatter = mdates.DateFormatter('')
    maj_formatter = mdates.DateFormatter('%Y')
    ax.xaxis.set_minor_locator(min_loc)
    ax.xaxis.set_minor_formatter(min_formatter)
    ax.xaxis.set_major_locator(maj_loc)
    ax.xaxis.set_major_formatter(maj_formatter)
    ax.xaxis.set_tick_params(labelsize=14, width=4, length=8)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")   
    # ax.set_xticklabels([], minor=True, width=4, length=8)
    ax.set_title('Year', fontsize=22, fontweight='bold')
        
    
    #Save the colorbar
    colorbar_path = r"../paper/colorbar_horizontal.png"
    plt.savefig(colorbar_path, bbox_inches='tight', pad_inches=0.05, frameon=False)
        

def vertical():
    plt.close('all')
    plt.rcParams["font.size"] = "20"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    # plt.subplots_adjust(top=0.925, bottom=0.5, right=0.95, left=0.05, hspace=0.25, wspace=0.25)
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=1972, vmax=2019)
    
    #Setup colorbar range
    datefunc = lambda x: mdates.date2num(datetime.strptime(x, '%Y-%m-%d'))
    start_year = 1972.75
    middle_year = 1985.75
    end_year = 2020
    dotted_months = int((middle_year - start_year) * 12) #Start Sept. 1972, end June 2019
    lined_months = int((end_year - middle_year) * 12) #Start Sept. 1972, end June 2019
    start_time = datefunc('1972-09-01')
    middle_time = datefunc('1985-01-01')
    end_time = datefunc('2019-07-01')
    
    #Draw dummy image and hide it to get colorbar to show up
    a = np.array([[start_time,end_time]])
    plt.figure(figsize=(1, 16))
    img = plt.imshow(a, cmap=cmap)
    plt.gca().set_visible(False)
    cax = plt.axes([0.5, 0.05, 0.3, 0.85])
    cb1 = plt.colorbar(orientation="vertical", cax=cax, cmap=cmap, norm=norm)
    
    #Perform colorbar label formatting
    ax = cb1.ax
    min_loc = YearLocator(1)
    maj_loc = YearLocator(5)
    min_formatter = mdates.DateFormatter('')
    maj_formatter = mdates.DateFormatter('%Y')
    ax.tick_params(direction='out')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_minor_locator(min_loc)
    ax.yaxis.set_minor_formatter(min_formatter)
    ax.yaxis.set_major_locator(maj_loc)
    ax.yaxis.set_major_formatter(maj_formatter)
    ax.yaxis.set_tick_params(labelsize=14, width=4, length=8)
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")   
    # ax.set_xticklabels([], minor=True, width=4, length=8)
    ax.set_title('Year   ', fontsize=22, fontweight='bold', horizontalalignment='center')
        
    
    #Save the colorbar
    colorbar_path = r"../paper/colorbar_vertical.png"
    plt.savefig(colorbar_path, bbox_inches='tight', pad_inches=0.05, frameon=False)
    
if __name__ == '__main__':
    vertical()