# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:39:27 2020

@author: Daniel
"""

# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1 import ImageGrid
import glob, os
from skimage.transform import resize


def get_title(image_path):
    """Takes in a Landsat image path, and returns date string as well as individual year, month, and day strings."""
    name_split = image_path.split(os.path.sep)[-1].split('_')
    domain = name_split[0]
    satellite = name_split[1]
    if satellite.startswith('S'):
        #Astakhov-Chugunov-Astapenko_S1B_EW_GRDM_1SDH_2018-06-26_011542_01536C_EB6F
        date = name_split[5]
    elif satellite.startswith('L'):
        #Brückner_LC08_L1TP_2015-06-14_232-014_T1_B5_66-1_validation
        date = name_split[3]
    if 'mask' in name_split[-1]:
        mask = 'mask'
    else:
        mask = ''
    title = domain + '\n' + date + ' ' + mask
    return title

def get_title_mohajerani(image_path):
    """Takes in a Landsat image path, and returns date string as well as individual year, month, and day strings."""
    name_split = image_path.split(os.path.sep)[-1].split('_')
    domain = 'Helheim'
    #LC08_L1TP_2015-06-14_232-014_T1_B5_66-1_validation
    date = name_split[3]
    year = date[0:4]
    month = date[4:6]
    day = date[6:8]
    date_dashed = '-'.join([year, month, day])
    title = domain + ' ' + date_dashed
    return title

def get_title_zhang(image_path):
    """Takes in a Landsat image path, and returns date string as well as individual year, month, and day strings."""
    name_split = image_path.split(os.path.sep)[-1].split('_')
    domain = 'Jakobshavn'
    #Jakobshavn-post-2010_TSX-1_2010-05-06T10_05_2027425_modified_2-1_overlay_comparison
    date_dashed = name_split[2].split('T')[0]
    title = domain + '\n' + date_dashed
    return title

def training_grid():
    """Run this script in the folder the images are in and and image pdf should be produced.
    Set desired number of columns with 'x'. Set desired column and row labels with x_label and y_label.
    The more columns you use, the more labels you will have to set. If there is an 'index out of range error',
    make sure the files have the right ending (.TIF).
    """
    folder_all = glob.glob(r"D:\Daniel\Documents\Github\CALFIN Repo\training\data\train\*B[0-9].png")
    total = len(folder_all)
    offset = int(round(total * 0.04))
    folder = folder_all[::offset]
    print(len(folder))
    x = 6 # number of columns in resulting image. Make sure the number of images is divideable by x
    y = 8
    
    fig = plt.figure(dpi=300)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(x, y),
                     axes_pad=0.1,
                     share_all = True)
    actual_range = int(x * y / 2)
    for i in range(actual_range):
        image_path_raw = folder[i]
        image_path_mask = image_path_raw[:-4] + '_mask.png' 
        image_paths = [image_path_raw, image_path_mask]
        for j in range(len(image_paths)):
            index = i * 2 + j
            image_path = image_paths[j]
            image = mpl.image.imread(image_path)
            image = resize(image, (1024, 1024))
            grid[index].imshow(image)
            print(image_path)
            title = get_title(image_path)
            text = grid[index].text(.5, .90, title,
                horizontalalignment='center', color='white', size= 4,
                transform=grid[index].transAxes, wrap=True)
            text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                           path_effects.Normal()])
            grid[index].get_yaxis().set_ticks([])
            grid[index].get_xaxis().set_ticks([])
            grid[index].tick_params(axis='both', which='both', labelsize=0, bottom=False, top=False, left=False)
    fig.savefig(r"D:\Daniel\Documents\Github\CALFIN Repo\paper\grid_training.png", bbox_inches='tight', pad_inches=0, frameon=False)

def validation_grid():
    """Run this script in the folder the images are in and and image pdf should be produced.
    Set desired number of columns with 'x'. Set desired column and row labels with x_label and y_label.
    The more columns you use, the more labels you will have to set. If there is an 'index out of range error',
    make sure the files have the right ending (.TIF).
    """
    folder = glob.glob(r"D:\Daniel\Documents\Github\CALFIN Repo\outputs\calfin_on_calfin_validation\quality_assurance\*\*overlay_comparison.png")
    x = 6 # number of columns in resulting image. Make sure the number of images is divideable by x
    y = 8
    
    plt.close('all')
    index = 0
    total = len(folder)
    fig_counter = 0
    plt.rcParams["figure.figsize"] = (11.75*1.5,8.25*1.5)
    legend = mpl.image.imread(r"D:\Daniel\Documents\Github\CALFIN Repo\paper\legend_validation.png")
    legend = resize(legend, (540, 1024))
    while index < total - 1:
        fig = plt.figure(fig_counter, dpi=300, figsize=(11.75*1.5,8.25*1.5))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(x, y),
                         axes_pad=0.1,
                         share_all = True)
        grid_size = int(x * y)
        grid[0].imshow(legend)
        grid[0].get_yaxis().set_ticks([])
        grid[0].get_xaxis().set_ticks([])
        grid[0].tick_params(axis='both', which='both', labelsize=0, bottom=False, top=False, left=False)
        for j in range(grid_size):
            image_path = folder[index]
            grid_index = j + 1
            print(index, grid_index)
            image = mpl.image.imread(image_path)
            image = resize(image, (1024, 1024))
            grid[grid_index].imshow(image)
            print(image_path)
            title = get_title(image_path)
            text = grid[grid_index].text(.5, .90, title,
                horizontalalignment='center', color='white', size= 6,
                transform=grid[grid_index].transAxes, wrap=True)
            text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                           path_effects.Normal()])
            grid[grid_index].get_yaxis().set_ticks([])
            grid[grid_index].get_xaxis().set_ticks([])
            grid[grid_index].tick_params(axis='both', which='both', labelsize=0, bottom=False, top=False, left=False)
            index = index + 1
            if index >= total - 1 or grid_index >= grid_size - 1:
                break
        fig.savefig(r"D:\Daniel\Documents\Github\CALFIN Repo\paper\grid_validation_calfin_" + str(fig_counter) + ".png", bbox_inches='tight', pad_inches=0.02, frameon=False)
        fig_counter = fig_counter + 1

def validation_mohajerani_grid():
    """Run this script in the folder the images are in and and image pdf should be produced.
    Set desired number of columns with 'x'. Set desired column and row labels with x_label and y_label.
    The more columns you use, the more labels you will have to set. If there is an 'index out of range error',
    make sure the files have the right ending (.TIF).
    """
    folder = glob.glob(r"D:\Daniel\Documents\Github\CALFIN Repo\outputs\calfin_on_mohajerani\quality_assurance\Helheim\*overlay_comparison.png")
    x = 2 # number of columns in resulting image. Make sure the number of images is divideable by x
    y = 6
    
    plt.close('all')
    index = 0
    total = len(folder)
    fig_counter = 0
    plt.rcParams["figure.figsize"] = (11.75*0.75,8.25*0.75)
    
    while index < total - 1:
        fig = plt.figure(fig_counter, dpi=300, figsize=(11.75*0.75,8.25*0.75))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(x, y),
                         axes_pad=0.1,
                         share_all = True)
        grid_size = int(x * y)
        legend = mpl.image.imread(r"D:\Daniel\Documents\Github\CALFIN Repo\paper\legend_validation.png")
        legend = resize(legend, (540, 1024))
        grid[0].imshow(legend)
        grid[0].get_yaxis().set_ticks([])
        grid[0].get_xaxis().set_ticks([])
        grid[0].tick_params(axis='both', which='both', labelsize=0, bottom=False, top=False, left=False)
            
        for j in range(grid_size):
            image_path = folder[index]
            grid_index = j + 1
            print(index, grid_index)
            image = mpl.image.imread(image_path)
            image = resize(image, (1024, 1024))
            grid[grid_index].imshow(image)
            print(image_path)
            title = get_title_mohajerani(image_path)
            text = grid[grid_index].text(.5, .90, title,
                horizontalalignment='center', color='white', size= 6,
                transform=grid[grid_index].transAxes, wrap=True)
            text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                           path_effects.Normal()])
            grid[grid_index].get_yaxis().set_ticks([])
            grid[grid_index].get_xaxis().set_ticks([])
            grid[grid_index].tick_params(axis='both', which='both', labelsize=0, bottom=False, top=False, left=False)
            index = index + 1
            if index > total - 1 or grid_index >= grid_size - 1:
                break
        fig.savefig(r"D:\Daniel\Documents\Github\CALFIN Repo\paper\grid_validation_mohajerani_" + str(fig_counter) + ".png", bbox_inches='tight', pad_inches=0.02, frameon=False)
        fig_counter = fig_counter + 1


def validation_zhang_grid():
    """Run this script in the folder the images are in and and image pdf should be produced.
    Set desired number of columns with 'x'. Set desired column and row labels with x_label and y_label.
    The more columns you use, the more labels you will have to set. If there is an 'index out of range error',
    make sure the files have the right ending (.TIF).
    """
    folder = glob.glob(r"D:\Daniel\Documents\Github\CALFIN Repo\outputs\calfin_on_zhang\quality_assurance\*\*overlay_comparison.png")
    x = 1 # number of columns in resulting image. Make sure the number of images is divideable by x
    y = 9
    
    plt.close('all')
    index = 0
    total = len(folder)
    fig_counter = 0
    plt.rcParams["figure.figsize"] = (11.75*0.75,8.25*0.75)
    
    while index < total - 1:
        fig = plt.figure(fig_counter, dpi=300, figsize=(11.75*0.75,8.25*0.75))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(x, y),
                         axes_pad=0.1,
                         share_all = True)
        grid_size = int(x * y)
        legend = mpl.image.imread(r"D:\Daniel\Documents\Github\CALFIN Repo\paper\legend_validation.png")
        legend = resize(legend, (540, 1024))
        grid[0].imshow(legend)
        grid[0].get_yaxis().set_ticks([])
        grid[0].get_xaxis().set_ticks([])
        grid[0].tick_params(axis='both', which='both', labelsize=0, bottom=False, top=False, left=False)
            
        for j in range(grid_size):
            image_path = folder[index]
            grid_index = j + 1
            print(index, grid_index)
            image = mpl.image.imread(image_path)
            image = resize(image, (1024, 1024))
            grid[grid_index].imshow(image)
            print(image_path)
            title = get_title_zhang(image_path)
            text = grid[grid_index].text(.5, .925, title,
                horizontalalignment='center', color='white', size= 6,
                transform=grid[grid_index].transAxes, wrap=True)
            text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                           path_effects.Normal()])
            grid[grid_index].get_yaxis().set_ticks([])
            grid[grid_index].get_xaxis().set_ticks([])
            grid[grid_index].tick_params(axis='both', which='both', labelsize=0, bottom=False, top=False, left=False)
            index = index + 1
            if index > total - 1 or grid_index >= grid_size - 1:
                break
        fig.savefig(r"D:\Daniel\Documents\Github\CALFIN Repo\paper\grid_validation_zhang_" + str(fig_counter) + ".png", bbox_inches='tight', pad_inches=0.02, frameon=False)
        fig_counter = fig_counter + 1
        plt.show()
     
def validation_baumhoer_grid():
    """Run this script in the folder the images are in and and image pdf should be produced.
    Set desired number of columns with 'x'. Set desired column and row labels with x_label and y_label.
    The more columns you use, the more labels you will have to set. If there is an 'index out of range error',
    make sure the files have the right ending (.TIF).
    """
    folder = glob.glob(r"D:\Daniel\Documents\Github\CALFIN Repo\outputs\calfin_on_baumhoer\quality_assurance\*\*overlay_comparison.png")
    x_list = [4, 6] # number of columns in resulting image. Make sure the number of images is divideable by x
    y_list = [8, 8]
    
    plt.close('all')
    index = 0
    total = len(folder)
    fig_counter = 0
    plt.rcParams["figure.figsize"] = (11.75*0.75,8.25*0.75)
    
    while index < total - 1:
        x = x_list[fig_counter]
        y = y_list[fig_counter]
        fig = plt.figure(fig_counter, dpi=300, figsize=(11.75*0.75,8.25*0.75))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(x, y),
                         axes_pad=0.1,
                         share_all = True)
        grid_size = int(x * y)
        legend = mpl.image.imread(r"D:\Daniel\Documents\Github\CALFIN Repo\paper\legend_validation.png")
        legend = resize(legend, (540, 1024))
        grid[0].imshow(legend)
        grid[0].get_yaxis().set_ticks([])
        grid[0].get_xaxis().set_ticks([])
        grid[0].tick_params(axis='both', which='both', labelsize=0, bottom=False, top=False, left=False)
            
        for j in range(grid_size):
            image_path = folder[index]
            grid_index = j + 1
            print(index, grid_index)
            image = mpl.image.imread(image_path)
            image = resize(image, (1024, 1024))
            grid[grid_index].imshow(image)
            print(image_path)
            title = get_title(image_path)
            text = grid[grid_index].text(.5, .92, title,
                horizontalalignment='center', color='white', size= 5,
                transform=grid[grid_index].transAxes, wrap=True)
            text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                           path_effects.Normal()])
            grid[grid_index].get_yaxis().set_ticks([])
            grid[grid_index].get_xaxis().set_ticks([])
            grid[grid_index].tick_params(axis='both', which='both', labelsize=0, bottom=False, top=False, left=False)
            index = index + 1
            if index > total - 1 or grid_index >= grid_size - 1:
                break
        fig.savefig(r"D:\Daniel\Documents\Github\CALFIN Repo\paper\grid_validation_baumhoer_" + str(fig_counter) + ".png", bbox_inches='tight', pad_inches=0.02, frameon=False)
        fig_counter = fig_counter + 1
        x = fig_counter
        plt.show()
if __name__ == "__main__":
    #training_grid()
#    validation_grid()
#    validation_mohajerani_grid()
#    validation_zhang_grid()
    validation_baumhoer_grid()