# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 18:06:26 2019

@author: Daniel
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave
from matplotlib.lines import Line2D
import os, cv2

os.environ['GDAL_DATA'] = r'D:\\ProgramData\\Anaconda3\\envs\\cfm\\Library\\share\\gdal' #Ensure crs are exported correctly by gdal/osr/fiona

from osgeo import gdal, osr

def tif_save(settings, metrics, dest_path, array):
#def pngToGeotiff(array:np.ndarray, domain:str, bounds:dict, dest_path:str) -> ('Driver', 'Dataset'):
    image_settings = settings['image_settings']
    domain = image_settings['domain']
    
    #write out shp file
    date_index = settings['date_index']
    image_settings = settings['image_settings']
    image_name_base = image_settings['image_name_base']
    image_name_base_parts = image_name_base.split('_')
    
    domain = image_name_base_parts[0]
    date =  image_name_base_parts[date_index]
    year = date.split('-')[0]
    tif_name = image_name_base + '.tif'
    source_tif_path = os.path.join(settings['tif_source_path'], domain, year, tif_name)
    
    if date == '2005-05-09':
        print('alert')
    # Load geotiff and get domain layer/bounding box of area to mask
    geotiff = gdal.Open(source_tif_path)
    
    #Get bounds
    geoTransform = geotiff.GetGeoTransform()
    xMin = geoTransform[0]
    yMax = geoTransform[3]
    xMax = xMin + geoTransform[1] * geotiff.RasterXSize
    yMin = yMax + geoTransform[5] * geotiff.RasterYSize
    
    #Get projection
    prj = geotiff.GetProjection()
    srs = osr.SpatialReference(wkt=prj)
    if srs.GetAttrValue("PROJCS|AUTHORITY", 1) is not None:
        rasterCRS = int(srs.GetAttrValue("PROJCS|AUTHORITY", 1))
    elif srs.GetAttrValue("AUTHORITY", 1) is not None:
        rasterCRS = int(srs.GetAttrValue("AUTHORITY", 1))
    else:
        rasterCRS = 32621
    
    #Transform from scaled pixel coordaintes to fractional scaled fractional original to original image to geotiff coordinates
    full_size = settings['full_size']
    bounding_box = image_settings['actual_bounding_box']
    fractional_bounding_box = np.array(bounding_box) / full_size
    
    xRange = xMax - xMin
    yRange = yMax - yMin
    subset_xMin = fractional_bounding_box[1] * xRange + xMin
    subset_xMax = (fractional_bounding_box[1] + fractional_bounding_box[3]) * xRange + xMin
    subset_yMax = yMax - fractional_bounding_box[0] * yRange
    subset_yMin = yMax - (fractional_bounding_box[0] + fractional_bounding_box[2]) * yRange
    
    bounds = {'xMin':subset_xMin, 'xMax':subset_xMax, 'yMin':subset_yMin, 'yMax':subset_yMax}
    
    # TODO: Fix X/Y coordinate mismatch and use ns/ew labels to reduce confusion. Also, general cleanup and refactoring.
    array = np.flip(array, axis=0)
    h, w = array.shape[:2]
    x_pixels = w  # number of pixels in x
    y_pixels = h  # number of pixels in y
    x_pixel_size = (bounds['xMax'] - bounds['xMin']) / x_pixels  # size of the pixel...        
    y_pixel_size = (bounds['yMax']  - bounds['yMin']) / y_pixels  # size of the pixel...        
    x_min = bounds['xMin'] 
    y_max = bounds['yMax']   # x_min & y_max are like the "top left" corner.
    
    
    driver = gdal.GetDriverByName('GTiff')
    
    dataset = driver.Create(
        dest_path,
        x_pixels,
        y_pixels,
        3,
        gdal.GDT_Byte, )
    
    dataset.SetGeoTransform((
        x_min,    # 0
        x_pixel_size,  # 1
        0,                      # 2
        y_max,    # 3
        0,                      # 4
        -y_pixel_size))  #6
    
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(rasterCRS)
    
    array = np.flipud(array)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(array[:,:,0])
    dataset.GetRasterBand(2).WriteArray(array[:,:,1])
    dataset.GetRasterBand(3).WriteArray(array[:,:,2])
    dataset.FlushCache()  # Write to disk.
    return dataset, dataset.GetRasterBand(1)  #If you need to return, remenber to return  also the dataset because the band don`t live without dataset.

def plot_validation_results(settings, metrics):
    """Plots a standardized set of 6 plots for validation of the neural network, and quantifies its error per image."""
    empty_image = settings['empty_image']
    scaling = settings['scaling']
    saving = settings['saving']
    plotting = settings['plotting']
    show_plots = settings['show_plots']
    dest_path_qa = settings['dest_path_qa']
    image_settings = settings['image_settings']
    if 'rerun' in settings:
        rerun = settings['rerun']
    else:
        rerun = False
    
    image_name_base = image_settings['image_name_base']
    bounding_box = image_settings['actual_bounding_box']
    meters_per_subset_pixel = image_settings['meters_per_subset_pixel']
    distances = image_settings['distances']
    raw_image = image_settings['raw_image']
    unprocessed_original_raw = image_settings['unprocessed_original_raw']
    original_raw = image_settings['original_raw']
    pred_image = image_settings['pred_image']
    polyline_image = image_settings['polyline_image_dilated']
    mask_image = image_settings['mask_final_dilated_f32']
    index = str(image_settings['i'] + 1) + '-' + str(image_settings['box_counter'])
    edge_iou_score = image_settings['edge_iou_score_subset']
    distances = image_settings['distances']
    distances_meters = distances * meters_per_subset_pixel
    
    #Set figure size for 1600x900 resolution, tight layout
    plt.rcParams["figure.figsize"] = (16,9)
    
    #Create the color key for each subplots' legends    
    preprocess_legend = [Line2D([0], [0], color='#ff0000', lw=4),
                         Line2D([0], [0], color='#00ff00', lw=4),
                         Line2D([0], [0], color='#0000ff', lw=4)]
    nn_legend = [Line2D([0], [0], color='#00ff00', lw=4),
                 Line2D([0], [0], color='#ff0000', lw=4)]
    front_legend = [Line2D([0], [0], color='#ff0000', lw=4)]
    comparison_legend = [Line2D([0], [0], color='#ff0000', lw=4),
                         Line2D([0], [0], color='#00ff00', lw=4)]
#    polyline_image[:,:,0] = skeletonize(polyline_image[:,:,0])
    #Begin plotting the 2x3 validation results output
    original_raw_gray = np.clip(np.stack((original_raw[:,:,0] / 255.0, original_raw[:,:,0] / 255.0, original_raw[:,:,0] / 255.0), axis=-1), 0.0, 1.0)
#    raw_image_gray = np.stack((raw_image[:,:,0], raw_image[:,:,0], raw_image[:,:,0]), axis=-1)
    start_point = (int(bounding_box[1]), int(bounding_box[0]))
    end_point = (int(bounding_box[1] + bounding_box[3]), int(bounding_box[0] + bounding_box[2]))
    original_raw_gray_patched = cv2.rectangle(original_raw_gray * 255, start_point, end_point, (255, 0, 0), 1).astype(np.uint8)
    raw_image = np.clip(raw_image, 0.0, 1.0)
    pred_image = np.clip(pred_image, 0.0, 1.0)
    extracted_front = np.clip(np.stack((polyline_image[:,:,0], empty_image, empty_image), axis=-1) + raw_image * 0.8, 0.0, 1.0)
    overlay = np.clip(np.stack((polyline_image[:,:,0], mask_image[:,:,0], empty_image), axis=-1) + raw_image * 0.8, 0.0, 1.0)
    
    if plotting:
        #Initialize plots
        hist_bins = 20
        f, axarr = plt.subplots(2, 3, num=index)
        f.suptitle(image_name_base, fontsize=18, weight='bold')
    
        axarr[0,0].imshow(original_raw_gray_patched)
        axarr[0,0].set_title(r'$\bf{a)}$ Raw Subset')
        
        axarr[0,1].imshow(raw_image)
        axarr[0,1].set_title(r'$\bf{b)}$ Preprocessed Input')
        axarr[0,1].legend(preprocess_legend, ['Raw', 'HDR', 'S/H'], prop={'weight': 'normal'}, facecolor='#eeeeee', loc='upper center', bbox_to_anchor=(0.5, 0.0), shadow=True, ncol=3)
        axarr[0,1].axis('off')
        
        axarr[0,2].imshow(pred_image)
        axarr[0,2].set_title(r'$\bf{c)}$ NN Output')
        axarr[0,2].legend(nn_legend, ['Land/Ice', 'Front'], prop={'weight': 'normal'}, facecolor='#eeeeee', loc='upper center', bbox_to_anchor=(0.5, 0.0), shadow=True, ncol=2)
        axarr[0,2].axis('off')
        
        axarr[1,0].imshow(extracted_front)
        axarr[1,0].set_title(r'$\bf{d)}$ Extracted Front')
        axarr[1,0].legend(front_legend, ['Front'], prop={'weight': 'normal'}, facecolor='#eeeeee', loc='upper center', bbox_to_anchor=(0.5, 0.0), shadow=True, ncol=1)
        axarr[1,0].axis('off')
        
        axarr[1,1].imshow(overlay)
        axarr[1,1].set_title(r'$\bf{e)}$ NN vs Ground Truth Front')
        axarr[1,1].set_xlabel('Jaccard Index: {:.4f}'.format(edge_iou_score))
        axarr[1,1].legend(comparison_legend, ['NN', 'GT'], prop={'weight': 'normal'}, facecolor='#eeeeee', loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)
        axarr[1,1].tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off') # labels along the bottom edge are off
        
        # which = both major and minor ticks are affected
        axarr[1,2].hist(distances_meters, bins=hist_bins, range=[0.0, 20.0 * scaling])
        axarr[1,2].set_xlabel('Distance to nearest point (mean=' + '{:.2f}m)'.format(np.mean(distances_meters)))
        axarr[1,2].set_ylabel('Number of points')
        axarr[1,2].set_title(r'$\bf{f)}$ Per-pixel Pairwise Error (meters)')
        
        #Refresh plot if necessary
        plt.subplots_adjust(top = 0.90, bottom = 0.075, right = 0.975, left = 0.025, hspace = 0.3, wspace = 0.2)
        f.canvas.draw()
        f.canvas.flush_events()
    
    #Save figure
    if saving:
        domain = image_settings['domain']
        dest_path_qa_domain = os.path.join(dest_path_qa, domain)
        dest_path_qa_bad_domain = os.path.join(dest_path_qa + '_bad', domain)
        if not os.path.exists(dest_path_qa_domain):
            os.mkdir(dest_path_qa_domain)
        
        if plotting:
            plt.savefig(os.path.join(dest_path_qa_domain, image_name_base + '_' + index + '_validation.png'))
            if not show_plots:
                plt.close()
        if rerun:
            if not os.path.exists(os.path.join(dest_path_qa_bad_domain, image_name_base + '_' + index + '_pred.png')):
                tif_save(settings, metrics, os.path.join(dest_path_qa_domain, image_name_base + '_' + index + '_subset_raw.tif'), (raw_image * 255).astype(np.uint8))
                tif_save(settings, metrics, os.path.join(dest_path_qa_domain, image_name_base + '_' + index + '_pred.tif'), (pred_image * 255).astype(np.uint8))
        else:
            imsave(os.path.join(dest_path_qa_domain, image_name_base + '_' + index + '_large_processed_raw.png'), (unprocessed_original_raw).astype(np.uint8))
            imsave(os.path.join(dest_path_qa_domain, image_name_base + '_' + index + '_raw.png'), (original_raw_gray * 255).astype(np.uint8))
            imsave(os.path.join(dest_path_qa_domain, image_name_base + '_' + index + '_raw_subset_highlight.png'), (original_raw_gray_patched).astype(np.uint8))
            imsave(os.path.join(dest_path_qa_domain, image_name_base + '_' + index + '_subset_raw.png'), (raw_image * 255).astype(np.uint8))
            imsave(os.path.join(dest_path_qa_domain, image_name_base + '_' + index + '_pred.png'), (pred_image * 255).astype(np.uint8))
            imsave(os.path.join(dest_path_qa_domain, image_name_base + '_' + index + '_front_only.png'), (np.clip(polyline_image, 0.0, 1.0) * 255).astype(np.uint8))
            imsave(os.path.join(dest_path_qa_domain, image_name_base + '_' + index + '_overlay_front.png'), (extracted_front * 255).astype(np.uint8))
            imsave(os.path.join(dest_path_qa_domain, image_name_base + '_' + index + '_overlay_comparison.png'), (overlay * 255).astype(np.uint8))
            tif_save(settings, metrics, os.path.join(dest_path_qa_domain, image_name_base + '_' + index + '_subset_raw.tif'), (raw_image * 255).astype(np.uint8))
            tif_save(settings, metrics, os.path.join(dest_path_qa_domain, image_name_base + '_' + index + '_pred.tif'), (pred_image * 255).astype(np.uint8))


def plot_production_results(settings, metrics):
    """Plots a standardized set of 6 plots for validation of the neural network, and quantifies its error per image."""
    empty_image = settings['empty_image']
    saving = settings['saving']
    plotting = settings['plotting']
    rerun = settings['rerun']
    show_plots = settings['show_plots']
    dest_path_qa = settings['dest_path_qa']
    image_settings = settings['image_settings']
    
    image_name_base = image_settings['image_name_base']
    bounding_box = image_settings['actual_bounding_box']
    raw_image = image_settings['raw_image']
    unprocessed_original_raw = image_settings['unprocessed_original_raw']
    original_raw = image_settings['original_raw']
    pred_image = image_settings['pred_image']
    polyline_image = image_settings['polyline_image_dilated']
    index = str(image_settings['i'] + 1) + '-' + str(image_settings['box_counter'])
    
    #Set figure size for 1600x900 resolution, tight layout
    plt.rcParams["figure.figsize"] = (16,9)
    
    #Create the color key for each subplots' legends    
    preprocess_legend = [Line2D([0], [0], color='#ff0000', lw=4),
                         Line2D([0], [0], color='#00ff00', lw=4),
                         Line2D([0], [0], color='#0000ff', lw=4)]
    nn_legend = [Line2D([0], [0], color='#00ff00', lw=4),
                 Line2D([0], [0], color='#ff0000', lw=4)]
    front_legend = [Line2D([0], [0], color='#ff0000', lw=4)]
#    polyline_image[:,:,0] = skeletonize(polyline_image[:,:,0])
    #Begin plotting the 2x3 validation results output
    original_raw_gray = np.clip(np.stack((original_raw[:,:,0] / 255.0, original_raw[:,:,0] / 255.0, original_raw[:,:,0] / 255.0), axis=-1), 0.0, 1.0)
#    raw_image_gray = np.stack((raw_image[:,:,0], raw_image[:,:,0], raw_image[:,:,0]), axis=-1)
    start_point = (int(bounding_box[1]), int(bounding_box[0]))
    end_point = (int(bounding_box[1] + bounding_box[3]), int(bounding_box[0] + bounding_box[2]))
    original_raw_gray_patched = cv2.rectangle(original_raw_gray * 255, start_point, end_point, (255, 0, 0), 1).astype(np.uint8)
    raw_image = np.clip(raw_image, 0.0, 1.0)
    pred_image = np.clip(pred_image, 0.0, 1.0)
    extracted_front = np.clip(np.stack((polyline_image[:,:,0], empty_image, empty_image), axis=-1) + raw_image * 0.8, 0.0, 1.0)
    
    if plotting:
        #Initialize plots
        f, axarr = plt.subplots(1, 4, num=index)
        f.suptitle(image_name_base, fontsize=18, weight='bold')
    
        axarr[0].imshow(original_raw_gray_patched)
        axarr[0].set_title(r'$\bf{a)}$ Raw Subset')
        
        axarr[1].imshow(raw_image)
        axarr[1].set_title(r'$\bf{b)}$ Preprocessed Input')
        axarr[1].legend(preprocess_legend, ['Raw', 'HDR', 'S/H'], prop={'weight': 'normal'}, facecolor='#eeeeee', loc='upper center', bbox_to_anchor=(0.5, 0.0), shadow=True, ncol=3)
        axarr[1].axis('off')
        
        axarr[2].imshow(pred_image)
        axarr[2].set_title(r'$\bf{c)}$ NN Output')
        axarr[2].legend(nn_legend, ['Land/Ice', 'Front'], prop={'weight': 'normal'}, facecolor='#eeeeee', loc='upper center', bbox_to_anchor=(0.5, 0.0), shadow=True, ncol=2)
        axarr[2].axis('off')
        
        axarr[3].imshow(extracted_front)
        axarr[3].set_title(r'$\bf{d)}$ Extracted Front')
        axarr[3].legend(front_legend, ['Front'], prop={'weight': 'normal'}, facecolor='#eeeeee', loc='upper center', bbox_to_anchor=(0.5, 0.0), shadow=True, ncol=1)
        axarr[3].axis('off')
        
        #Refresh plot if necessary
        plt.subplots_adjust(top = 0.90, bottom = 0.075, right = 0.97, left = 0.03, hspace = 0.3, wspace = 0.2)
        f.canvas.draw()
        f.canvas.flush_events()
    
    #Save figure
    if saving:
        domain = image_settings['domain']
        dest_path_qa_domain = os.path.join(dest_path_qa, domain)
        if not os.path.exists(dest_path_qa_domain):
            os.mkdir(dest_path_qa_domain)
        
        if plotting:
            plt.savefig(os.path.join(dest_path_qa_domain, image_name_base + '_' + index + '_results.png'))
            if not show_plots:
                plt.close()
        if rerun:
            if os.path.exists(os.path.join(dest_path_qa_domain, image_name_base + '_' + index + '_pred.png')):
                tif_save(settings, metrics, os.path.join(dest_path_qa_domain, image_name_base + '_' + index + '_subset_raw.tif'), (raw_image * 255).astype(np.uint8))
                tif_save(settings, metrics, os.path.join(dest_path_qa_domain, image_name_base + '_' + index + '_pred.tif'), (pred_image * 255).astype(np.uint8))
        else:
            imsave(os.path.join(dest_path_qa_domain, image_name_base + '_' + index + '_large_processed_raw.png'), (unprocessed_original_raw).astype(np.uint8))
            imsave(os.path.join(dest_path_qa_domain, image_name_base + '_' + index + '_subset_raw.png'), (raw_image * 255).astype(np.uint8))
            imsave(os.path.join(dest_path_qa_domain, image_name_base + '_' + index + '_pred.png'), (pred_image * 255).astype(np.uint8))
            imsave(os.path.join(dest_path_qa_domain, image_name_base + '_' + index + '_overlay_front.png'), (extracted_front * 255).astype(np.uint8))
            tif_save(settings, metrics, os.path.join(dest_path_qa_domain, image_name_base + '_' + index + '_subset_raw.tif'), (raw_image * 255).astype(np.uint8))
            tif_save(settings, metrics, os.path.join(dest_path_qa_domain, image_name_base + '_' + index + '_pred.tif'), (pred_image * 255).astype(np.uint8))


def plot_troubled_ones(settings, metrics):
    """Plots a standardized set of 6 plots for validation of the neural network, and quantifies its error per image."""
    saving = settings['saving']
    plotting = settings['plotting']
    rerun = settings['rerun']
    show_plots = settings['show_plots']
    dest_path_qa_bad = settings['dest_path_qa_bad']
    image_settings = settings['image_settings']
    
    image_name_base = image_settings['image_name_base']
    bounding_box = image_settings['actual_bounding_box']
    raw_image = image_settings['raw_image']
    unprocessed_original_raw = image_settings['unprocessed_original_raw']
    original_raw = image_settings['original_raw']
    pred_image = image_settings['pred_image']
    index = str(image_settings['i'] + 1) + '-' + str(image_settings['box_counter'])
    
    #Set figure size for 1600x900 resolution, tight layout
    plt.rcParams["figure.figsize"] = (16,4.5)
    
    #Create the color key for each subplots' legends    
    preprocess_legend = [Line2D([0], [0], color='#ff0000', lw=4),
                         Line2D([0], [0], color='#00ff00', lw=4),
                         Line2D([0], [0], color='#0000ff', lw=4)]
    nn_legend = [Line2D([0], [0], color='#00ff00', lw=4),
                 Line2D([0], [0], color='#ff0000', lw=4)]
#    polyline_image[:,:,0] = skeletonize(polyline_image[:,:,0])
    #Begin plotting the 2x3 validation results output
    original_raw_gray = np.clip(np.stack((original_raw[:,:,0] / 255.0, original_raw[:,:,0] / 255.0, original_raw[:,:,0] / 255.0), axis=-1), 0.0, 1.0)
#    raw_image_gray = np.stack((raw_image[:,:,0], raw_image[:,:,0], raw_image[:,:,0]), axis=-1)
    start_point = (int(bounding_box[1]), int(bounding_box[0]))
    end_point = (int(bounding_box[1] + bounding_box[3]), int(bounding_box[0] + bounding_box[2]))
    original_raw_gray_patched = cv2.rectangle(original_raw_gray * 255, start_point, end_point, (255, 0, 0), 1).astype(np.uint8)
    raw_image = np.clip(raw_image, 0.0, 1.0)
    pred_image = np.clip(pred_image, 0.0, 1.0)
    
    if plotting:
        #Initialize plots
        f, axarr = plt.subplots(1, 3, num=index)
        f.suptitle(image_name_base, fontsize=18, weight='bold')
    
        axarr[0].imshow(original_raw_gray_patched)
        axarr[0].set_title(r'$\bf{a)}$ Raw Subset')
        
        axarr[1].imshow(raw_image)
        axarr[1].set_title(r'$\bf{b)}$ Preprocessed Input')
        axarr[1].legend(preprocess_legend, ['Raw', 'HDR', 'S/H'], prop={'weight': 'normal'}, facecolor='#eeeeee', loc='upper center', bbox_to_anchor=(0.5, 0.0), shadow=True, ncol=3)
        axarr[1].axis('off')
        
        axarr[2].imshow(pred_image)
        axarr[2].set_title(r'$\bf{c)}$ NN Output')
        axarr[2].legend(nn_legend, ['Land/Ice', 'Front'], prop={'weight': 'normal'}, facecolor='#eeeeee', loc='upper center', bbox_to_anchor=(0.5, 0.0), shadow=True, ncol=2)
        axarr[2].axis('off')
        
        
        #Refresh plot if necessary
        plt.subplots_adjust(top = 0.85, bottom = 0.075, right = 0.97, left = 0.03, hspace = 0.3, wspace = 0.2)
        f.canvas.draw()
        f.canvas.flush_events()
    
    #Save figure
    if saving:
        domain = image_settings['domain']
        dest_path_qa_bad_domain = os.path.join(dest_path_qa_bad, domain)
        if not os.path.exists(dest_path_qa_bad_domain):
            os.mkdir(dest_path_qa_bad_domain)
        
        if plotting:
            plt.savefig(os.path.join(dest_path_qa_bad_domain, image_name_base + '_' + index + '_results.png'))
            if not show_plots:
                plt.close()
        if not rerun:
            imsave(os.path.join(dest_path_qa_bad_domain, image_name_base + '_' + index + '_large_processed_raw.png'), (unprocessed_original_raw).astype(np.uint8))
            imsave(os.path.join(dest_path_qa_bad_domain, image_name_base + '_' + index + '_subset_raw.png'), (raw_image * 255).astype(np.uint8))
            imsave(os.path.join(dest_path_qa_bad_domain, image_name_base + '_' + index + '_pred.png'), (pred_image * 255).astype(np.uint8))

def plot_histogram(distances, name, dest_path, saving, scaling):
    """Plots a standardized set of 6 plots for validation of the neural network, and quantifies its error per image."""
    #Initialize plots
    hist_bins = 20
    f, axarr = plt.subplots(1, 1, num=name)
    f.suptitle('Validation Set: Per-point Distance from True Front', fontsize=16)
    
    axarr.hist(distances, bins=hist_bins, range=[0.0, 20.0 * scaling])
    axarr.set_xlabel('Distance to nearest point (meters)')
    axarr.set_ylabel('Number of points')
    plt.figtext(0.5, 0.01, r'Mean Distance = {:.2f}m'.format(np.mean(distances)), wrap=True, horizontalalignment='center', fontsize=14, weight='bold')
        
    #Set figure size for 1600x900 resolution, tight layout
    plt.rcParams["figure.figsize"] = (8,4.5)
    plt.subplots_adjust(top = 0.90, bottom = 0.15, right = 0.90, left = 0.1, hspace = 0.25, wspace = 0.25)
    
    #Refresh plot if necessary
    f.canvas.draw()
    f.canvas.flush_events()
    
    #Save figure
    if saving:
        plt.savefig(os.path.join(dest_path, name + '.png'))


def plot_scatter(data, name, dest_path, saving):
    """Plots a standardized set of 6 plots for validation of the neural network, and quantifies its error per image."""
    #Initialize plots
    f, axarr = plt.subplots(1, 1, num=name)
    f.suptitle('Validation Set: Per-point Distance from True Front', fontsize=16)
    
    axarr.scatter(data[:,0], data[:,1])
    axarr.set_xlabel('Resolution (meters per pixel)')
    axarr.set_ylabel('Average Mean Distance')
        
    #Set figure size for 1600x900 resolution, tight layout
    plt.rcParams["figure.figsize"] = (8,4.5)
    plt.subplots_adjust(top = 0.90, bottom = 0.15, right = 0.90, left = 0.1, hspace = 0.25, wspace = 0.25)
    
    #Refresh plot if necessary
    f.canvas.draw()
    f.canvas.flush_events()
    
    #Save figure
    if saving:
        plt.savefig(os.path.join(dest_path, name + '.png'))
