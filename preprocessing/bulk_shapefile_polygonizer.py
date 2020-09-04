# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:38:07 2019

@author: Daniel
"""
import os, glob, sys
from collections import defaultdict
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import median_filter
from skimage import measure
import numpy as np
import cv2
import matplotlib.pyplot as plt

#Ensure crs are exported correctly by gdal/osr/fiona
os.environ['GDAL_DATA'] = r'D:\\ProgramData\\Anaconda3\\envs\\cfm\\Library\\share\\gdal'

import rasterio
from rasterio import features
from rasterio.windows import from_bounds
import fiona

sys.path.insert(1, '../postprocessing')
from mask_to_shp import mask_to_polygon_shp

#level 0 should inlcude all subsets (preprocessed)
#Make individual ones, domain ones, and all available
#indivudal ones include QA, tif, and shapefile

#have front line,
#to use smoother, must have full mask
#to get full mask, repeated polygonization using regular front line


def landsat_sort(file_path):
    """Sorting key function derives date from landsat file path.
    Also orders manual masks in front of auto masks."""
    file_name_parts = file_path.split(os.path.sep)[-1].split('_')
    if 'validation' in file_name_parts[-1]:
        return file_name_parts[3] + 'a'
    return file_name_parts[3] + 'b'


def window_from_extent(xmin, xmax, ymin, ymax, aff):
    """Calculates bounding box pixel window from coordinates and affine geotransform."""
    col_start, row_start = ~aff * (xmin, ymax)
    col_stop, row_stop = ~aff * (xmax, ymin)
    return ((int(row_start), int(row_stop)), (int(col_start), int(col_stop)))

def get_file_lists(file_list, bad_file_list):
    """Converts file lists into date indexed dictionaries."""
    dates = defaultdict(list)
    bad_dates = defaultdict(list)
    for file_path in bad_file_list:
        file_name = os.path.basename(file_path)
        file_name_parts = file_name.split('_')
        satellite = file_name_parts[1]
        if satellite.startswith('S'):
            #Astakhov-Chugunov-Astapenko_S1B_EW_GRDM_1SDH_2018-06-26_011542_01536C_EB6F
            date_dashed = file_name_parts[4]
        elif satellite.startswith('L'):
            #Brückner_LC08_L1TP_2015-06-14_232-014_T1_B5_66-1_validation
            date_dashed = file_name_parts[3]
        bad_dates[date_dashed].append(file_path)
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        file_name_parts = file_name.split('_')
        satellite = file_name_parts[1]
        if satellite.startswith('S'):
            #Astakhov-Chugunov-Astapenko_S1B_EW_GRDM_1SDH_2018-06-26_011542_01536C_EB6F
            date_dashed = file_name_parts[4]
        elif satellite.startswith('L'):
            #Brückner_LC08_L1TP_2015-06-14_232-014_T1_B5_66-1_validation
            date_dashed = file_name_parts[3]
        if date_dashed not in bad_dates:
            dates[date_dashed].append(file_path)
        else:    
            #Skip dates where not all fronts are good
            print('bad date:', date_dashed)
    return dates


def consolidate_shapefiles(source_path_manual, source_path_auto, fjord_boundary_path):
    """Consolidates closed polygonal Shapefiles into a single file with features."""
    source_manual_qa_path = os.path.join(source_path_manual, 'quality_assurance')
    source_auto_qa_path = os.path.join(source_path_auto, 'quality_assurance')
    
    counter = 0
    domains = ['Qeqertarsuup', 'Kakiffaat', 'Nunatakavsaup', 'Alangorssup', 'Akullikassaap',
               'Upernavik-NE', 'Upernavik-NW', 'Upernavik-SE' 'Sermikassak-N', 'Sermikassak-S',
               'Inngia', 'Umiammakku', 'Rink-Isbrae', 'Kangerlussuup', 'Kangerdluarssup',
               'Perlerfiup', 'Sermeq-Silarleq', 'Kangilleq', 'Sermilik', 'Lille', 'Store']
    domains = ['Qeqertarsuup', 'Kakiffaat', 'Nunatakavsaup', 'Alangorssup', 'Upernavik-NW',
               'Sermikassak-N', 'Sermikassak-S', 'Kangerlussuup', 'Kangerdluarssup',
               'Perlerfiup', 'Sermeq-Silarleq', 'Kangilleq', 'Sermilik', 'Lille', 'Store']
    domains = ['Upernavik-NE']
    domains = ['Upernavik-NE']
    for domain in os.listdir(source_manual_qa_path):
        if '.' in domain:
            continue
        if domain not in domains:
            continue
        file_list_manual = glob.glob(os.path.join(source_manual_qa_path, domain, '*_pred.tif'))
        file_list_auto = glob.glob(os.path.join(source_auto_qa_path, domain, '*_pred.tif'))
        file_list = file_list_manual + file_list_auto
        file_list.sort(key=landsat_sort)
        
        bad_file_list_manual = glob.glob(os.path.join(source_manual_qa_path + '_bad', domain, '*_pred.tif'))
        bad_file_list_auto = glob.glob(os.path.join(source_auto_qa_path + '_bad', domain, '*_pred.tif'))
        bad_file_list_manual_pruned = glob.glob(os.path.join(source_manual_qa_path + '_bad', domain, '*_overlay_front.png'))
        bad_file_list_auto_pruned = glob.glob(os.path.join(source_auto_qa_path + '_bad', domain, '*_overlay_front.png'))
        bad_file_list = bad_file_list_manual + bad_file_list_auto + bad_file_list_manual_pruned + bad_file_list_auto_pruned
        bad_file_list.sort(key=landsat_sort)
                
        fjord_boundary_file_path = os.path.join(fjord_boundary_path, domain + '_fjord_boundaries.tif')
        dates_exits = dict()
        dates_lambdas = dict()
        with rasterio.open(fjord_boundary_file_path) as fjord_boundary_tif:
            original_mask = fjord_boundary_tif.read(1)
            fjord_bounds = rasterio.transform.array_bounds(original_mask.shape[0], original_mask.shape[1], fjord_boundary_tif.transform)
            dates = get_file_lists(file_list, bad_file_list)
            #For each date, retrieve mask by projecting each calving front onto a single mask,
            #count "exits" to handle multi-
            for date_key in dates.keys():
                result = process_date(counter, domain, dates, date_key, original_mask, fjord_bounds, fjord_boundary_tif, fjord_boundary_file_path)
                if result is not None:
                    counter = result[0]
                    dates_exits[date_key] = result[1]
                    dates_lambdas[date_key] = result[2]
                
            median_exits = np.median(dates_exits.values())
            print(median_exits)
            plt.rcParams["figure.figsize"] = (16,9)
            f, axarr = plt.subplots(1, 1, num=domain)
            axarr[0].hist(dates_exits.values(), bins=3)
            plt.show()

            #If all available fronts are masked, proceed with Shapefile output.
            # for date_key in dates.keys():
            #     if dates_exits[date_key] <= median_exits:
            #         dates_lambdas[date_key]()
                
def process_date(counter, domain, dates, date_key, original_mask, fjord_bounds, fjord_boundary_tif, fjord_boundary_file_path):
    """Processes a single day of calving front(s), and save it to a single polygon Shapefile."""
    # if date_key >= '1991-08-06':
    #       continue
    mask = np.empty_like(original_mask)
    np.copyto(mask, original_mask)
    combined_pred_mask = 255 - (mask / mask.max() * 255)
    combined_pred_mask_validity = 1 - mask / mask.max()
    for file_path in dates[date_key]:
        counter, mask, combined_pred_mask, combined_pred_mask_validity = process_file(combined_pred_mask, combined_pred_mask_validity, counter, domain, file_path, fjord_bounds, fjord_boundary_tif, mask, original_mask)
    
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    sizes = stats[:, cv2.CC_STAT_AREA]
    ordering = list(reversed(np.argsort(sizes)))

    #for every component in the image, keep it only if it's above min_size
    min_size_dynamic = 0
    if len(sizes) > 1:
        min_size_dynamic = sizes[ordering[0]] * 0.01
    min_size_percentage = 0.025
    min_size_floor = max(output.size * min_size_percentage, min_size_dynamic)
    
    new_mask = np.ones(mask.shape) * 255
    
    plt.figure(str(counter) + '-combined_pred_mask_validity')
    plt.imshow(combined_pred_mask_validity)
    plt.figure(str(counter) + '-combined_pred_mask')
    plt.imshow(combined_pred_mask)
    for i in range(len(sizes)):
        if sizes[ordering[i]] >= min_size_floor:
            mask_indices = output == ordering[i]
            valid_indices = combined_pred_mask_validity == 1
            valid_mask_indices = np.logical_and(mask_indices, valid_indices)
            image_component = combined_pred_mask[valid_mask_indices] #get green channel
            mean_value = np.mean(image_component)
            # mean_values.append(mean_value)
            
            image_component_mask = np.zeros(mask.shape)
            image_component_mask[valid_mask_indices] = 1.0
            plt.figure(str(counter) + '-image_component_mask-' + str(i))
            plt.imshow(image_component_mask)
            plt.figure(str(counter) + '-image_component-' + str(i))
            plt.imshow(combined_pred_mask * image_component_mask)
            
            if mean_value < 128:
                new_mask[mask_indices] = 0
    plt.figure(date_key + '-' + str(counter) + '-Final')
    plt.imshow(new_mask)
    plt.show()
    exit()

    # replace all values in land/ice mask with correct color
    new_mask = median_filter(new_mask, size=3)
    new_mask_padded = np.pad(new_mask, 1, constant_values=255)
    contours = measure.find_contours(new_mask_padded, 127)              
    
    if len(contours) < 1:
        print('no contours')
        plt.show()
        # exit()
        return
    file_name = os.path.basename(file_path)
    file_name_parts = file_name.split('_')
    file_basename = "_".join(file_name_parts[0:-2])
    image_name_base = file_basename
    front_line = np.transpose(contours[0]) - 1
    
    source_tif_path = fjord_boundary_file_path
    if 'production' in file_path:
        dest_root_path = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\production_staging'
    elif 'mask_extractor' in file_path:
        dest_root_path = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\mask_extractor'
                
    #count exits, and return function ready to generate the shp if the number of exits is <= the median # of  exits for the domain
    exits = count_exits(new_mask)
    call_mask_to_polygon_shp = lambda: mask_to_polygon_shp(image_name_base, front_line, source_tif_path, dest_root_path, domain)
    return counter, exits, call_mask_to_polygon_shp

def process_file(combined_pred_mask, combined_pred_mask_validity, counter, domain, file_path, fjord_bounds, fjord_boundary_tif, mask, original_mask):
    """Processes each file of possibly many within a single date. 
        Returns the validity mask and polyline projected onto the land-ice/ocean mask."""
    print(file_path)
    file_name = os.path.basename(file_path)
    file_name_parts = file_name.split('_')
    file_basename = "_".join(file_name_parts[0:-2])

    if 'production' in file_path:
        source_domain_path = os.path.join(source_path_auto, 'domain')
        source_pred_path = os.path.join(source_path_auto, 'quality_assurance', domain)
    elif 'mask_extractor' in file_path:
        source_domain_path = os.path.join(source_path_manual, 'domain')
        source_pred_path = os.path.join(source_path_manual, 'quality_assurance', domain)
    
    reprocessing_id = file_name_parts[-2][-1]
    old_file_shp_name = file_basename + '_' + reprocessing_id + '_cf.shp'
    old_file_shp_file_path = os.path.join(source_domain_path, domain, old_file_shp_name)
    pred_img_name = file_basename + '_' + file_name_parts[-2] + '_pred.tif'
    pred_img_path = os.path.join(source_pred_path, pred_img_name)
    with fiona.open(old_file_shp_file_path, 'r', encoding='utf-8') as source_shp:
        with rasterio.open(pred_img_path) as pred_tif:
            # Read croped array
            x_min_fjord = fjord_bounds[0]
            y_min_fjord = fjord_bounds[1]
            x_max_fjord = fjord_bounds[2]
            y_max_fjord = fjord_bounds[3]
            window = from_bounds(x_min_fjord, y_min_fjord, x_max_fjord, y_max_fjord, pred_tif.transform)
            pred_mask = pred_tif.read(2, window=window, out_shape=mask.shape, boundless=True, fill_value=0)
            original_pred_mask = pred_tif.read(1)
            
            pred_bounds = rasterio.transform.array_bounds(original_pred_mask.shape[0], original_pred_mask.shape[1], pred_tif.transform)
            x_min_pred = pred_bounds[0]
            y_min_pred = pred_bounds[1]
            x_max_pred = pred_bounds[2]
            y_max_pred = pred_bounds[3]
            
            x_range = fjord_bounds[2] - fjord_bounds[0]
            y_range = fjord_bounds[3] - fjord_bounds[1]
            sub_x1 = int(round(max((x_min_pred - fjord_bounds[0]) / x_range * original_mask.shape[1], 0)))
            sub_x2 = int(round(min((x_max_pred - fjord_bounds[0]) / x_range * original_mask.shape[1], original_mask.shape[1])))
            sub_y1 = int(round(max(original_mask.shape[0] - (y_max_pred - fjord_bounds[1]) / y_range * original_mask.shape[0], 0)))
            sub_y2 = int(round(min(original_mask.shape[0] - (y_min_pred - fjord_bounds[1]) / y_range * original_mask.shape[0], original_mask.shape[0])))
                                        
            # print(actual_bounding_box)
            # print(pred_bounds, fjord_bounds)
            validity = np.zeros(combined_pred_mask_validity.shape)
            validity[sub_y1:sub_y2, sub_x1:sub_x2] = 1
            combined_pred_mask = np.maximum(combined_pred_mask, pred_mask)
            combined_pred_mask_validity = np.maximum(combined_pred_mask_validity, validity)
            
            # plt.close('all')
            # bounding_boxes_pred = results_pred[1]
            
            plt.figure(str(counter) + '-fjord_boundary_original')
            plt.imshow(mask)
            
            plt.figure(str(counter) + '-fjord_boundary')
            plt.imshow(original_mask)
                                        
            plt.figure(str(counter) + '-pred_mask')
            plt.imshow(pred_mask)
            
            plt.figure(str(counter) + '-original_pred_mask')
            plt.imshow(original_pred_mask)
            
            plt.figure(str(counter) + '-validity')
            plt.imshow(validity)
            
            plt.figure(str(counter) + '-combined_pred_mask_validity')
            plt.imshow(combined_pred_mask_validity)
            
            plt.figure(str(counter) + '-combined_pred_mask')
            plt.imshow(combined_pred_mask)
            plt.show()
            # exit()
            
            polyline_coords = source_shp[0]['geometry']['coordinates']
            fjord_distances, fjord_indices = distance_transform_edt(original_mask, return_indices=True)
            endpoint_pixel_coord_1 = rasterio.transform.rowcol(fjord_boundary_tif.transform, polyline_coords[0][0], polyline_coords[0][1])
            endpoint_pixel_coord_2 = rasterio.transform.rowcol(fjord_boundary_tif.transform, polyline_coords[-1][0], polyline_coords[-1][1])
            closest_pixel_1 = [fjord_indices[0][endpoint_pixel_coord_1], fjord_indices[1][endpoint_pixel_coord_1]]
            closest_pixel_2 = [fjord_indices[0][endpoint_pixel_coord_2], fjord_indices[1][endpoint_pixel_coord_2]]
            closest_coord_1 = rasterio.transform.xy(fjord_boundary_tif.transform, closest_pixel_1[0], closest_pixel_1[1])
            closest_coord_2 = rasterio.transform.xy(fjord_boundary_tif.transform, closest_pixel_2[0], closest_pixel_2[1])
            
            geometry = {'type': 'LineString'}
            geometry['coordinates'] = [closest_coord_1] + polyline_coords + [closest_coord_2]
            # geometry['coordinates'] = polyline_coords
            # mask = np.ones(mask.shape)
            counter += 1
            mask = features.rasterize([(geometry, 0)],
                # all_touched=True,
                out=mask,
                out_shape=fjord_boundary_tif.shape,
                transform=fjord_boundary_tif.transform)
            plt.figure(str(counter) + '-mask')
            plt.imshow(mask)
            # plt.show()
            # exit()
            return counter, mask, combined_pred_mask, combined_pred_mask_validity

def count_exits(image):
    """Given an image, detects the number of exits (mask boundary changes / 2) in a land-ice/ocean mask."""
    top_row = image[0, :]
    right_col = image[1:-1, -1]
    bottom_row = reversed(image[-1, :])
    left_col = reversed(image[0:-1, 0]) #add top left pixel again to "loop" the border and detect the edge case there
    border = top_row + right_col + bottom_row + left_col
    exits = np.sum(np.abs(np.diff(border))) / 2
    return exits
    
if __name__ == "__main__":
    #Initialize plots
    plt.close('all')
    font = {'family' : 'normal',
            'size'   : 14}
    plt.rc('font', **font)
    
    source_path_manual = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\mask_extractor'
    source_path_auto = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\production_staging'
    fjord_boundary_path = r'D:\Daniel\Documents\Github\CALFIN Repo\training\data\fjord_boundaries_tif'
    
    consolidate_shapefiles(source_path_manual, source_path_auto, fjord_boundary_path)
