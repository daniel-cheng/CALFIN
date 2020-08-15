# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:38:07 2019

@author: Daniel
"""
import numpy as np
import cv2
import os, shutil, glob, copy, sys
os.environ['GDAL_DATA'] = r'D:\\ProgramData\\Anaconda3\\envs\\cfm\\Library\\share\\gdal' #Ensure crs are exported correctly by gdal/osr/fiona

import rasterio
from rasterio import features
from rasterio.windows import from_bounds
from scipy.ndimage.morphology import distance_transform_edt
from scipy.spatial import KDTree
from scipy.ndimage import median_filter
from skimage.io import imsave, imread
from skimage import measure
from skimage.transform import resize
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from pyproj import Proj, transform
from shapely.geometry import mapping, Polygon, LineString
from collections import defaultdict
import fiona
from fiona.crs import from_epsg
from dateutil.parser import parse

sys.path.insert(1, '../postprocessing')
from error_analysis import extract_front_indicators
from mask_to_shp import mask_to_polygon_shp, mask_to_polyline
from ordered_line_from_unordered_points import is_outlier

#level 0 should inlcude all subsets (preprocessed)
#Make individual ones, domain ones, and all available
#indivudal ones include QA, tif, and shapefile

#have front line,
#to use smoother, must have full mask
#to get full mask, repeated polygonization using regular front line


def landsat_sort(file_path):
    """Sorting key function derives date from landsat file path. Also orders manual masks in front of auto masks."""
    file_name_parts = file_path.split(os.path.sep)[-1].split('_')
    if 'validation' in file_name_parts[-1]:
        return file_name_parts[3] + 'a'
    else:
        return file_name_parts[3] + 'b'

def duplicate_prefix_filter(file_list):
    caches = set()
    results = []
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        file_name_parts = file_name.split('_')
        prefix = "_".join(file_name_parts[0:-2])
        # check whether prefix already exists
        if prefix not in caches:
            results.append(file_path)
            caches.add(prefix)
        else:
            print('override:', prefix)
    return results

def window_from_extent(xmin, xmax, ymin, ymax, aff):
    col_start, row_start = ~aff * (xmin, ymax)
    col_stop, row_stop = ~aff * (xmax, ymin)
    return ((int(row_start), int(row_stop)), (int(col_start), int(col_stop)))


def consolidate_shapefiles(source_path_manual, source_path_auto, dest_domain_path, dest_all_path, version, fjord_boundary_path, pred_path, settings):
    schema = {
        'geometry': 'LineString',
        'properties': {
            'GlacierID': 'int',
            'Center_X': 'float',
            'Center_Y': 'float',
            'Sequence#': 'int',
            'QualFlag': 'int',
            'Satellite': 'str',
            'Date': 'str',
            'Year': 'int',
            'Month': 'int',
            'Day': 'int',
            'ImageID': 'str',
            'GrnlndcN': 'str',
            'OfficialN': 'str',
            'AltName': 'str',
            'RefName': 'str',
            'Author': 'str'
        },
    }
#     plt.close('all')
    outProj = Proj('epsg:3413') #3413 (NSIDC Polar Stereographic North)
    crs = from_epsg(3413)
    source_manual_quality_assurance_path = os.path.join(source_path_manual, 'quality_assurance')
    source_auto_quality_assurance_path = os.path.join(source_path_auto, 'quality_assurance')
    output_all_shp_path = os.path.join(dest_all_path, 'termini_1972-2019_calfin_' + version + '.shp')
#     with fiona.open(output_all_shp_path,
#         'w',
#         driver='ESRI Shapefile',
#         crs=fiona.crs.from_epsg(3413),
#         schema=schema,
#         encoding='utf-8') as output_all_shp_file:
    counter = 0
    for domain in os.listdir(source_manual_quality_assurance_path):
        if '.' in domain:
            continue
#         if not('Upernavik' in domain or 'Rink-Isbrae' in domain or 'Inngia' in domain or 'Umiammakku' in domain):
#             continue
        # if 'Petermann' not in domain:
            # continue
        if 'Upernavik-NE' not in domain:
            continue
        file_list_manual = glob.glob(os.path.join(source_manual_quality_assurance_path, domain, '*_pred.tif'))
        file_list_auto = glob.glob(os.path.join(source_auto_quality_assurance_path, domain, '*_pred.tif'))
        file_list = file_list_manual + file_list_auto
        file_list.sort(key=landsat_sort)
                
        bad_file_list_manual = glob.glob(os.path.join(source_manual_quality_assurance_path + '_bad', domain, '*_pred.tif'))
        bad_file_list_auto = glob.glob(os.path.join(source_auto_quality_assurance_path + '_bad', domain, '*_pred.tif'))
        bad_file_list = bad_file_list_manual + bad_file_list_auto
        bad_file_list.sort(key=landsat_sort)
                
        fjord_boundary_file_path = os.path.join(fjord_boundary_path, domain + '_fjord_boundaries.tif')
        with rasterio.open(fjord_boundary_file_path) as fjord_boundary_tif:
            original_mask = fjord_boundary_tif.read(1)
            fjord_bounds = rasterio.transform.array_bounds(original_mask.shape[0], original_mask.shape[1], fjord_boundary_tif.transform)
            x_min_fjord = fjord_bounds[0]
            y_min_fjord = fjord_bounds[1]
            x_max_fjord = fjord_bounds[2]
            y_max_fjord = fjord_bounds[3]
            combined_pred_mask = np.zeros(original_mask.shape)
            combined_pred_mask_validity = np.zeros(original_mask.shape)
            dates = defaultdict(list)
            for file_path in file_list:
                file_name = os.path.basename(file_path)
                file_name_parts = file_name.split('_')
                file_basename = "_".join(file_name_parts[0:-2])
                satellite = file_name_parts[1]
                if satellite.startswith('S'):
                    #Astakhov-Chugunov-Astapenko_S1B_EW_GRDM_1SDH_2018-06-26_011542_01536C_EB6F
                    date_dashed = file_name_parts[4]
                    dates[date_dashed].append(file_path)
                elif satellite.startswith('L'):
                    #Brückner_LC08_L1TP_2015-06-14_232-014_T1_B5_66-1_validation
                    date_dashed = file_name_parts[3]
                    dates[date_dashed].append(file_path)
            bad_dates = defaultdict(list)
            for file_path in bad_file_list:
                file_name = os.path.basename(file_path)
                file_name_parts = file_name.split('_')
                file_basename = "_".join(file_name_parts[0:-2])
                satellite = file_name_parts[1]
                if satellite.startswith('S'):
                    #Astakhov-Chugunov-Astapenko_S1B_EW_GRDM_1SDH_2018-06-26_011542_01536C_EB6F
                    date_dashed = file_name_parts[4]
                    bad_dates[date_dashed].append(file_path)
                elif satellite.startswith('L'):
                    #Brückner_LC08_L1TP_2015-06-14_232-014_T1_B5_66-1_validation
                    date_dashed = file_name_parts[3]
                    bad_dates[date_dashed].append(file_path)
            
            
            for date_key in dates.keys():
                #Skip dates where not all fronts are good
                if date_key in bad_dates:
#                     print('bad date:', date_key)
                    continue
#                 if '2017-04-16' >= date_key:
#                      continue
                
                mask = np.empty_like(original_mask)
                np.copyto(mask, original_mask)
                
                for file_path in dates[date_key]:

                    print(file_path)
                    file_name = os.path.basename(file_path)
                    file_name_parts = file_name.split('_')
                    file_basename = "_".join(file_name_parts[0:-2])
                    satellite = file_name_parts[1]
                    if satellite.startswith('S'):
                        #Astakhov-Chugunov-Astapenko_S1B_EW_GRDM_1SDH_2018-06-26_011542_01536C_EB6F
                        datatype = file_name_parts[2]
                        level = file_name_parts[3]
                        date_dashed = file_name_parts[4]
                        date_parts = date_dashed.split('-')
                        year = date_parts[0]
                        month = date_parts[1]
                        day = date_parts[2]
                        date = date_dashed.replace('-', '')
                        orbit = file_name_parts[5]
                        bandpol = 'hh'
                    elif satellite.startswith('L'):
                        #Brückner_LC08_L1TP_2015-06-14_232-014_T1_B5_66-1_validation
                        datatype = file_name_parts[2]
                        date_dashed = file_name_parts[3]
                        date_parts = date_dashed.split('-')
                        year = date_parts[0]
                        month = date_parts[1]
                        day = date_parts[2]
                        date = date_dashed.replace('-', '')
                        orbit = file_name_parts[4].replace('-', '')
                        level = file_name_parts[5]
                        bandpol = file_name_parts[6]
                        scene_id = landsat_scene_id_lookup(date, orbit, satellite, level)
                    else:
                        print('Unrecognized sattlelite!')
                        continue
    
                    if 'production' in file_path:
                        source_domain_path = os.path.join(source_path_auto, 'domain')
                        source_pred_path = os.path.join(source_path_auto, 'quality_assurance', domain)
                    elif 'mask_extractor' in file_path:
                        source_domain_path = os.path.join(source_path_manual, 'domain')
                        source_pred_path = os.path.join(source_path_manual, 'quality_assurance', domain)
    
        #                if not landsat_output_lookup(domain, date, orbit, satellite, level):
        #                    print('duplicate pick, continuing:', date, orbit, satellite, level, domain)
        #                    continue
                    reprocessing_id = file_name_parts[-2][-1]
                    old_file_shp_name = file_basename + '_' + reprocessing_id + '_cf.shp'
                    old_file_shp_file_path = os.path.join(source_domain_path, domain, old_file_shp_name)
                    # with fiona.open(old_file_shp_file_path, 'r', encoding='utf-8') as source_shp:
                    pred_img_name = file_basename + '_' + file_name_parts[-2] + '_pred.tif'
                    # front_img_name = file_basename +'_' + file_name_parts[-2] + '_front_only.png'
                    pred_img_path = os.path.join(source_pred_path, pred_img_name)
                    # front_img_path = os.path.join(source_pred_path, front_img_name)
                    with fiona.open(old_file_shp_file_path, 'r', encoding='utf-8') as source_shp:
#                         pred_img_name = file_basename +'_' + file_name_parts[-2] + '_pred.tif'
#                         pred_img_path = os.path.join(source_pred_path, pred_img_name)
#                         with rasterio.open(pred_img_path) as pred_tif:
#                             # Read croped array
#                             window=from_bounds(xMin, yMin, xMax, yMax, pred_tif.transform)
#                             pred_mask = pred_tif.read(2, window=window, out_shape=mask.shape, boundless=True, fill_value=127)
#     
#                             fjord_distances, fjord_indices = distance_transform_edt(original_mask, return_indices=True)
#                             polyline_coords = source_shp[0]['geometry']['coordinates']
                        with rasterio.open(pred_img_path) as pred_tif:
                            # Read croped array
                            window=from_bounds(x_min_fjord, y_min_fjord, x_max_fjord, y_max_fjord, pred_tif.transform)
                            pred_mask = pred_tif.read(2, window=window, out_shape=mask.shape, boundless=True, fill_value=0)
                            original_pred_mask = pred_tif.read(1)
                            
                            pred_bounds = rasterio.transform.array_bounds(original_pred_mask.shape[0], original_pred_mask.shape[1], pred_tif.transform)
                            x_min_pred = pred_bounds[0]
                            y_min_pred = pred_bounds[1]
                            x_max_pred = pred_bounds[2]
                            y_max_pred = pred_bounds[3]
                            fjord_window=from_bounds(x_min_pred, y_min_pred, x_max_pred, y_max_pred, fjord_boundary_tif.transform)
                            
                            x_range = fjord_bounds[2] - fjord_bounds[0]
                            y_range = fjord_bounds[3] - fjord_bounds[1]
                            sub_x1 = int(round(max((x_min_pred - fjord_bounds[0]) / x_range * original_mask.shape[1], 0)))
                            sub_x2 = int(round(min((x_max_pred - fjord_bounds[0]) / x_range * original_mask.shape[1], original_mask.shape[1])))
                            sub_y1 = int(round(max(original_mask.shape[0] - (y_max_pred - fjord_bounds[1]) / y_range * original_mask.shape[0], 0)))
                            sub_y2 = int(round(min(original_mask.shape[0] - (y_min_pred - fjord_bounds[1]) / y_range * original_mask.shape[0], original_mask.shape[0])))
                            #sub_x1 = pixel startx in original image
                            #actual_bounding_box[0] = pixel starx in full_size image
                            x_scale = original_mask.shape[1] / original_mask.shape[1]
                            y_scale = original_mask.shape[0] / original_mask.shape[0]
                            actual_bounding_box = [sub_x1 * x_scale, 
                                                   sub_y1 * y_scale, 
                                                   (sub_x2 - sub_x1) * x_scale, 
                                                   (sub_y2 - sub_y1) * y_scale]
                            # print(actual_bounding_box)
                            # print(pred_bounds, fjord_bounds)
                            validity = np.zeros(combined_pred_mask_validity.shape)
                            validity[ sub_y1:sub_y2, sub_x1:sub_x2] = 1
                            combined_pred_mask = np.maximum(combined_pred_mask, pred_mask)
                            combined_pred_mask_validity = np.maximum(combined_pred_mask_validity, validity)
                            
                            
                            #Redo masking
                            # empty_image = np.zeros(original_pred_mask.shape)
                            # results_pred = mask_polyline(original_pred_mask, fjord_boundary, settings, min_size_percentage=0.0005, use_extracted_front=False)
                            # polyline_image = np.stack((results_pred[0], empty_image, empty_image), axis=-1)
                            # plt.close('all')
                            # bounding_boxes_pred = results_pred[1]
                            
                            # plt.figure(str(counter) + '-fjord_boundary_original')
                            # plt.imshow(mask)
                            
                            # plt.figure(str(counter) + '-fjord_boundary')
                            # plt.imshow(fjord_boundary)
                            
                            # plt.figure(str(counter) + '-polyline_image_initial')
                            # plt.imshow(polyline_image)
                            
                            # plt.figure(str(counter) + '-pred_mask')
                            # plt.imshow(pred_mask)
                            
                            # plt.figure(str(counter) + '-original_pred_mask')
                            # plt.imshow(original_pred_mask)
                            
                            # plt.figure(str(counter) + '-validity')
                            # plt.imshow(validity)
                            
                            # plt.figure(str(counter) + '-combined_pred_mask_validity')
                            # plt.imshow(combined_pred_mask_validity)
                            
                            # plt.figure(str(counter) + '-combined_pred_mask')
                            # plt.imshow(combined_pred_mask)
                            # plt.show()
                            # exit()
                            #mask out largest front
    #                         if len(bounding_boxes_pred) < 2:
    #                             print("no front detected, skipping")
    # #                                 metrics['no_detection_skip_count'] += 1
    # #                                 return found_front, metrics
    #                             continue
    #                         image_settings = dict()
    #                         # image_settings['box_counter'] = box_counter
    #                         image_settings['used_bounding_boxes'] = []
    #                         image_settings['actual_bounding_box'] = actual_bounding_box
    #                         settings['image_settings'] = image_settings
    #                         polyline_image, bounding_box_pred = mask_bounding_box(bounding_boxes_pred, polyline_image, settings)
                            
                            # plt.figure(str(counter) + '-polyline_image_initial_masked')
                            # plt.imshow(polyline_image)
                            # exit()
                            # results_polyline = extract_front_indicators(polyline_image[:,:,0], z_score_cutoff=25.0)
                            # polyline_image = np.stack((results_polyline[0][:,:,0] / 255.0, empty_image, empty_image), axis=-1)
                            
                            
                            # polyline_image = results_polyline[0][:,:,0]
                            # plt.figure(str(counter) + '-polyline_image')
                            # plt.imshow(polyline_image)
                            # plt.figure(str(counter) + '-front_mask')
                            # plt.imshow(front_mask)
                            
                            # polyline_coords = mask_to_polyline(pred_img_path, polyline_image, original_pred_mask, fjord_boundary)
                            polyline_coords = source_shp[0]['geometry']['coordinates']
                            fjord_distances, fjord_indices = distance_transform_edt(original_mask, return_indices=True)
                            endpoint_pixel_coord_1 = rasterio.transform.rowcol(fjord_boundary_tif.transform, polyline_coords[0][0], polyline_coords[0][1])
                            endpoint_pixel_coord_2 = rasterio.transform.rowcol(fjord_boundary_tif.transform, polyline_coords[-1][0], polyline_coords[-1][1])
                            closest_pixel_1 = [fjord_indices[0][endpoint_pixel_coord_1], fjord_indices[1][endpoint_pixel_coord_1]]
                            closest_pixel_2 = [fjord_indices[0][endpoint_pixel_coord_2], fjord_indices[1][endpoint_pixel_coord_2]]
                            closest_coord_1 = rasterio.transform.xy(fjord_boundary_tif.transform, closest_pixel_1[0], closest_pixel_1[1])
                            closest_coord_2 = rasterio.transform.xy(fjord_boundary_tif.transform, closest_pixel_2[0], closest_pixel_2[1])
                            
                            #output interp, save uninterped
                            # polyline_coords = source_shp[0]['geometry']['coordinates']
                            geometry = {'type': 'LineString'}
                            geometry['coordinates'] = [closest_coord_1] + polyline_coords + [closest_coord_2]
                            
                            #polyline_to_shp()
                            mask = features.rasterize([(geometry, 0)],
                                #                         all_touched=True,
                                out=mask,
                                out_shape=fjord_boundary_tif.shape,
                                transform=fjord_boundary_tif.transform)
                            counter += 1
                #loop through others of same date as long as no dates are in qa_bad
                #use connected components

                # sample random number of pixels in dindices within each component to determine mean value
                nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)

                
                #connectedComponentswithStats yields every seperated component with information on each of them, such as size
                sizes = stats[:,cv2.CC_STAT_AREA]
                ordering = list(reversed(np.argsort(sizes)))

                #for every component in the image, keep it only if it's above min_size
#                     min_size_floor = output.size * min_size_percentage
                min_size_dynamic = 0
                if len(sizes) > 1:
                    min_size_dynamic = sizes[ordering[0]] * 0.01
                
                min_size_percentage = 0.025
                min_size_floor = max(output.size * min_size_percentage, min_size_dynamic)
                mean_values = []
                
                new_mask = np.ones(mask.shape) * 255
                window=from_bounds(x_min_fjord, y_min_fjord, x_max_fjord, y_max_fjord, fjord_boundary_tif.transform)
                
                # plt.figure(str(counter) + '-combined_pred_mask_validity')
                # plt.imshow(combined_pred_mask_validity)
                # plt.figure(str(counter) + '-combined_pred_mask')
                # plt.imshow(combined_pred_mask)
                for i in range(len(sizes)):
                    if sizes[ordering[i]] >= min_size_floor:
                        mask_indices = output == ordering[i]
                        image_component = combined_pred_mask[mask_indices] #get green channel
                        mean_value = np.mean(image_component)
                        mean_values.append(mean_value)
                        # print(mean_values)
                        if mean_value < 128:
                            new_mask[mask_indices] = 0
                # plt.figure(str(counter) + '-mask')
                # plt.imshow(mask)
                # print(mean_values)
                plt.figure(date_key + '-' + str(counter) + '-Final')
                plt.imshow(new_mask)


                # replace all values in land/ice mask with correct color
                new_mask = median_filter(new_mask, size=3)
                new_mask_padded = np.pad(new_mask, 1, constant_values=255)
                contours = measure.find_contours(new_mask_padded, 127)
#                 polyline_coords = mask_to_polyline(pred_img_path, polyline_image, original_pred_mask, zoomed_fjord_boundary_mask)
                            
                
    
                # mask_edge = cv2.Canny(new_mask_padded.astype(np.uint8), 250, 255 * 2) #thresholds = Use diagonals to detect strong edges, then connect anything with at least a single edge
#                 results_polyline = extract_front_indicators(mask_edge, z_score_cutoff=25.0)
#                 empty_image = np.zeros(mask_edge.shape)
#                 if results_polyline is None:
#                     print('nothing for', file_path)
# #                     plt.show()
#                     continue
#                 polyline_image = np.stack((results_polyline[0][:,:,0] / 255.0, empty_image, empty_image), axis=-1)
                # plt.figure(str(counter) + '-Polyline3')
                # plt.imshow(new_mask_padded)
                # plt.figure(str(counter) + '-mask_edge')
                # plt.imshow(mask_edge)
                # fig, ax = plt.subplots()
                # ax.title.set_text(str(counter) + '-Contour3')
#                 ax.imshow(new_mask)
#                 front_line = results_polyline[1] - 1
#                 ax.plot(front_line[:, 0], front_line[:, 1], linewidth=1, color='r')
#                 ax.plot(results_polyline[1][:, 0], results_polyline[1][:, 0], linewidth=1, color='b')
                # for n, contour in enumerate(contours):
                #     contour = contour - 1
                #     ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='g')
                plt.show()
                
                image_name_base = file_basename
                front_line = np.transpose(contours[0]) - 1
                source_tif_path = fjord_boundary_file_path
                if 'production' in file_path:
                    dest_root_path = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\production_staging'
                elif 'mask_extractor' in file_path:
                    dest_root_path = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\mask_extractor'
                
                mask_to_polygon_shp(image_name_base, front_line, source_tif_path, dest_root_path, domain)
                exit()

def center(x):
    return x['geometry']['coordinates']

def landsat_output_lookup(domain, date, orbit, satellite, level):
    output_hash_table[domain][date][orbit][satellite][level] += 1
    return not output_hash_table[domain][date][orbit][satellite][level] > 1

def landsat_scene_id_lookup(date, orbit, satellite, level):
    return scene_hash_table[date][orbit][satellite][level]

if __name__ == "__main__":
    plotting = True
    show_plots = False
    saving = True
    rerun = False

    #Initialize plots
    plt.close('all')
    font = {'family' : 'normal',
            'size'   : 14}
    plt.rc('font', **font)

    validation_files = glob.glob(r"D:\Daniel\Documents\Github\CALFIN Repo\reprocessing\landsat_raw\Petermann\*B[0-9].png")

    #Initialize output folders
    dest_root_path = r"..\outputs\mask_extractor"
    dest_path_qa = os.path.join(dest_root_path, 'quality_assurance')
    if not os.path.exists(dest_root_path):
        os.mkdir(dest_root_path)
    if not os.path.exists(dest_path_qa):
        os.mkdir(dest_path_qa)

    scaling = 96.3 / 1.97
    
    full_size = 512
    img_size = 448
    stride = 32
    
    settings = dict()
    settings['driver'] = 'mask_extractor'
    settings['validation_files'] = validation_files
    settings['date_index'] = 3 #The position of the date when the name is split by '_'. Used to differentiate between TerraSAR-X images.
    settings['log_file_name'] = 'logs_mask_extractor.txt'
#    settings['model'] = model
    settings['results'] = []
    settings['plotting'] = plotting
    settings['show_plots'] = show_plots
    settings['saving'] = saving
    settings['rerun'] = rerun
    settings['full_size'] = full_size
    settings['img_size'] = img_size
    settings['stride'] = stride
    settings['line_thickness'] = 3
    settings['kernel'] = cv2.getStructuringElement(cv2.MORPH_RECT, (settings['line_thickness'], settings['line_thickness']))
    settings['fjord_boundaries_path'] = r"..\training\data\fjord_boundaries"
    settings['tif_source_path'] = r"..\preprocessing\calvingfrontmachine\CalvingFronts\tif"
    settings['dest_path_qa'] = dest_path_qa
    settings['dest_root_path'] = dest_root_path
    settings['save_path'] = r"..\processing\landsat_preds"
    settings['total'] = len(validation_files)
    settings['empty_image'] = np.zeros((settings['full_size'], settings['full_size']))
    settings['scaling'] = scaling
    settings['domain_scalings'] = dict()
    settings['always_use_extracted_front'] = False
    settings['mask_confidence_strength_threshold'] = 0.875
    settings['edge_confidence_strength_threshold'] = 0.575
    settings['sub_padding_ratio'] = 1.5
    settings['edge_detection_threshold'] = 0.25 #Minimum confidence threshold for a prediction to be contribute to edge size
    settings['edge_detection_size_threshold'] = full_size / 8 #32 minimum pixel length required for an edge to trigger a detection
    settings['mask_detection_threshold'] = 0.25 #Minimum confidence threshold for a prediction to be contribute to edge size
    settings['mask_detection_ratio_threshold'] = 16 #if land/ice area is 32 times bigger than ocean/mélange, classify as no front/unconfident prediction
    settings['inter_box_distance_threshold'] = full_size / 16
    settings['image_settings'] = dict()
    settings['negative_image_names'] = []
    
    name_id_dict = dict()
    name_id_dict['Akullikassaap'] = 32621
    name_id_dict['Alangorssup'] = 32621
    name_id_dict['Alanngorliup'] = 32621
    name_id_dict['Brückner'] = 32624
    name_id_dict['Christian-IV'] = 32624
    name_id_dict['Cornell'] = 32621
    name_id_dict['Courtauld'] = 32624
    name_id_dict['Dietrichson'] = 32621
    name_id_dict['Docker-Smith'] = 32621
    name_id_dict['Eqip'] = 32621
    name_id_dict['Fenris'] = 32624
    name_id_dict['Frederiksborg'] = 32624
    name_id_dict['Gade'] = 32621
    name_id_dict['Glacier-de-France'] = 32624
    name_id_dict['Hayes'] = 32621
    name_id_dict['Heim'] = 32624
    name_id_dict['Helheim'] = 32624
    name_id_dict['Hutchinson'] = 32624
    name_id_dict['Illullip'] = 32621
    name_id_dict['Inngia'] = 32621
    name_id_dict['Issuusarsuit'] = 32621
    name_id_dict['Jakobshavn'] = 32621
    name_id_dict['Kakiffaat'] = 32621
    name_id_dict['Kangerdluarssup'] = 32621
    name_id_dict['Kangerlussuaq'] = 32624
    name_id_dict['Kangerlussuup'] = 32621
    name_id_dict['Kangiata-Nunaata'] = 32621
    name_id_dict['Kangilerngata'] = 32621
    name_id_dict['Kangilinnguata'] = 32621
    name_id_dict['Kangilleq'] = 32621
    name_id_dict['Kjer'] = 32621
    name_id_dict['Kong-Oscar'] = 32621
    name_id_dict['Kælvegletscher'] = 32624
    name_id_dict['Lille'] = 32621
    name_id_dict['Midgård'] = 32624
    name_id_dict['Morell'] = 32621
    name_id_dict['Nansen'] = 32621
    name_id_dict['Narsap'] = 32621
    name_id_dict['Nordenskiold'] = 32621
    name_id_dict['Nordfjord'] = 32624
    name_id_dict['Nordre-Parallelgletsjer'] = 32624
    name_id_dict['Nunatakassaap'] = 32621
    name_id_dict['Nunatakavsaup'] = 32621
    name_id_dict['Petermann'] = 32620
    name_id_dict['Perlerfiup'] = 32621
    name_id_dict['Polaric'] = 32624
    name_id_dict['Qeqertarsuup'] = 32621
    name_id_dict['Rink-Gletsjer'] = 32621
    name_id_dict['Rink-Isbrae'] = 32621
    name_id_dict['Rosenborg'] = 32624
    name_id_dict['Saqqarliup'] = 32621
    name_id_dict['Sermeq-Avannarleq-69'] = 32621
    name_id_dict['Sermeq-Avannarleq-70'] = 32621
    name_id_dict['Sermeq-Avannarleq-73'] = 32621
    name_id_dict['Sermeq-Kujalleq-70'] = 32621
    name_id_dict['Sermeq-Kujalleq-73'] = 32621
    name_id_dict['Sermeq-Silarleq'] = 32621
    name_id_dict['Sermikassak-N'] = 32621
    name_id_dict['Sermikassak-S'] = 32621
    name_id_dict['Sermilik'] = 32621
    name_id_dict['Sorgenfri'] = 32624
    name_id_dict['Steenstrup'] = 32621
    name_id_dict['Store'] = 32621
    name_id_dict['Styrtegletsjer'] = 32624
    name_id_dict['Sverdrup'] = 32621
    name_id_dict['Søndre-Parallelgletsjer'] = 32624
    name_id_dict['Umiammakku'] = 32621
    name_id_dict['Upernavik-NE'] = 32621
    name_id_dict['Upernavik-NW'] = 32621
    name_id_dict['Upernavik-SE'] =  32621

    version = "v1.0"
    source_path_manual = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\mask_extractor'
    source_path_auto = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\production_staging'
    dest_domain_path = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\upload_production\v1.0\level-1_shapefiles-domain-termini'
    dest_all_path = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\upload_production\v1.0\level-1_shapefiles-greenland-termini'
    fjord_boundary_path = r'D:\Daniel\Documents\Github\CALFIN Repo\training\data\fjord_boundaries_tif'
    pred_path = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\mask_extractor\quality_assurance'
    glacierIds = fiona.open(r'D:\Daniel\Downloads\GlacierIDs\GlacierIDsRef.shp', 'r', encoding='utf-8')
    glacier_centers = np.array(list(map(center, glacierIds)))
    centers_kdtree = KDTree(glacier_centers)

    with open(r"D:\Daniel\Documents\Github\CALFIN Repo\downloader\scenes\all_scenes.txt", 'r') as scenes_file:
        scene_list = scenes_file.readlines()
        scene_hash_table = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(str))))
        for scene in scene_list:
            scene = scene.split('\t')[0]
            scene_parts = scene.split('_')
            satellite = scene_parts[0]
            orbit = scene_parts[2]
            date = scene_parts[3]
            level = scene_parts[6]
            if date in scene_hash_table and orbit in scene_hash_table[date] and satellite in scene_hash_table[date][orbit] and level in scene_hash_table[date][orbit][satellite]:
                print('hash collision:', scene.split()[0], scene_hash_table[date][orbit][satellite][level].split()[0])
            else:
                scene_hash_table[date][orbit][satellite][level] = scene

    output_hash_table = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))))

    consolidate_shapefiles(source_path_manual, source_path_auto, dest_domain_path, dest_all_path, version, fjord_boundary_path, pred_path, settings)


# for each shapefile, for each date, check if there are no bad QA files, then




