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
from osgeo import gdal, osr
import rasterio
from rasterio import features
from rasterio.windows import from_bounds
import fiona
from shapely.geometry import mapping, Point
from shapely.geometry.polygon import Polygon
from pyproj import Proj, transform

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
    bad_file_name_list = defaultdict(bool)
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
        bad_file_name_list[file_name.split('_overlay_front.png')[0]] = True
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
        # if date_dashed not in bad_dates:
        dates[date_dashed].append(file_path)
        # else:    
        #     #Skip  dates where not all fronts are good
        #     print('bad date:', date_dashed)
    return dates, bad_file_name_list


def consolidate_shapefiles(source_path_manual, source_path_auto, fjord_boundary_path, domain_path, exits):
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
    domains = ['Qeqertarsuup', 'Kakiffaat', 'Nunatakavsaup', 'Upernavik-NW',
               'Umiammakku', 'Rink-Isbrae', 'Kangerlussuup', 'Kangerdluarssup',
                'Perlerfiup', 'Sermeq-Silarleq', 'Kangilleq', 'Sermilik', 'Lille', 'Store']
    
    
    domains = ['Gade', 'Upernavik-NE', 'Morell', 'Kangerlussuaq' 'Kronborg', 'Midgård']
    domains = ['Akullikassaap', 'Alanngorliup', 'Brückner', 'Christian-IV', 'Cornell', 'Courtauld', 'Upernavik-NE', 'Upernavik-SE']
    domains = ['Dietrichson', 'Docker-Smith', 'Eqip', 'Fenris', 'Frederiksborg', 'Gade', 'Glacier-de-France', 'Hayes', 'Heim', 'Helheim', 'Hutchinson', 'Illullip']
    domains = ['Issuusarsuit', 'Kælvegletscher', 'Kangiata-Nunaata', 'Kangilerngata', 'Kangilinnguata', 'Kjer', 'Kong-Oscar']
    domains = ['Nansen', 'Narsap', 'Nordenskiold', 'Nordfjord', 'Nordre-Parallelgletsjer', 'Søndre-Parallelgletsjer', 'Nunatakassaap', 'Petermann', 'Polaric', 'Rink-Gletsjer', 'Rosenborg', 'Saqqarliup', 'Sermeq-Avannarleq-69']
    domains = ['Kakiffaat', 'Brückner', 'Søndre-Parallelgletsjer', 'Kjer']
    domains = ['Nordenskiold']
    domains = ['Steenstrup']
    # domains = ['Kjer']
    sections = [[0, 17], [17, 35], [35, 52], [52, 69]]
    section = sections[3]
    # section = [37, 55]
    # section = [0, 15]
    # for domain in os.listdir(source_auto_qa_path)[section[0]:section[1]]:
    for domain in os.listdir(source_auto_qa_path):
        if '.' in domain:
            continue
        if domain not in domains:
            continue
        counter = 0
        file_list_manual = glob.glob(os.path.join(source_manual_qa_path, domain, '*_pred.tif'))
        file_list_auto = glob.glob(os.path.join(source_auto_qa_path, domain, '*_pred.tif'))
        file_list = file_list_manual + file_list_auto
        file_list = file_list_auto
        file_list = file_list_manual
        file_list.sort(key=landsat_sort)
        
        bad_file_list_manual = glob.glob(os.path.join(source_manual_qa_path + '_bad', domain, '*_pred.tif'))
        bad_file_list_auto = glob.glob(os.path.join(source_auto_qa_path + '_bad', domain, '*_pred.tif'))
        bad_file_list_manual_pruned = glob.glob(os.path.join(source_manual_qa_path + '_bad', domain, '*_overlay_front.png'))
        bad_file_list_auto_pruned = glob.glob(os.path.join(source_auto_qa_path + '_bad', domain, '*_overlay_front.png'))
        bad_file_list = bad_file_list_manual + bad_file_list_auto + bad_file_list_manual_pruned + bad_file_list_auto_pruned
        bad_file_list = bad_file_list_auto + bad_file_list_auto_pruned
        bad_file_list = bad_file_list_manual + bad_file_list_manual_pruned

        bad_file_list.sort(key=landsat_sort)
                
        fjord_boundary_file_path = os.path.join(fjord_boundary_path, domain + '_fjord_boundaries.tif')
        fjord_boundary_overrides_file_path = os.path.join(fjord_boundary_path, domain + '_fjord_boundaries_overrides.tif')
        with rasterio.open(fjord_boundary_file_path) as fjord_boundary_tif:
            original_mask = fjord_boundary_tif.read(1)
            fjord_bounds = rasterio.transform.array_bounds(original_mask.shape[0], original_mask.shape[1], fjord_boundary_tif.transform)
            
            original_mask_overrides = None
            if os.path.exists(fjord_boundary_overrides_file_path):
                fjord_boundary_overrides_tif = rasterio.open(fjord_boundary_overrides_file_path)
                original_mask_overrides = np.where(fjord_boundary_overrides_tif.read(1) > 127, 1.0, 0.0)
                original_mask = original_mask * original_mask_overrides
            
            domain_exits = filter_domain_exits(fjord_bounds, domain_path, domain, exits)
            dates, bad_file_name_list = get_file_lists(file_list, bad_file_list)
            #For each date, retrieve mask by projecting each calving front onto a single mask
            for date_key in dates.keys():
                if date_key == '2019-04-23' or date_key == '2018-07-09':
                # if True:
                    counter = process_date(counter, domain, dates, bad_file_name_list, date_key, original_mask, fjord_bounds, fjord_boundary_tif, fjord_boundary_file_path, original_mask_overrides, domain_exits)
                # exit()
        print('Completed', counter, 'out of', len(dates.keys()))


def process_date(counter, domain, dates, bad_file_name_list, date_key, original_mask, fjord_bounds, fjord_boundary_tif, fjord_boundary_file_path, original_mask_overrides, exits):
    """Processes a single day of calving front(s), and save it to a single polygon Shapefile."""
    mask = np.empty_like(original_mask)
    np.copyto(mask, original_mask)
    combined_pred_mask = 255 - (mask / mask.max() * 255)
    combined_pred_mask_validity = 1 - mask / mask.max()
    count = 0
    for file_path in dates[date_key]:
        file_name = os.path.basename(file_path)
        file_basename = file_name.split('_pred.tif')[0]
        if file_basename not in bad_file_name_list:
            mask, combined_pred_mask, combined_pred_mask_validity, count = process_file(combined_pred_mask, combined_pred_mask_validity, domain, file_path, fjord_bounds, fjord_boundary_tif, mask, original_mask, count)
    if count == 0:
        print('No contours (all pruned) for:' + domain + ' ' + date_key)
        return counter
    
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
    
    # plt.figure(str(counter) + '-combined_pred_mask_validity')
    # plt.imshow(combined_pred_mask_validity)
    # plt.figure(str(counter) + '-combined_pred_mask')
    # plt.imshow(combined_pred_mask)
    # plt.figure(str(counter) + '-mask')
    # plt.imshow(mask)
    mean_values = []
    for i in range(len(sizes)):
        if sizes[ordering[i]] >= min_size_floor:
            mask_indices = output == ordering[i]
            valid_indices = combined_pred_mask_validity == 1
            valid_mask_indices = np.logical_and(mask_indices, valid_indices)
            image_component = combined_pred_mask[valid_mask_indices] #get green channel
            mean_value = np.mean(image_component)
            mean_values.append(mean_value)
            
            # image_component_mask = np.zeros(mask.shape)
            # image_component_mask[valid_mask_indices] = 1.0
            # plt.figure(str(counter) + '-image_component_mask-' + str(i))
            # plt.imshow(image_component_mask)
            # plt.figure(str(counter) + '-image_component-' + str(i))
            # plt.imshow(combined_pred_mask * image_component_mask)
            # print(mean_value)
            if mean_value < 224:
                new_mask[mask_indices] = 0
   
    # replace all values in land/ice mask with correct color
    new_mask = median_filter(new_mask, size=3)
    new_mask_padded = np.pad(new_mask, 1, constant_values=255)
    contours = measure.find_contours(new_mask_padded, 127)              
    
    if len(contours) < 1:
        print('No contours for:' + domain + ' ' + date_key)
        # plt.figure(date_key + '-' + str(counter) + '-Final')
        # plt.imshow(new_mask)
        # plt.show()
        # exit()
        # print(mean_values)
        return counter
    
    # plt.figure(date_key + '-' + str(counter) + '-Final')
    # plt.imshow(new_mask_padded)
    # plt.show()
    
    file_name = os.path.basename(file_path)
    file_name_parts = file_name.split('_')
    file_basename = "_".join(file_name_parts[0:-2])
    image_name_base = file_basename
    id_str = file_name_parts[-2]
    
    #For each polygon, swap rows and columns, and de-pad
    front_lines = []
    for contour in contours:
        transposed_contour = np.transpose(contour)
        #Perform final filtering by size (to avoid iterative median filtering).
        if transposed_contour.shape[1] > 10:
            front_lines.append(transposed_contour - 1) 
    
    source_tif_path = fjord_boundary_file_path
    if 'production' in file_path:
        dest_root_path = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\production_staging'
    elif 'mask_extractor' in file_path:
        dest_root_path = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\mask_extractor'
                
    #count exits to ensure certain pre-selected areas are always masked 
    #(and whose failure to be masked indicates missing contours)
    if exits_excluded(new_mask, fjord_boundary_tif, exits):
        mask_to_polygon_shp(image_name_base, id_str, front_lines, fjord_boundary_tif, source_tif_path, dest_root_path, domain)
        # plt.figure(date_key + '-' + str(counter) + '-Final')
        # plt.imshow(new_mask)
        # plt.show()
        print('#' + str(counter) + ': ' + domain + ' ' + date_key + ' complete')
        counter += 1
        # exit
    else:
        print('Missing contours for:' + domain + ' ' + date_key)
        
        # plt.figure(date_key + '-' + str(counter) + '-mask')
        # plt.imshow(mask)
        # plt.show()
        # print(mean_values)
    return counter


def process_file(combined_pred_mask, combined_pred_mask_validity, domain, file_path, fjord_bounds, fjord_boundary_tif, mask, original_mask, count):
    """Processes each file of possibly many within a single date. 
        Returns the validity mask and polyline projected onto the land-ice/ocean mask."""
    # print('\t' + file_path)
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
            
            validity = np.zeros(combined_pred_mask_validity.shape)
            validity[sub_y1:sub_y2, sub_x1:sub_x2] = 1
            combined_pred_mask = np.maximum(combined_pred_mask, pred_mask)
            combined_pred_mask_validity = np.maximum(combined_pred_mask_validity, validity)
            
            polyline_coords = source_shp[0]['geometry']['coordinates']
            polyline_coords = snap_polyline_to_boundaries(original_mask, polyline_coords, x_min_fjord, x_max_fjord, y_min_fjord, y_max_fjord, fjord_boundary_tif)
            if polyline_coords is None:
                return mask, combined_pred_mask, combined_pred_mask_validity, count
            
            #Rasterize the polyline
            geometry = {'type': 'LineString'}
            geometry['coordinates'] = polyline_coords
            mask = features.rasterize([(geometry, 0)],
                # all_touched=True,
                out=mask,
                out_shape=fjord_boundary_tif.shape,
                transform=fjord_boundary_tif.transform)
            return mask, combined_pred_mask, combined_pred_mask_validity, count + 1

def exits_excluded(mask, fjord_boundary_tif, domain_exits):
    """Detects if all exits are excluded from the polygon mask."""
    for domain_exit in domain_exits:
        pixel_coord = rasterio.transform.rowcol(fjord_boundary_tif.transform, domain_exit[0], domain_exit[1])
        if mask[pixel_coord] == 0:
            return False
    return True


def filter_domain_exits(fjord_bounds, domain_path, domain, exits):
    """Detects if all exits are excluded from the polygon mask."""
    #Reproject fjord bounds into 3413.
    domain_prj_path = os.path.join(domain_path, domain + '.prj')
    prj_txt = open(domain_prj_path, 'r').read()
   
    #Must project epsg:3413 points to polygon's projection to allow for easy bounding box point-in-polygon testing
    #otherwise the boxes will be skewed
    srs = osr.SpatialReference()
    srs.ImportFromESRI([prj_txt])
    srs.AutoIdentifyEPSG()
    code = srs.GetAuthorityCode(None)
    
    in_proj = Proj(init='epsg:3413')
    out_proj = Proj(init='epsg:' + code)
    
    x_min = fjord_bounds[0]
    y_min = fjord_bounds[1]
    x_max = fjord_bounds[2]
    y_max = fjord_bounds[3]
    top_left = (x_min, y_max)
    top_right = (x_max, y_max)
    bottom_right = (x_max, y_min)
    bottom_left = (x_min, y_min)
    polygon_points = [top_left, top_right, bottom_right, bottom_left]
    fjord_polygon = Polygon(polygon_points)
    
    #Test if each point in polygon.
    domain_exits = []
    for exit_coords in exits:
        f_id, exit_coords = exit_coords
        projected_exit_coords = transform(in_proj, out_proj, exit_coords[0], exit_coords[1])
        exit_point = Point(projected_exit_coords)
        if fjord_polygon.contains(exit_point):
            domain_exits.append(projected_exit_coords)
    print('domain_exits:', domain_exits)
    return domain_exits


def snap_polyline_to_boundaries(original_mask, polyline_coords, x_min_fjord, x_max_fjord, y_min_fjord, y_max_fjord, fjord_boundary_tif):
    """Connects the polyline to the nearest fjord boundary or image edge."""
    edge_image = np.pad(np.ones((original_mask.shape[0] - 1, original_mask.shape[1] - 1)), 1)
    edge_distances, edge_indices = distance_transform_edt(edge_image, return_indices=True)      
    fjord_distances, fjord_indices = distance_transform_edt(original_mask, return_indices=True)
    
    #Clip coordinates on boundary to ensure no negative indexing occurs during coordinate transform
    polyline_coords_x = np.array(polyline_coords)[:,0]
    polyline_coords_y = np.array(polyline_coords)[:,1]
    polyline_coords_x_clipped = np.clip(polyline_coords_x, x_min_fjord, x_max_fjord)
    polyline_coords_y_clipped = np.clip(polyline_coords_y, y_min_fjord, y_max_fjord)
    polyline_coords = list(zip(polyline_coords_x_clipped, polyline_coords_y_clipped))
    
    #Find closest fjord boundary/image edge positions
    endpoint_pixel_coord_1 = rasterio.transform.rowcol(fjord_boundary_tif.transform, polyline_coords[0][0], polyline_coords[0][1])
    endpoint_pixel_coord_2 = rasterio.transform.rowcol(fjord_boundary_tif.transform, polyline_coords[-1][0], polyline_coords[-1][1])
    closest_edge_pixel_1 = np.array([edge_indices[0][endpoint_pixel_coord_1], edge_indices[1][endpoint_pixel_coord_1]])
    closest_edge_pixel_2 = np.array([edge_indices[0][endpoint_pixel_coord_2], edge_indices[1][endpoint_pixel_coord_2]])
    closest_fjord_pixel_1 = np.array([fjord_indices[0][endpoint_pixel_coord_1], fjord_indices[1][endpoint_pixel_coord_1]])
    closest_fjord_pixel_2 = np.array([fjord_indices[0][endpoint_pixel_coord_2], fjord_indices[1][endpoint_pixel_coord_2]])
    
    #If endpoints are not on image edge, add the closest point to a fjord boundary.
    endpoint_pixel_coord_array_1 = np.array(endpoint_pixel_coord_1)
    endpoint_pixel_coord_array_2 = np.array(endpoint_pixel_coord_2)
    dist_edge_1 = np.linalg.norm(endpoint_pixel_coord_array_1 - closest_edge_pixel_1)
    dist_edge_2 = np.linalg.norm(endpoint_pixel_coord_array_2 - closest_edge_pixel_2)
    dist_fjord_1 = np.linalg.norm(endpoint_pixel_coord_array_1 - closest_fjord_pixel_1)
    dist_fjord_2 = np.linalg.norm(endpoint_pixel_coord_array_2 - closest_fjord_pixel_2)
    
    #If the jump distance is too large, discard the front.
    jump_limit = 0.15 * np.mean([original_mask.shape[0], original_mask.shape[1]])
    if min(dist_edge_1, dist_fjord_1) > jump_limit or min(dist_edge_2, dist_fjord_2) > jump_limit:
        print('Large jump detected, discarding front...')
        return None
    
    #Snape to closest fjord boundary/image edge
    edge_bias = 1.5 #edge bias ratio (edge must be X times as fjord to endpoints to be chosen), allows for fjord close to image edges.
    if dist_edge_1 * edge_bias < dist_fjord_1:
        closest_pixel_1 = closest_edge_pixel_1
    else:
        closest_pixel_1 = closest_fjord_pixel_1
    if dist_edge_2 * edge_bias < dist_fjord_2:
        closest_pixel_2 = closest_edge_pixel_2
    else:
        closest_pixel_2 = closest_fjord_pixel_2
            
    closest_coord_1 = rasterio.transform.xy(fjord_boundary_tif.transform, closest_pixel_1[0], closest_pixel_1[1])
    closest_coord_2 = rasterio.transform.xy(fjord_boundary_tif.transform, closest_pixel_2[0], closest_pixel_2[1])
    return [closest_coord_1] + polyline_coords + [closest_coord_2]

if __name__ == "__main__":
    #Initialize plots
    plt.close('all')
    
    source_path_manual = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\mask_extractor'
    source_path_auto = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\production_staging'
    fjord_boundary_path = r'D:\Daniel\Documents\Github\CALFIN Repo\training\data\fjord_boundaries_tif'
    domain_path = r'D:\Daniel\Documents\Github\CALFIN Repo\preprocessing\domains'
    exits_path = r"D:\Daniel\Documents\Github\CALFIN Repo\postprocessing\GlacierExclusionRef.shp"
            
    exits_shp = fiona.open(exits_path, 'r', encoding='utf-8')
    exits = []
    for exit_point in exits_shp:
        # print(exit_point)
        if exit_point['geometry'] is not None:
            exits.append([exit_point['id'], exit_point['geometry']['coordinates']])
    
    consolidate_shapefiles(source_path_manual, source_path_auto, fjord_boundary_path, domain_path, exits)
