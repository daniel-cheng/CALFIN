import os, glob
import numpy as np
os.environ['GDAL_DATA'] = r'D://ProgramData//Anaconda3//envs//cfm//Library//share//gdal' #Ensure crs are exported correctly by gdal/osr/fiona
import fiona

from osgeo import gdal, osr
from skimage.io import imread


def bounds_from_shp(domain_shp_path):
    """Returns QgsRectangle representing bounds of geotiff in projection coordinates
        :domain_shp_path: str
        :bounds: QgsRectangle
    """
    with fiona.open(domain_shp_path, 'r', encoding='utf-8') as source_shp:
        coords = np.array(list(source_shp)[0]['geometry']['coordinates'])        
        x = coords[0, :, 0]
        y = coords[0, :, 1]
        return {"xMin":float(np.min(x)), "yMin":float(np.min(y)), "xMax":float(np.max(x)), "yMax":float(np.max(y))}


def png_to_geotiff(array, bounds, dest_path, srs):
    """Save a Geotiff raster from a png array.
        :param array: ndarray
    """
    
    # TODO: Fix X/Y coordinate mismatch and use ns/ew labels to reduce confusion. Also, general cleanup and refactoring.
    rows, cols = array.shape[:2]
    x_pixel_size = (bounds['xMax'] - bounds['xMin']) / cols  # size of the pixel...        
    y_pixel_size = (bounds['yMax']  - bounds['yMin']) / rows  # size of the pixel...        
    x_min = bounds['xMin']
    y_max = bounds['yMax']   # x_min & y_max are like the "top left" corner.
    
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        dest_path,
        cols,
        rows,
        1,
        gdal.GDT_Float32, )
    dataset.SetGeoTransform((
        x_min,    # 0
        x_pixel_size,  # 1
        0,                      # 2
        y_max,    # 3
        0,                      # 4
        -y_pixel_size))  #6
    
    dataset.SetProjection(srs.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()  # Write to disk.
    
    #Compress it
    base_command = ['gdal_translate', '-co', 'compress=lzw']
    temp_path = dest_path[0:-4] + '_temp.tif'
    command = base_command + [dest_path, temp_path]
    status = subprocess.run(command)
    shutil.move(temp_name, src_name)

    return dataset, dataset.GetRasterBand(1)  #If you need to return, remenber to return  also the dataset because the band don`t live without dataset.

        
if __name__ == "__main__":
    fjord_boundary_source_path = r'../training/data/fjord_boundaries'
    domain_path = r'../preprocessing/domains'
    fjord_boundary_tif_path = r'../training/data/fjord_boundaries_tif'
    
    #Use domain shapefiles
    for fjord_boundary_path in glob.glob(os.path.join(fjord_boundary_source_path, '*overrides.png')):
        try:
            basename = os.path.basename(fjord_boundary_path)
            stripped_basename = os.path.splitext(basename)[0]
            domain = basename.split('_')[0]
            domain_shp_path = os.path.join(domain_path, domain + '.shp')
            domain_prj_path = os.path.join(domain_path, domain + '.prj')
            
            srs = osr.SpatialReference()
            prj_file = open(domain_prj_path, 'r')
            prj_txt = prj_file.read()
            srs = osr.SpatialReference()
            srs.ImportFromESRI([prj_txt])
            
            fjord_boundary_img = imread(fjord_boundary_path, as_gray=True).astype(np.float32)
            shp_bounds = bounds_from_shp(domain_shp_path)
            if shp_bounds is not None:
                dest_path = os.path.join(fjord_boundary_tif_path, stripped_basename + '.tif')
                print(dest_path)
                png_to_geotiff(fjord_boundary_img, shp_bounds, dest_path, srs)
        except FileNotFoundError as e:
            print(e)
