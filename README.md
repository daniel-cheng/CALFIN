# CALFIN
Calving Front Machine. Automated detection of glacial positions, using neural networks.

<Cite this>

<Dataset link>

<Image>

<Dependancies>
Python 3.6, Keras v  
<How to Run>
<-Preprocessing>
First, create a square Shapefile polygon in the projection of your source imagery in preprocessing/domains.
Then, subset all source images, perform detail enhancment, and save the final 256x256 RGB image in processing/landsat_raw_processed.
Next, execute post_processing/run_production.py.
Optionally, veify the results of outputs/production_staging/quality_assurance/<domain>, and copy any *overlay_front.png files that are incorrect to the corresponding outputs/production_staging/quality_assurance_bad/<domain> folders to eliminate it from the final output.
Finally, run preprocessing/bulk_shapefile_polygonizer.py, preprocessing/bulk_shapefile_consolidator.py, and preprocessing/bulk_shapefile_consolidator_closed.py to create the final shapefile outputs in outputs/upload_production/v1.0/level-1_shapefiles-domain-termini.
<-Training>
<Processing>
<Postprocessing>

<Outputs>

<Running CALFIN on New Domains>
If you plan to use CALFIN on a domain outside of the existing set, be familiar with the training set and the set of conditions CALFIN can handle.
CALFIN was trained using Landsat (optical) and Sentinel-1 (SAR) data. The training set includes 1600+ Greenlandic glaciers and 200+ Antarctic glaciers/ice shelves.
CALFIN can handle ice tongues, branching, Landsat 7 Scanline Corrector Errors, sea ice, shadows, and light cloud cover.
CALFIN requires a fjord boundaries mask in order to function - this must 

<Project Structure>

<Acknowledgements>

<Contact>

<License>