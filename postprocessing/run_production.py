import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from keras import backend as K
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

import sys, glob, cv2, os, datetime
sys.path.insert(1, '../training/keras-deeplab-v3-plus')
sys.path.insert(2, '../training')

from validation import print_calfin_domain_metrics, print_calfin_all_metrics
from preprocessing import preprocess
from processing import process, compile_model, compile_hrnet_model
from postprocessing import postprocess

def main(settings, metrics):
    #Begin processing validation images
    troubled_ones = [3, 14, 22, 43, 66, 83, 97, 114, 161]
    troubled_ones = [3, 213, 238, 246, 283, 284, 1231, 1294, 1297, 1444, 1563, 1800, 2903, 6523, 
                     6122, 7200, 7512, 7611, 9123, 10200, 10302, 10400, 11101, 11219, 21641, 21341, 
                     23043, 23045, 23046, 23050, 23068, 23086, 23091, 23138, 23330, 23896, 23902, 23905,
                     24000, 24048, 24201, 24242]
#    troubled_ones = [23896, 23902, 23905, 24048]  
#    troubled_ones = [1800]  
#    troubled_ones = [23966, 23963, 23965] 
#    troubled_ones = [24238] 
    
    
#    troubled_ones = [23043, 23045, 23046, 23050, 23068, 23086   ]
#    troubled_ones = [23050]
    
#    troubled_ones = (np.random.rand(5) * 21000).astype(int)
#    troubled_ones = [10302]
#    troubled_ones = [21641]
#    10302-10405
#    for i in range(10233, 10234):
    
    
#    for i in range(10303, len(settings['validation_files'])): #Kronborg
    domains = ['Qeqertarsuup', 'Kakiffaat', 'Nunatakavsaup', 'Alangorssup', 'Akullikassaap', 'Upernavik-NE', 'Upernavik-NW',
               'Upernavik-SE' 'Sermikassak-N', 'Sermikassak-S', 'Inngia', 'Umiammakku', 'Rink-Isbrae', 'Kangerlussuup',
               'Kangerdluarssup', 'Perlerfiup', 'Sermeq-Silarleq', 'Kangilleq', 'Sermilik', 'Lille', 'Store']
#    domains = ['79North', 'Qeqertarsuup', 'Kakiffaat', 'Nunatakavsaup', 'Alangorssup', 'Akullikassaap', 'Upernavik-NE',
#           'Sermikassak-N', 'Sermikassak-S', 'Inngia', 'Umiammakku', 'Rink-Isbrae', 'Kangerlussuup', 
#           'Kangerdluarssup', 'Perlerfiup', 'Sermeq-Silarleq', 'Kangilleq', 'Sermilik', 'Lille', 'Store']
    domains = ['Upernavik-SE']
    for i in range(23000, len(settings['validation_files'])):
#    for i in range(24048, len(settings['validation_files'])):
#    for i in range(1444, 1445):
#    for i in troubled_ones:
        name = settings['validation_files'][i]
#        if '79North' not in name and '79North' not in name and 'Spaltegletsjer' not in name and 'Sermikassak' not in name and 'Upernavik-NW' not in name and 'Kronborg' not in name:
#            if 'Upernavik-NW' in name or '79North' in name or 'Spaltegletsjer' in name or 'Sermikassak' in name:
#        if 'Upernavik' in name:
        domain = name.split(os.path.sep)[-1].split('_')[0]
        if domain in domains:
#        if True:
            preprocess(i, settings, metrics)
            process(settings, metrics)
            postprocess(settings, metrics)
#    for i in range(0, 23890):
##    for i in range(1444, 1445):
##    for i in troubled_ones:
#        name = settings['validation_files'][i]
##        if '79North' not in name and '79North' not in name and 'Spaltegletsjer' not in name and 'Sermikassak' not in name and 'Upernavik-NW' not in name and 'Kronborg' not in name:
##            if 'Upernavik-NW' in name or '79North' in name or 'Spaltegletsjer' in name or 'Sermikassak' in name:
##        if 'Upernavik' in name:
#        domain = name.split(os.path.sep)[-1].split('_')[0]
##        if domain in domains:
#        if True:
#            preprocess(i, settings, metrics)
#            process(settings, metrics)
#            postprocess(settings, metrics)
    #Print statistics
#    print_calfin_domain_metrics(settings, metrics)
#    print_calfin_all_metrics(settings, metrics)
    plt.show()
    
    return settings, metrics


def initialize(img_size):
    #initialize settings and model if not already done
    plotting = True
    show_plots = False
    saving = True
    rerun = False

    #Initialize plots
    plt.close('all')
    font = {'family' : 'normal',
            'size'   : 14}
    plt.rc('font', **font)
    plt.rcParams["figure.figsize"] = (16,9)
    np.set_printoptions(precision=3)
    
    validation_files = glob.glob(r"..\processing\landsat_raw_processed\*B[0-9].png")

    #Initialize output folders
    dest_root_path = r"..\outputs\production_staging"
    dest_path_qa = os.path.join(dest_root_path, 'quality_assurance')
    dest_path_qa_bad = os.path.join(dest_root_path, 'quality_assurance_bad')
    if not os.path.exists(dest_root_path):
        os.mkdir(dest_root_path)
    if not os.path.exists(dest_path_qa):
        os.mkdir(dest_path_qa)
    if not os.path.exists(dest_path_qa_bad):
        os.mkdir(dest_path_qa_bad)

    scaling = 96.3 / 1.97
    full_size = 256
    stride = 16

    #Intialize processing pipeline variables
    settings = dict()
    settings['driver'] = 'production'
    settings['validation_files'] = validation_files
    settings['date_index'] = 3 #The position of the date when the name is split by '_'. Used to differentiate between TerraSAR-X images.
    settings['log_file_name'] = 'logs_production.txt'
    settings['model'] = model
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
    settings['confidence_kernel'] = cv2.getStructuringElement(cv2.MORPH_RECT, (settings['line_thickness']*5, settings['line_thickness']*5))
    settings['fjord_boundaries_path'] = r"..\training\data\fjord_boundaries"
    settings['tif_source_path'] = r"..\preprocessing\calvingfrontmachine\CalvingFronts\tif"
    settings['dest_path_qa'] = dest_path_qa
    settings['dest_root_path'] = dest_root_path
    settings['save_path'] = r"..\processing\landsat_preds"
    settings['dest_path_qa_bad'] = dest_path_qa_bad
    settings['save_to_all'] = False
    settings['total'] = len(validation_files)
    settings['empty_image'] = np.zeros((settings['full_size'], settings['full_size']))
    settings['scaling'] = scaling
    settings['domain_scalings'] = dict()
    settings['always_use_extracted_front'] = True
    settings['mask_confidence_strength_threshold'] = 0.875
    settings['edge_confidence_strength_threshold'] = 0.525
    settings['sub_padding_ratio'] = 2.5
    settings['edge_detection_threshold'] = 0.25 #Minimum confidence threshold for a prediction to be contribute to edge size
    settings['edge_detection_size_threshold'] = full_size / 8 #32 minimum pixel length required for an edge to trigger a detection
    settings['mask_detection_threshold'] = 0.5 #Minimum confidence threshold for a prediction to be contribute to edge size
    settings['mask_detection_ratio_threshold'] = 32 #if land/ice area is 32 times bigger than ocean/mélange, classify as no front/unconfident prediction
    settings['mask_edge_buffered_mean_threshold'] = 0.13 #threshold deviation of the mean of mask pixels around the deteccted edge from 0 (mask-edge agreement = 0.0 deviation
    settings['image_settings'] = dict()
    settings['negative_image_names'] = []

    metrics = dict()
    metrics['confidence_skip_count'] = 0
    metrics['no_detection_skip_count'] = 0
    metrics['front_count'] = 0
    metrics['image_skip_count'] = 0
    metrics['mean_deviations_pixels'] = np.array([])
    metrics['mean_deviations_meters'] = np.array([])
    metrics['validation_distances_pixels'] = np.array([])
    metrics['validation_distances_meters'] = np.array([])
    metrics['domain_mean_deviations_pixels'] = defaultdict(lambda: np.array([]))
    metrics['domain_mean_deviations_meters'] = defaultdict(lambda: np.array([]))
    metrics['domain_validation_distances_pixels'] = defaultdict(lambda: np.array([]))
    metrics['domain_validation_distances_meters'] = defaultdict(lambda: np.array([]))
    metrics['domain_validation_edge_ious'] = defaultdict(lambda: np.array([]))
    metrics['domain_validation_mask_ious'] = defaultdict(lambda: np.array([]))
    metrics['domain_validation_calendar'] = defaultdict(lambda: dict((k, 0) for k in range(1972, datetime.datetime.now().year)))
    metrics['resolution_deviation_array'] = np.zeros((0,2))
    metrics['validation_edge_ious'] = np.array([])
    metrics['validation_mask_ious'] = np.array([])
    metrics['resolution_iou_array'] = np.zeros((0,2))
    metrics['true_negatives'] = 0
    metrics['false_negatives'] = 0
    metrics['false_positive'] = 0
    metrics['true_positives'] = 0

    #Each 256x256 image will be split into 9 overlapping 224x224 patches to reduce boundary effects
    #and ensure confident predictions. To normalize this when overlaying patches back together,
    #generate normalization image that scales the predicted image based on number of patches each pixel is in.
    strides = int((full_size - img_size) / stride + 1) #(256 - 224) / 16 + 1 = 3
    pred_norm_image = np.zeros((full_size, full_size, 3))
    pred_norm_patch = np.ones((img_size, img_size, 3))
    for x in range(strides):
        for y in range(strides):
            x_start = x * stride
            x_end = x_start + img_size
            y_start = y * stride
            y_end = y_start + img_size
            pred_norm_image[x_start:x_end, y_start:y_end] += pred_norm_patch
    settings['pred_norm_image'] = pred_norm_image
    
#    log_path = os.path.join(dest_root_path, settings['log_file_name'])
#    sys.stdout = open(log_path, 'a')

    return settings, metrics


if __name__ == '__main__':
    #Initialize model once, and setup variable passing/main function. Must be done in global namespace to benefit from model reuse.
    img_size = 224
    try:
        model
    except NameError:
        model = compile_model(img_size)
    settings, metrics = initialize(img_size)

    #Execute calving front extraction pipeline.
    main(settings, metrics)