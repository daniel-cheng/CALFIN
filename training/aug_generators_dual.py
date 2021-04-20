from albumentations import LongestMaxSize, PadIfNeeded, Compose, OneOf, CLAHE, IAASharpen, IAAEmboss, IAAAdditiveGaussianNoise, GaussNoise, RandomRotate90, Flip, Transpose, RandomBrightnessContrast, ShiftScaleRotate, RandomContrast, RandomBrightness, HueSaturationValue, ChannelShuffle, RandomCrop, CenterCrop
import cv2, glob, os
import numpy as np
import skimage
from skimage.io import imsave, imread
import tensorflow as tf
#from tensorflow import keras
from keras.applications import imagenet_utils
from skimage.transform import resize
from random import shuffle
from skimage import exposure
import numpngw
import matplotlib.pyplot as plt
from dateutil.parser import parse

def aug_validation(prob=1.0, img_size=224):
    return Compose([
        LongestMaxSize(max_size=img_size),
        PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
    ], p=prob)

def aug_resize(prob=1.0, img_size=224, interpolation=0):
    return Compose([
        LongestMaxSize(max_size=img_size, interpolation=interpolation)
    ], p=prob)
    
def aug_pad(prob=1.0, img_size=224):
    return Compose([
        PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT)
    ], p=prob)

def aug_daniel_part1(prob=1.0, img_size=224):
    return Compose([
        OneOf([
            CLAHE(clip_limit=2, p=.6),
            IAASharpen(p=.2),
            IAAEmboss(p=.2)
        ], p=.7),
        OneOf([
            IAAAdditiveGaussianNoise(p=.3),
            GaussNoise(p=.7),
        ], p=.5),
        RandomRotate90(p=0.5),
        Flip(p=0.5),
        Transpose(p=0.5),
        #OneOf([
        #    MotionBlur(p=.2),
        #    MedianBlur(blur_limit=3, p=.3),
        #    Blur(blur_limit=3, p=.5),
        #], p=.4),
        RandomBrightnessContrast(p=.5),
    ], p=prob)

def aug_daniel_part2(prob=1.0, img_size=224):
    return Compose([
        #PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
        ShiftScaleRotate(shift_limit=.0625, scale_limit=0.0, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=.75),
        #OneOf([
        #    OpticalDistortion(interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
        #    GridDistortion(distort_limit=0.2, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
        #    #ElasticTransform(approximate=True, sigma=50, alpha_affine=10, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT), # approximate gives up to 2x speedup on large images. Elastic disabled because it makes images hard to understand.
        #    IAAPiecewiseAffine(scale=(0.005, 0.015), mode='constant'),
        #    ##IAAPerspective(), #produces interpolation artifacts - already tried setting to 0, 1, but perhapps not all the way?
        #    JpegCompression(quality_lower=40)
        #], p=0.7)
        #JpegCompression(quality_lower=40, p=.3)
    ], p=prob)

def aug_daniel(prob=0.8):
    return Compose([
        RandomRotate90(p=0.5),
        Transpose(p=0.5),
        Flip(p=0.5),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
            #Blur(),
        ], p=0.3),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            OneOf([
                RandomContrast(),
                RandomBrightness(),
            ]),
            #Blur(),
            #GaussNoise()
        ], p=0.5),
        HueSaturationValue(p=0.5)
        ], p=prob)

def aug_daniel_prepadded(prob=1.0, image_size=448):
    return Compose([
        RandomRotate90(p=0.5),
        Transpose(p=0.5),
        Flip(p=0.5),
        OneOf([
            RandomCrop(height=image_size, width=image_size, p=0.3),
            Compose([
                #3.94 determined by largest angle possible rotatable without introducing nodata pixels into center crop area
                ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0, rotate_limit=6, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                CenterCrop(height=int(round(236/224*image_size)), width=int(round(236/224*image_size)), p=1.0),
                RandomCrop(height=image_size, width=image_size, p=1.0)
            ], p=0.4),
            Compose([
                #3.94 determined by largest angle possible rotatable without introducing nodata pixels into center crop area
                ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0, rotate_limit=12, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                CenterCrop(height=image_size, width=image_size, p=1.0)
            ], p=0.3)
        ], p=1.0),
        #OneOf([
            #IAASharpen(),
            #IAAEmboss(),
#            RandomBrightnessContrast(brightness_limit=0.01, contrast_limit=0.01) # This causes a blackout for some reason
            #Blur(),
            #GaussNoise()
        #], p=0.5),
        IAASharpen(p=0.2),
        IAAAdditiveGaussianNoise(p=0.2),
#        HueSaturationValue(p=0.3)
        #ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0, rotate_limit=2, border_mode=cv2.BORDER_CONSTANT, p=.75),
        #ChannelShuffle(p=0.33)
    ], p=prob)

def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Input array scaled to [-1.,1.]
    """
    return imagenet_utils.preprocess_input(x, mode='tf')

def extract_patches(img, window_shape=(512, 512), stride=64):
    #In order to extract patches, determine the resolution of the image needed to extract regular patches
    #Pad image if necessary
    nr, nc = img.shape

    if stride != 0:
        #If image is smaller than window size, pad to fit.
        #else, pad to the least integer multiple of stride
        #Window shape is assumed to multiple of stride.
        #Find the smallest multiple of stride that is greater than image dimensions
        leastRowStrideMultiple = (np.ceil(nr / stride) * stride).astype(np.uint16)
        leastColStrideMultiple = (np.ceil(nc / stride) * stride).astype(np.uint16)
        #If image is smaller than window, pad to window shape. Else, pad to least stride multiple.
        nrPad = max(window_shape[0], leastRowStrideMultiple) - nr
        ncPad = max(window_shape[1], leastColStrideMultiple) - nc
        #Add Stride border around image, and nrPad/ncPad to image to make sure it is divisible by stride.
        stridePadding = int(stride / 2)
        #TEST: Turning off stride padding
        stridePadding = 0
        paddingRow = (stridePadding, nrPad + stridePadding)
        paddingCol = (stridePadding, ncPad + stridePadding)
        padding = (paddingRow, paddingCol)
        imgPadded = np.pad(img, padding, 'constant')

        patches = skimage.util.view_as_windows(imgPadded, window_shape, stride)
        nR, nC, H, W = patches.shape
        nWindow = nR * nC
        patches = np.reshape(patches, (nWindow, H, W))
    else:
        patches = np.reshape(img, (1, nr, nc))


    return patches

def extract_rgb_patches(img, window_shape=(512, 512, 3), stride=32):
    #In order to extract patches, determine the resolution of the image needed to extract regular patches
    #Pad image if necessary
    nr = img.shape[0] #rows/y
    nc = img.shape[1] #cols/x
    nd = img.shape[2] #color depth

    if stride != 0:
        #If image is smaller than window size, pad to fit.
        #else, pad to the least integer multiple of stride
        #Window shape is assumed to multiple of stride.
        #Find the smallest multiple of stride that is greater than image dimensions
        leastRowStrideMultiple = (np.ceil(nr / stride) * stride).astype(np.uint16)
        leastColStrideMultiple = (np.ceil(nc / stride) * stride).astype(np.uint16)
        #If image is smaller than window, pad to window shape. Else, pad to least stride multiple.
        nrPad = max(window_shape[0], leastRowStrideMultiple) - nr
        ncPad = max(window_shape[1], leastColStrideMultiple) - nc
        #Add Stride border around image, and nrPad/ncPad to image to make sure it is divisible by stride.
        stridePadding = int(stride / 2)
        #TEST: Turning off stride padding
        stridePadding = 0
        paddingRow = (stridePadding, nrPad + stridePadding)
        paddingCol = (stridePadding, ncPad + stridePadding)
        padding = (paddingRow, paddingCol, (0, 0))
        imgPadded = np.pad(img, padding, 'constant')

        patches = skimage.util.view_as_windows(imgPadded, window_shape, stride)
        nR, nC, nD, H, W, D = patches.shape
        nWindow = nR * nC * nD
        patches = np.reshape(patches, (nWindow, H, W, D))
    else:
        patches = np.reshape(img, (1, nr, nc, nd))

    return patches

def create_unaugmented_data_from_image(img, mask):    
    #Normalize inputs.
    img_pre = img.astype('float32')
    img_pre = preprocess_input(img_pre)
    
    img_resize = img_pre[np.newaxis,:,:,np.newaxis]
    if mask is None:
        mask_resize = None
    else:
        mask_resize = mask[np.newaxis,:,:,np.newaxis]
    
    return img_resize, mask_resize

def create_unaugmented_data_patches_from_image(img, mask, window_shape=(512, 512), stride=64):    
    #Normalize inputs.
    img_patches = extract_patches(img.astype('float32'), window_shape, stride)
    img_pre = preprocess_input(img_patches)
    img_reshaped = img_pre[:,:,:,np.newaxis]
    
    if mask is None:
        mask_reshaped = None
    else:
        mask_patches = extract_patches(mask.astype('float32'), window_shape, stride)
        mask_reshaped = mask_patches[:,:,:,np.newaxis]
    
    return img_reshaped, mask_reshaped

def create_unaugmented_data_patches_from_rgb_image(img, mask, window_shape=(512, 512, 3), stride=64, mask_channels=3):
    #Normalize inputs.
    img_patches = extract_rgb_patches(img.astype('float32'), window_shape, stride)
    img_pre = preprocess_input(img_patches)
    img_reshaped = img_pre[:,:,:,:]
    
    if mask is None:
        return img_reshaped
    else:
        mask_patches = extract_rgb_patches(mask.astype('float32'), window_shape, stride)
        mask_reshaped = mask_patches[:,:,:,0:mask_channels]
    
        return img_reshaped, mask_reshaped

def imgaug_generator_patched(batch_size=1, img_size=640, patch_size=512, patch_stride=64, steps_per_epoch=8000):
    id_str = str(img_size) + '_' + str(patch_size) + '_' + str(patch_stride)
    train_data_path = 'data/train_patched_dual_' + id_str
    #temp_path = 'temp/train_patched_dual_' + id_str
    #if not os.path.exists(temp_path):
    #    os.mkdir(temp_path)
    curriculum = ["Kangerlussuaq", "Kangiata-Nunaata", "Rink-Isbrae", "Upernavik", "Kong-Oscar", "Jakobshavn", "Hayes", ""] #ordered in terms of difficulty/iou_score
    curriculum = [""] #curriculum not really working
    curriculum_max_length = len(curriculum) - 1
    source_counter = 0
    source_limit = 0
    images_per_epoch = batch_size * steps_per_epoch
    images_per_metabatch = 16
    augs_per_image = 4

    augs = aug_daniel_prepadded(image_size=patch_size)
    counter = 0
    while True:
        returnCount = 0
        batch_img = None
        batch_mask = None

        #Process up to <images_per_metabatch> images in one batch to maitain randomness
        for i in range(images_per_metabatch):
            #Load images, resetting source "iterator" when reaching the end
            if source_counter == source_limit:
                curriculum_index = int(min(np.floor(float(returnCount) / float(images_per_epoch)), curriculum_max_length))
                images = glob.glob(train_data_path + '/' + curriculum[curriculum_index] + '*.png')
                images = list(filter(lambda x: '_mask' not in x, images))
                shuffle(images)
                source_counter = 0
                source_limit = len(images)
            image_name = images[source_counter]
            image_name = image_name.split(os.path.sep)[-1]
            image_mask_name = image_name.split('.')[0] + '_mask.png'
            
            #Work around for Unicode characters not being read by cv2 imread (works for skimage imageio)
            img_stream = open(os.path.join(train_data_path, image_name), "rb")
            img_bytes_array = bytearray(img_stream.read())
            img_np_bytes_array = np.asarray(img_bytes_array, dtype=np.uint8)
            img_3_uint16 = cv2.imdecode(img_np_bytes_array, cv2.IMREAD_UNCHANGED)[:,:,::-1]
            img_stream.close()
    
            mask_stream = open(os.path.join(train_data_path, image_mask_name), "rb")
            mask_bytes_array = bytearray(mask_stream.read())
            mask_np_bytes_array = np.asarray(mask_bytes_array, dtype=np.uint8)
            mask_3_uint16 = cv2.imdecode(mask_np_bytes_array, cv2.IMREAD_UNCHANGED)
            mask_stream.close()
        
            img_3_f64 = resize(img_3_uint16, (img_size, img_size), preserve_range=True)  #np.float64 [0.0, 65535.0]
            mask_3_f64 = resize(mask_3_uint16, (img_size, img_size), order=0, preserve_range=True) #np.float64 [0.0, 65535.0]
            
            source_counter += 1
            #print(source_counter, image_name)

            #Convert greyscale to RGB greyscale
            img_max = img_3_f64.max()
            img_min = img_3_f64.min()
            img_range = img_max - img_min
            mask_max = mask_3_f64.max()
            if (img_max != 0.0 and img_range > 255.0):
                img_3_f32 = (img_3_f64 / img_max * 255.0).astype(np.float32) #np.float32 [0.0, 255.0] Keep range 0-255 to conform to imagenet standards, but keep dtype float32 to keep precision
            else:
                img_3_f32 = img_3_f64.astype(np.float32)
            if (mask_max != 0.0):
                mask_3_f32 = np.floor(mask_3_f64 / mask_max * 255.0).astype(np.float32) #np.uint8 [0, 255]
            else:
                mask_3_f32 = mask_3_f64.astype(np.float32)

            #Run each image through 8 random augmentations per image
            for j in range(augs_per_image):
                #Augment image
                dat = augs(image=img_3_f32, mask=mask_3_f32)
                img_3_aug_f32 = dat['image'].astype('float32') #np.float32 [0.0, 255.0]
                mask_3_aug_f32 = dat['mask'].astype('float32') #np.float32 [0.0, 255.0]
                mask_final_f32 = mask_3_aug_f32 / 255.0 #np.float32 [0.0, 1.0]

                patches, maskPatches = create_unaugmented_data_patches_from_rgb_image(img_3_aug_f32, mask_final_f32, window_shape=(patch_size, patch_size, 3), stride=0, mask_channels=2)
                
                #patch_path = os.path.join(temp_path, image_name.split('.')[0] + "_" + str(j) + '.png')
                #patch_img = np.round((patches[0,:,:,:]+1)/2*65535).astype(np.uint16)
                #print(patch_img.max())
                #numpngw.write_png(patch_path, patch_img)
                #imsave(os.path.join(temp_path, image_name.split('.')[0] + "_" + str(j) + '_edge.png'), (255 * maskPatches[0,:,:,0]).astype(np.uint8))
                
                #Add to batches
                if batch_img is not None:
                    batch_img = np.concatenate((batch_img, patches)) #np.float32 [-1.0, 1.0], imagenet mean (~0.45)
                    batch_mask = np.concatenate((batch_mask, maskPatches))  #np.float32 [0.0, 1.0]
                    counter += 1
                else:
                    batch_img = patches
                    batch_mask = maskPatches
                    
        #Should have total of augs_per_image * images_per_metabatch to randomly choose from
        totalPatches = len(batch_img)
        #Now, return up <batch_size> number of patches, or generate new ones if exhausting curent patches
        #Shuffle
        idx = np.random.permutation(len(batch_img))
        if (len(batch_img) != len(batch_mask)):
            print('batch img/mask mismatch!')
            continue
        batch_img = batch_img[idx]
        batch_mask = batch_mask[idx]
        while returnCount + batch_size < totalPatches:
            batch_image_return = batch_img[returnCount:returnCount+batch_size,:,:,:]
            batch_mask_return = batch_mask[returnCount:returnCount+batch_size,:,:,:]
            returnCount += batch_size
            yield (batch_image_return, batch_mask_return)
            
if __name__ == '__main__':
    train_generator = imgaug_generator_patched(1, img_size=512, patch_size=448, patch_stride=32)
    for i in range(1):
        next(train_generator)
#    train_generator = imgaug_generator(2, 512)
#    for i in range(25):
#        next(train_generator)
