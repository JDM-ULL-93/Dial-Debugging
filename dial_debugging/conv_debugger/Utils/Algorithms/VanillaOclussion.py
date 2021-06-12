
import tensorflow as tf
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
class VanillaOclussion(object):
    """description of class"""
    def __init__(self, model,preprocessorFunction):
        self._model = model
        self._preprocessorFunction = preprocessorFunction
        return

    def __call__(self,
                 img_array, 
                 class_index,
                 *,
                 occluding_size=8,
                 occluding_pixel=0,
                 occluding_stride=8):
        return self.algorithm(img_array,class_index,occluding_size=occluding_size,occluding_pixel=occluding_pixel,occluding_stride=occluding_stride)

    tf.function
    def algorithm(self,
                   img_array,
                   class_index,
                   *,
                   occluding_size=8,
                   occluding_pixel=0,
                   occluding_stride=8):
        inputSize = self._model.layers[0].output.shape[1:3]

        imgTensor = img_array.copy()
        _,height, width, channels = imgTensor.shape
        import math
        output_height = int(math.ceil((height-occluding_size)/occluding_stride+1))
        output_width = int(math.ceil((width-occluding_size)/occluding_stride+1))
  
        import multiprocessing
        cores = multiprocessing.cpu_count()
        import numpy as np
        input_images = np.array([np.zeros((height,width,channels))]*(output_height*output_width)) 
        ###################################
        def prepareInputImagesProc(input_images,h,w,height,width,output_width,preprocessorFunction,imgTensor, occluding_stride,occluding_pixel,occluding_size):
            #occluder region
            h_start = h*occluding_stride
            w_start = w*occluding_stride
            h_end = min(height, h_start + occluding_size)
            w_end = min(width, w_start + occluding_size)

            input_image = np.array(imgTensor, copy=True) 
            input_image[:,h_start:h_end,w_start:w_end,:] = occluding_pixel
            input_image = preprocessorFunction(input_image)
            input_images[h*output_width+w] = input_image
            return
        ###################################
       
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=cores) as executor:
            for h in range(output_height):
                for w in range(output_width):
                    executor.submit(prepareInputImagesProc,
                                    input_images,
                                    h,w,
                                    height,width,
                                    output_width,
                                    self._preprocessorFunction,
                                    imgTensor,
                                    occluding_stride,
                                    occluding_pixel,
                                    occluding_size)
        executor.shutdown(wait=True)
        probs = self._model.predict(input_images)
        saliencyCoeffs = np.zeros((output_height, output_width))
        for h in range(output_height):
            for w in range(output_width):
                saliencyCoeffs[h,w] = 1 - probs[h*output_width+w][class_index] # the probability of the correct class

        saliencyCoeffs = np.stack((saliencyCoeffs,)*3, axis=-1) #(W,H,1) --> (W,H,3)
        from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
        saliencyCoeffs = array_to_img(saliencyCoeffs)
        saliencyCoeffs = saliencyCoeffs.resize((inputSize[1], inputSize[0]), 1)
        saliencyCoeffs = img_to_array(saliencyCoeffs)
        from ..ImageUtils import ImageUtils
        return ImageUtils.overlay(img_array[0], ImageUtils.normalize(saliencyCoeffs), emphasize = False)


