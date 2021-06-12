from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
class GradCAM(object):
    """description of class"""
    def __init__(self, model,preprocessorFunction):
        self._model = model
        self._preprocessorFunction = preprocessorFunction
        return

    def __findLastConvLayer(self,model):
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                return layer
        return None#raise ValueError("Could not find 4D layer. Cannot apply GradCAM")

    def __call__(self,
                 img_array, 
                 predictIndex):
        #https://glassboxmedicine.com/2020/05/29/grad-cam-visual-explanations-from-deep-networks/
        softmaxOutput = self._model.output
        processed = self._preprocessorFunction(img_array.copy())
        convLayerOutput = self.__findLastConvLayer(self._model).output
        # Input Shape Format: (Nº Samples, Width, Height, Channels)
        expectedshape = (self._model.inputs[0].shape[1],self._model.inputs[0].shape[2])

        import tensorflow as tf
        gradModel = tf.keras.Model(
            inputs = self._model.inputs,
            outputs = [convLayerOutput,softmaxOutput]
        )
        with tf.GradientTape() as tape:
            inputs = tf.cast(processed, tf.float32)
            (convOuts, preds) = gradModel(inputs)  # preds after softmax
            if predictIndex < 0: #ToDo: Documentar que si el indice es < a 0, lo que hace es buscar el top 1
                predictIndex = np.argmax(preds)
            loss = preds[:, predictIndex]

        # compute gradients with automatic differentiation
        grads = tape.gradient(loss, convOuts) #Gradiente de loss en relación a cada pixel i,j de convOuts
        # discard batch
        convOuts = convOuts[0]
        grads = grads[0]
        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(tf.keras.backend.epsilon()))
        # compute weights
        weights = tf.reduce_mean(norm_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1)
        # Apply reLU
        import numpy as np
        cam = np.maximum(cam, 0)
        #cam = cam / (np.max(cam)+1e-07)
        from ..ImageUtils import ImageUtils
        cam = ImageUtils.normalize(cam)
        # convert to 3D
        cam3 = cam[...,None]# (W,H) --> (W,H,1) #old way: cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3]) # (W,H,1) --> (W,H,3)
        #return self.overlay(img_array[0],cam3,inverted=True),"Grad-Cam.SoftMax Index:{}".format(predictIndex)
        return ImageUtils.overlay(img_array[0],cam3,emphasize = True)

