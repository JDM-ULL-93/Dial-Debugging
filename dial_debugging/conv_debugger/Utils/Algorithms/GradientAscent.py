
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import tensorflow as tf
import numpy as np

class GradientAscent(object):
    """
        Esta clase se encarga de ejecutar el algoritmo para obtener el resultado de la tecnica del 
        "Gradient Ascent" . Esta tecnica tiene como objetivo, apartir de una imagen aleatoria, obtener
        la imagen que maximize las salidas de la neurona estudiada.
    """
    def __init__(self, model,processorFunction):
        self._model = model
        self._img = self.__initializeImage(tuple(self._model.input.shape[1:]),processorFunction)
        return

    def __call__(self,outputIndex,*,
                 iterations : Optional[int] = 20,
                 learning_rate : Optional[float] = 2000.0):
        #Variables locales
        input_image = self._img.copy()
        for iteration in range(iterations):
            input_image,loss  = self.__gradient_ascent_step(input_image, outputIndex, learning_rate)
            if loss <= 0: #Por x razones, el gradiente no puede crecer más
                break;
        #return loss, self.__processImage(input_image.numpy()[0],0.5).round().astype(np.uint8)
        from ..ImageUtils import ImageUtils
        return loss, ImageUtils.processGradients(input_image.numpy()[0],0.3)


    @tf.function #Sirve para hacer el codigo más rapido. Quitar si se quiere hacer debugging
    def __gradient_ascent_step(self,img, filter_index, learning_rate):
        """
            Este metodo es el responsable de computar el gradiente
        """
        with tf.GradientTape() as tape:
            tape.watch(img) #La variable
            loss = self.__compute_loss(img, filter_index)
        # Compute gradients --> d(Loss) / d(img)
        grads = tape.gradient(loss, img)
        # Normalize gradients.
        grads = tf.math.l2_normalize(grads)
        img += learning_rate * grads
        return img, loss

    def __compute_loss(self,input_image,filter_index):
        """
            La función de coste va a ser la media del filtro obtenido en la salida
            estudiada de la capa elegida. Dado que la imagen pasada como input es un
            array que va creciendo con el gradiente obtenido, lo que se va obteniendo
            es la imagen idonea para que el filtro maximize valores.
        """
        filters = self._model(input_image)
        filter_activation = filters[:, :, :, filter_index]
        return tf.reduce_mean(filter_activation)


    def __initializeImage(self,shape,processorFunction):
        """
            Este metodo crea una imagen RGB aleatoria.
            Luego centramos esos valores entorno al gris (128) con ruido alrededor de forma
            que el algoritmo tenga camino para maximizar.
        """
        from ..ImageUtils import ImageUtils
        array = ImageUtils.randomArray(shape)
        array = array*20.0 + 128.0 #Multiplicamos para aumentar la dispersión y luego desplazamos entorno a 128.
        #array = processorFunction(array)
        return array[None,...]


