
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import tensorflow as tf
import numpy as np

class GradientAscent(object):
    """
        Esta clase se encarga de ejecutar el algoritmo para obtener el resultado de la tecnica del 
        "Gradient Ascent" . Esta tecnica tiene como objetivo, apartir de una imagen aleatoria, obtener
        la imagen que maximize las salidas de la neurona estudiada.
    """
    def __init__(self, model,processor_function):
        self._model = model
        self._img = self.__initializeImage(tuple(self._model.input.shape[1:]),processor_function)
        return

    def __call__(self,output_index,*,
                 iterations : Optional[int] = 20,
                 learning_rate : Optional[float] = 10.0):
        #Variables locales
        input_image = self._img.copy()
        for iteration in range(iterations):
            input_image,loss  = self.gradient_ascent_step(input_image, output_index, learning_rate)
            #if loss <= 0: #Por x razones, el gradiente no puede crecer más
                #break;
        #return loss, self.__processImage(input_image.numpy()[0],0.5).round().astype(np.uint8)
        #from ..ImageUtils import ImageUtils
        #return loss, ImageUtils.processGradients(input_image.numpy()[0],0.3)
        return loss, self.__deprocess_image(input_image.numpy()[0])


    @tf.function #Sirve para hacer el codigo más rapido. Quitar si se quiere hacer debugging
    def gradient_ascent_step(self,img, output_index, learning_rate):
        """
            Este metodo es el responsable de computar el gradiente
        """
        img = tf.convert_to_tensor(img)
        with tf.GradientTape() as tape:
            tape.watch(tf.convert_to_tensor(img)) #La variable
            loss = self.compute_loss(img, output_index)
        # Compute gradients --> d(Loss) / d(img) . Derivada de la función 'loss' respecto a la variable(matriz de variables WxHx3) 'img'
        grads = tape.gradient(loss, img)
        # Normalize gradients.
        # +Info sobre la normalización L2:https://montjoile.medium.com/l0-norm-l1-norm-l2-norm-l-infinity-norm-7a7d18a4f40c
        grads = tf.math.l2_normalize(grads)
        img += learning_rate * grads
        return img, loss

    def compute_loss(self,input_image,output_index):
        """
            La función de coste va a ser la media del filtro obtenido en la salida
            estudiada de la capa elegida. Dado que la imagen pasada como input es un
            array que va creciendo con el gradiente obtenido, lo que se va obteniendo
            es la imagen idonea en la que el output del filtro es maximo.
        """
        filters = self._model(input_image)
        filter_activation = filters[:, :, :, output_index]
        return filter_activation#tf.reduce_mean(filter_activation)


    def __initializeImage(self,shape,processor_function):
        """
            Este metodo crea una imagen RGB aleatoria.
            Luego centramos esos valores entorno al gris (128) con ruido alrededor de forma
            que el algoritmo tenga camino para maximizar.
        """
        from ..ImageUtils import ImageUtils
        array = ImageUtils.randomArray(shape)
        array = array*20.0 + 128.0 #Multiplicamos para aumentar la dispersión y luego desplazamos entorno a 128.
        #array = processorFunction(array)
        #array = np.zeros(shape)
        return array[None,...]

    def __deprocess_image(self,x):
        x -= x.mean() #Centramos la media en 0
        x /= (x.std() + 1e-5) #♣Desviación en 1
        x *= 0.1 #Desviación en 0.1
        x += 0.5 #Media en 0.5
        x = np.clip(x, 0, 1) #Acabamos con valores outliers
        x *= 255 #Llevamos al rango de valores de una imagen
        x = x.astype(np.uint8) #Convertirmos en entero
        return x


