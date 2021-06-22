import tensorflow as tf
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
class IntegratedGradients(object):
    """
        #https://www.tensorflow.org/tutorials/interpretability/integrated_gradients
    """
    def __init__(self, model : tf.keras.Model, preprocessorFunction : 'Function'):
        self._copyModel = tf.keras.models.clone_model(model, clone_function = self.__removeSoftMax)
        self._copyModel.set_weights(model.get_weights())
        self.__model = self._copyModel #model
        self.__preprocessor = preprocessorFunction
        return

    def __removeSoftMax(self,layer):
        """
            Queremos que el gradiente se calcule con los "logits", esto es, el output de la red antes de ser procesado por
            softmax.
        """
        newLayer = layer.__class__.from_config(layer.get_config())
        if hasattr(newLayer,"activation") and newLayer.activation == tf.keras.activations.softmax:
                newLayer.activation = tf.keras.activations.linear #No computa nada, deja pasar los valores --> f(x) = x
        return newLayer

    def __call__(self,
                 img_array, 
                 class_index,
                 *,
                 num_steps : Optional[int] = 50,
                 batch_size : Optional[int] = 32):
        image = self.__preprocessor(img_array.copy())[0]
        attributions  = self.algorithm(image,class_index,num_steps = num_steps, batch_size=batch_size)
        attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)
        from ..ImageUtils import ImageUtils
        #normalized_attribution_mask = ImageUtils.normalize(attribution_mask.numpy())
        #import cv2
        #return ImageUtils.overlay(img_array[0],1-normalized_attribution_mask,emphasize=False,hif=1.75,overlay_colormap=cv2.COLORMAP_DEEPGREEN)
        return ImageUtils.grayToRGB( ImageUtils.processGradients(attribution_mask.numpy()) )
    
    @tf.function
    def algorithm(self,
                    image,
                    class_index,
                    num_steps : Optional[int] = 50,
                    batch_size : Optional[int] = 32):

        # 0. Generamos la imagen "baseline" con las mismas dimensiones que la imagen original
        baseline = tf.zeros(image.shape)
        # 1. Generamos array de longitud 'num_steps+1' relleno con valores 'alpha'(valores dentro del rango [0,1])
        alphas = tf.linspace(start=0.0, stop=1.0, num=num_steps+1)

        # Reservamos en memoria el espacio para el array que contendrá los gradientes de la función de predicción
        # de nuestro modelo (ocasionalmente softmax por ejemplo) respecto a cada carecteristica(==pixel) de la imagen interpolada.
        gradient_batches = tf.TensorArray(tf.float32, size=num_steps+1)

        # Iteramos por los elementos en alpha y lo ejecutamos en "lotes" para mejorar la velocidad y la eficiencia de memoria, de forma
        # que el algoritmo implementado sea escalable a "num_steps" mas altos (es decir, evitamos procesar el algoritmo de una pasada)
        for from_ in range(0, alphas.shape[0], batch_size): #Esta forma de range sirve para iterar de 0 hasta "batch_size" pero sin sobrepasar len(laphas)
            to = tf.minimum(from_ + batch_size, alphas.shape[0]) #Nos aseguramos que no nos salimos de rango
            alpha_batch = alphas[from_:to] #Recogemos el lote de valores "alphas"

            # 2. Generamos las interpolaciones entre la imagen original(image) y el "baseline" en función de los alphas indicados
            interpolated_path_input_batch = self._interpolate_images(baseline,
                                                               image,
                                                               alpha_batch)
            # 3. Calculamos el gradiente anteriormente mencionados.(Gradiente de la función de predicción respecto a las imagenes interpoladas)
            # Y de ese gradiente, seleccionamos los gradientes que correspondan con el resultado para la clase estudiada 'class_index'
            gradient_batch = self._compute_gradients(images=interpolated_path_input_batch,
                                               class_index=class_index)

            # Guardamos los resultados dentro del array reservado anteriormente, insertandolos en la posición "from" hasta "to"
            gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch) 
        
        # Stack path gradients together row-wise into single tensor.
        total_gradients = gradient_batches.stack() #Recupera los tensores guardados dentro del "TensorArray"

        # 4. Calculamos la integral (aproximada usando el principio de la Suma de Trapecios de Riemann) de todos los gradientes
        avg_gradients = self._integral_approximation(total_gradients)

        # 5. Scale integrated gradients with respect to input.
        # Restamos la imagen original con el baseline y el resultado de esa diferencia lo multiplicamos por 'avg_gradients' 
        # que en este contexto se puede interpretar como una matrix de pesos que indican la importancia del pixel, asociado 
        # a la misma posición, en la clasificación de la clase estudiada.
        integrated_gradients = (image - baseline) * avg_gradients

        return integrated_gradients

    def _interpolate_images(self,
                             baseline,
                             image,
                             alphas):
        """
            La idea de este metodo es la combinación de una imagen, normalmente una
            totalmente oscura, (baseline) de igual dimensiones a la imagen original(image)
            y combinarlas. ¿Como? Sumando cada pixel de la imagen "baseline" al resultado
            de la diferencia entre cada pixel de imagen original "image" y la imagen "baseline",
            y al resultado de esa diferencia, multiplicado por un escalar (alpha) que es un valor
            entre 0 y 1 para indicar la proporción de la diferencia que es sumado a la imagen "baseline"

            Este metodo lo que hace es repetir la operación anterior varias veces por cada valor alpha en 
            el array "alphas", de forma que devuelve un "batch" de imagenes "baseline"
        """
        #Ampliamos las dimensiones de los escalares para ser matrices 3D multiplicables por otras matrices(imagenes) 3D
        alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
        delta = image - baseline #(x - x')
        images = baseline +  alphas_x * delta #x + a_i(x - x') for a_i in alphas
        return images

    def _compute_gradients(self,
                            images,
                            class_index):
        """
            Computa el gradiente de la función de probabilidad softmax de la clase "class_index"
            respecto a la variable matrix "imagen interpolada"
        """
        with tf.GradientTape() as tape:
            tape.watch(images)
            logits = self.__model(images)
            probs = tf.nn.softmax(logits, axis=-1)[:, class_index]
        return tape.gradient(probs, images)

    def _integral_approximation(self,
                                 gradients):
        # riemann_trapezoidal: Una aproximación de la integral
        # Se suman los elementos vecinos a distancia 1 y se dividen por 2
        # (es decir, se suma el 1º con el 2º y se divide por 2, el 2º con el 3º y se divide por 2, ...)
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        #Se reduce la dimensionalidad calculando la media entre los N gradientes
        integrated_gradients = tf.math.reduce_mean(grads, axis=0) #Calculamos la media
        return integrated_gradients
