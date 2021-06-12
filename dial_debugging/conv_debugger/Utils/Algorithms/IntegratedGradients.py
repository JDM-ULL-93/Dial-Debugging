import tensorflow as tf
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
class IntegratedGradients(object):
    """
        #https://www.tensorflow.org/tutorials/interpretability/integrated_gradients
    """
    def __init__(self, model : tf.keras.Model, preprocessorFunction : 'Function'):
        self.__model = model
        self.__preprocessor = preprocessorFunction
        return

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
        normalized_attribution_mask = ImageUtils.normalize(attribution_mask.numpy())
        import cv2
        return ImageUtils.overlay(img_array[0],1-normalized_attribution_mask,emphasize=False,hif=1.,overlay_colormap=cv2.COLORMAP_DEEPGREEN)
    
    @tf.function
    def algorithm(self,
                               image,
                               class_index,
                               num_steps : Optional[int] = 50,
                               batch_size : Optional[int] = 32):

        # 0. Generamos la imagen "baseline" con las mismas dimensiones que la imagen original
        baseline = tf.zeros(image.shape)
        # 1. Generamos array de valores alpha que van de 0 hasta 1
        alphas = tf.linspace(start=0.0, stop=1.0, num=num_steps+1)

        # Reservamos en memoria el espacio para el array que contendrá los gradientes de la función de predicción
        # de nuestro modelo (ocasionalmente softmax por ejemplo) respecto a cada carecteristica(==pixel) de la imagen interpolada.
        gradient_batches = tf.TensorArray(tf.float32, size=num_steps+1)

        # Iteramos por los elementos en alpha y creamos "lotes" para mejorar la velocidad y la eficiencia de memoria, de forma
        # que el algoritmo implementado sea escalable a "num_steps" mas altos (es decir, evitamos procesar el algoritmo de 1)
        for from_ in range(0, alphas.shape[0], batch_size): #Esta forma de range sirve para iterar de 0 hasta "batch_size" pero sin sobrepasar len(laphas)
            to = tf.minimum(from_ + batch_size, alphas.shape[0]) #Nos aseguramos que no nos salimos de rango
            alpha_batch = alphas[from_:to] #Recogemos el lote de valores "alphas"

            # 2. Generamos las interpolaciones entre la imagen original(image) y el "baseline" en función de los alphas indicados
            interpolated_path_input_batch = self.__interpolate_images(baseline,
                                                               image,
                                                               alpha_batch)
            # 3. Calculamos el gradiente anteriormente mencionados.(Gradiente de la función de predicción respecto a las imagenes interpoladas)
            # Y de ese gradiente, seleccionamos los gradientes que correspondan con el resultado para la clase estudiada 'class_index'
            gradient_batch = self.__compute_gradients(images=interpolated_path_input_batch,
                                               class_index=class_index)

            # Guardamos los resultados dentro del array reservado anteriormente, insertandolos en la posición "from" hasta "to"
            gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch) 
        
        # Stack path gradients together row-wise into single tensor.
        total_gradients = gradient_batches.stack()

        # 4. Calculamos la integral (aproximada usando el principio de la Suma de Trapecios de Riemann) de todos los gradientes
        avg_gradients = self.__integral_approximation(total_gradients)

        # 5. Scale integrated gradients with respect to input.
        integrated_gradients = (image - baseline) * avg_gradients

        return integrated_gradients

    def __interpolate_images(self,
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

    def __compute_gradients(self,
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

    def __integral_approximation(self,
                                 gradients):
        # riemann_trapezoidal: Una aproximación de la integral se consigue mediante la media del
        # primer elemento y el último. A mas elemento se incoporen (segundo y penultimo, tercero y antepenultimo)
        # mas preciso es la aproximación
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        integrated_gradients = tf.math.reduce_mean(grads, axis=0) #Calculamos la media
        return integrated_gradients
