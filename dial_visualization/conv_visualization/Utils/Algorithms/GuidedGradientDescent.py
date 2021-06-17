import tensorflow as tf

class GuidedGradientDescent(object):
    """
        Esta clase se encarga de ejecutar la tecnica "Guided Gradient Descent" o "Guided BackPropagation".
        Esta tecnica se basa en, apartir del output de la última capa convolucional seleccionada y el calculo del 
        gradiente para cada pixel i,j de la imagen original, reconstruir la imagen original. Realmente lo novedoso aqui es
        que se modifique ligeralmente una parte del calculo del gradiente, especificamente en la función ReLu para que, en el 
        proceso del backpropagation, los valores negativos que viajaban hacia atrás con el gradiente natural función ReLu en función
        de la variable de entrada X(i,j,c) (que representa un pixel especifico de la imagen):
        ReLu:{
            return 0 if x <= 0
            return x if x > 0
        }

        d(ReLu):{
            return 0 if x <= 0
            return 1 if x > 0
        }
        
        Dejan de hacerlo con la modificación siguiente que tiene en cuenta el valor del gradiente de la capa anterior:
        d(ReLu):{
            return 0 if x <= 0 or dy <= 0
            return 1 if x > 0 or dy > 0
        }

        Con lo cual se obtiene una imagen con menos ruido y mas definida. (Se puede probar a modificar esta parte 

    """
    def __init__(self, model,processorFunction):
        self.__model = tf.keras.models.clone_model(model, clone_function = self.__insertGuidedReLu)
        self.__model.set_weights(model.get_weights())
        self.__processorFunction = processorFunction
        return

    def __call__(self, img):
        processed_img = self.__processorFunction(img.copy())
        with tf.GradientTape() as tape:
            inputs = tf.cast(processed_img, tf.float32)
            tape.watch(inputs)
            #FastForward
            outputs = self.__model(inputs)

        grads = tape.gradient(outputs, inputs)[0]
        from ..ImageUtils import ImageUtils
        return ImageUtils.processGradients(grads.numpy())

    def __insertGuidedReLu(self,layer):
        newLayer = layer.__class__.from_config(layer.get_config())
        if hasattr(newLayer,"activation") and newLayer.activation == tf.keras.activations.relu:
                newLayer.activation = GuidedGradientDescent.guidedReLu
        return newLayer

    #https://glassboxmedicine.com/2019/10/06/cnn-heat-maps-gradients-vs-deconvnets-vs-guided-backpropagation/
    @staticmethod
    @tf.custom_gradient
    def guidedReLu(x):
        """
            Este metodo lo que hace es modificar el gradiente de la función Relu, de forma que
            en el fast-forward se sigue utilizando la función:
                return 0 if x <= 0
                return x if x > 0
            Pero en el backpropagation, su gradiente ya no es simplemente la derivada de lo anterior:
                return 0 if x <= 0
                return 1 if x > 0
            Como nos interesa reconstruir la imagen apartir de la salida, la anterior derivada
            permitia propagar hacia atras valores negativos (Vanilla BackPropagation). Asi que,
            de la necesidad de evitar eso, ampliamos la derivada ReLu anterior:
                return 0 if x <= 0
                return 1 if x > 0
                return 0 if dy <= 0
                return 1 if dy > 0
            Y a esta nueva forma del gradiente es lo que se conoce como (Guided ReLu BackPropagation)
        """
        def grad(dy):
            return tf.cast(x>0, "float32") * tf.cast(dy>0,"float32") * dy
        return tf.nn.relu(x), grad
