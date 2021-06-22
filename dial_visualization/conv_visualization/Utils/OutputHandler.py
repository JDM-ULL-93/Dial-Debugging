import numpy as np
import tensorflow as tf

#DIOS --> https://github.com/poloclub/cnn-explainer
#https://github.com/sar-gupta/convisualize_nb


from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from .Handler import Handler
class OutputHandler(Handler):
    #More Feature Visualizations : https://github.com/timsainb/tensorflow-2-feature-visualization-notebooks
    # Pay attention to "0.1-Visualize-receptive-fields-and-features.ipynb"
    def __init__(self, 
                 inputs: List['tf.python.keras.engine.keras_tensor.KerasTensor'], 
                 outputs : List['tf.python.keras.engine.keras_tensor.KerasTensor'],
                 preprocessorFunction):
        super(OutputHandler, self).__init__(self.doFeatureAndGradientAscent,OutputHandler)
        self.__model = tf.keras.Model(
            inputs = inputs,
            outputs = outputs
        )#keras.backend.function(input,[output])
        self.__preprocessorFunction = preprocessorFunction
        return

   

    def __call__(self,**kwargs):
        #Cargamos los argumentos esperados (que no tienen porque ser todos):
        who,dragMode=kwargs["who"],kwargs["dragMode"]
        #Proceso:
        if who in self._handlers:
            img = None
            img_array = kwargs["img_array"]
            if who == "Guided BackPropagation":
                img = self._handlers[who](self,img_array)
            elif who == "Score-CAM":
                classifier,max_N = kwargs["classifier"], kwargs["max_N"]
                img = self._handlers[who](self,classifier,img_array,max_N=max_N)
            else: #Custom one
                kwargs["self"] = self
                kwargs["cnn_model"] = self.__model
                kwargs["preprocessorFunction"] = self.__preprocessorFunction
                return self._handlers[who](**kwargs)
            layerName = kwargs["layerName"]
            who = "Output of:{} - {}".format(layerName,who)
            if dragMode:
                return img,who
            else:
                self.mathPlot([img],[who],[None],1,1)
        else: #Por defecto ejecutamos Feature Map junto a Gradient Ascent
            img_array,outputIndex = kwargs["img_array"],kwargs["outputIndex"]
            iterations,learning_rate =  kwargs["iterations"],kwargs["learning_rate"]
            feature,gradAsc,input_img = self.doFeatureAndGradientAscent(self,img_array,
                                                              outputIndex,
                                                              iterations=iterations,
                                                              learning_rate=learning_rate)
            if dragMode:
                return self.prepareImages([feature,gradAsc]),who
            else:
                from .ImageUtils import ImageUtils
                processed = ImageUtils.processGradients(self.__preprocessorFunction(input_img.copy()))
                self.mathPlot(
                [feature,gradAsc,input_img.round().astype(np.uint8),processed],
                ["Feature Map","Gradient Ascent","Original Image","Input Image"],
                ["gray",None,None,None],
                gridRows = 2,
                gridCols = 2)
        return
    ############################################################################################ 
    ############################################################################################
    ##HANDLERS
    @staticmethod
    def doFeatureAndGradientAscent(self,
                                   img_array:np.ndarray,
                                   outputIndex:int,
                                   *,
                                   iterations : int = 20,
                                   learning_rate : float = 1000.0) -> np.ndarray:

            feature = self.getSingleFeatureMap(img_array,outputIndex)
            _,gradAsc = self.doGradientAscent(outputIndex,iterations=iterations,learning_rate=learning_rate)
            return feature,gradAsc,img_array[0]

    def __loadData(self, img_array : np.ndarray) ->np.ndarray: ## Simple Feature Map
        """
            Procesa la imagen aplicando convolución por todas las capas convolucionales y devuelve el resultado
            de esta cadena de convoluciones, normalmente una imagen de dimensión igual o reducida donde cada canal
            ahora representa una "potencial" imagen en escala de grises para la salida de una de las neuronas de la capa.
            Potencial por que la imagen debe tratarse antes para entrar dentro del rango [0,255] y poder ser visualizada.
        """
        return self.__model(img_array)[0]

    def getSingleFeatureMap(self,
                            img_array: np.ndarray,
                            index : int) -> np.ndarray:
        """
            ToDo: Aplicar un mapeado a img_no_activation para que los valores negativos (<127.5 cuando escalados)
            aparezcan en rojo y los valores positivos (>=127.5 cuando escalados) aparezcan en azul
            Cuanto más cerca del umbral, más claros, cuanto menos, mas oscuros.
            MIRAR seismic color map de mathplotlib
            MIRAR https://stackoverflow.com/questions/54327272/creating-custom-colormap-for-opencv-python para incorporar
                  custom colormaps a cv2 
        """
       

        #lastLayer = self._OutputHandler__model.layers[-1]
        #if hasattr(lastLayer,"activation"):
            #activation = lastLayer.activation
            #lastLayer.activation = None
        #img_no_activation = self.__loadData(img_array)[:, :, index]
        img_activation = self.__loadData(img_array)[:, :, index]
        #img_activation = activation(img_no_activation)
        #if hasattr(lastLayer,"activation"):
            #lastLayer.activation = activation
        from .ImageUtils import ImageUtils
        #img_no_activation = ImageUtils.scaleValues(img_no_activation.numpy(),(0,255)).round()
        #img_no_activation = ImageUtils.grayToRGB(img_no_activation)
        img_activation = ImageUtils.scaleValues(img_activation.numpy(),(0,255)).round()
        img_activation = ImageUtils.grayToRGB(img_activation)
        #import cv2
        #img_no_activation = cv2.applyColorMap(img_no_activation, cv2.COLORMAP_VIRIDIS)
        return img_activation#ImageUtils.composeImages([img_no_activation,img_activation],img_no_activation.shape,margin=2)

    ###########################################################
    ## Gradient Ascent
    def doGradientAscent(self,
                         outputIndex : int,
                         *,
                         iterations : int = 20,
                         learning_rate : float = 10.0 ) -> np.ndarray:
        from .Algorithms.GradientAscent import GradientAscent
        return GradientAscent(self.__model,self.__preprocessorFunction)(outputIndex,
                                            iterations = iterations,
                                            learning_rate = learning_rate)
    ###########################################################
    ## ScoreCam
    @staticmethod
    def doScoreCam(self,
                   model : tf.keras.Model,
                   img_array : np.ndarray,
                   *,
                   max_N:int=-1) -> np.ndarray:
        from .Algorithms.ScoreCam import ScoreCam
        return ScoreCam(self.__model,self.__preprocessorFunction)(model,
                                      img_array,
                                      max_N=max_N)

    ###########################################################
    ## Guided BackPropagation
    #https://github.com/nguyenhoa93/cnn-visualization-keras-tf2
    @staticmethod
    def doGuidedBackPropagation(self,img : np.ndarray) ->np.ndarray:
        from .Algorithms.GuidedGradientDescent import GuidedGradientDescent
        return GuidedGradientDescent(self.__model,self.__preprocessorFunction)(img)

    ###########################################################
    ## Image operations  
    def prepareImages(self,imgs:List[np.ndarray],margin :int = 5):
        input_shape = (self.__model.input.shape[1],self.__model.input.shape[2] )# OBtenemos las dimensiones (width,height)
        from .ImageUtils import ImageUtils
        return ImageUtils.composeImages(imgs,input_shape,margin=margin)

    ##############################################
