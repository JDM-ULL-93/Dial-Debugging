import numpy as np
import tensorflow as tf

#DIOS --> https://github.com/poloclub/cnn-explainer
#https://github.com/sar-gupta/convisualize_nb


from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from .Handler import Handler
class OutputHandler(Handler):
    def __init__(self, 
                 inputs: List['tf.python.keras.engine.keras_tensor.KerasTensor'], 
                 outputs : List['tf.python.keras.engine.keras_tensor.KerasTensor'],
                 preprocessorFunction):
        Handler.__init__(self)
        self.__model = tf.keras.Model(
            inputs = inputs,
            outputs = outputs
        )#keras.backend.function(input,[output])
        self.__preprocessorFunction = preprocessorFunction
        self.__handlers = {
             "SimpleFeatureMap": self.getSingleFeatureMap,
             "Score-CAM": self.doScoreCam,
             "Guided BackPropagation" : self.doGuidedBackPropagation,
             "GradientAscent " : self.doGradientAscent
            }
        
        return

    

    def getHandler(self,who): 
        """
            De momento, la unica utilidad de este metodo es para poder preguntar por argumentos 
            opcionales desde el widget "conv_debugger_widget"
        """
        if who in self.__handlers:
            return self.__handlers[who]
        elif who in self.custom_handlers:
             return self.custom_handlers[who]
        return self.doFeatureAndGradientAscent

    def __call__(self,**kwargs):
        #Cargamos los argumentos esperados (que no tienen porque ser todos):
        who,dragMode=kwargs["who"],kwargs["dragMode"]
        #Proceso:
        if who in self.custom_handlers: #Custom Handler
            kwargs["cnn_model"] = self.__model
            kwargs["preprocessorFunction"] = self.__preprocessorFunction
            return self.__custom_handlers[who](**kwargs)
        elif who in self.__handlers:
            #return self.__handlers[who]
            img = None
            img_array = kwargs["img_array"]
            if who == "Guided BackPropagation":
                img = self.__handlers[who](img_array)
            else: #who == "Score-CAM"
                classifier,max_N = kwargs["classifier"], kwargs["max_N"]
                img = self.__handlers[who](classifier,img_array,max_N=max_N)
            layerName = kwargs["layerName"]
            who = "Output of:{} - {}".format(layerName,who)
            if dragMode:
                return img,who
            else:
                self.mathPlot([img],[who],[None],1,1)
        else: #Por defecto ejecuta Feature Map y Gradient Ascent
            img_array,outputIndex = kwargs["img_array"],kwargs["outputIndex"]
            iterations,learning_rate =  kwargs["iterations"],kwargs["learning_rate"]
            feature,gradAsc,input_img = self.doFeatureAndGradientAscent(img_array,
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
    def doFeatureAndGradientAscent(self,
                                   img_array:np.array,
                                   outputIndex:int,
                                   *,
                                   iterations : Optional[int] = 20,
                                   learning_rate : Optional[float] = 2000.0):
            feature = self.getSingleFeatureMap(img_array,outputIndex)
            _,gradAsc = self.doGradientAscent(outputIndex,iterations=iterations,learning_rate=learning_rate)
            from .ImageUtils import ImageUtils
            input_img = img_array#ImageUtils.processGradients(self.__preprocessorFunction(img_array.copy()))
            return feature,gradAsc,input_img[0]

    def __loadData(self, img_array : np.array): ## Simple Feature Map
        """
            Procesa la imagen aplicando convolución por todas las capas convolucionales y devuelve el resultado
            de esta cadena de convoluciones, normalmente una imagen de dimensión igual o reducida donde cada canal
            ahora representa una "potencial" imagen en escala de grises para la salida de una de las neuronas de la capa.
            Potencial por que la imagen debe tratarse antes para entrar dentro del rango [0,255] y poder ser visualizada.
        """
        return self.__model(img_array)[0]

    def getSingleFeatureMap(self,
                            img_array: np.array,
                            index : int) -> np.array:
        img = img_array.copy() #self.__preprocessorFunction(img_array.copy())
        img = self.__loadData(img)[:, :, index]
        from .ImageUtils import ImageUtils
        img = ImageUtils.scaleValues(img.numpy(),(0,255)).round()
        img = ImageUtils.grayToRGB(img)
        return img

    ###########################################################
    ## Gradient Ascent 
    def doGradientAscent(self,
                         outputIndex,
                         *,
                         iterations : Optional[int] = 20,
                         learning_rate : Optional[float] = 2000.0 ):
        from .Algorithms.GradientAscent import GradientAscent
        return GradientAscent(self.__model,self.__preprocessorFunction)(outputIndex,
                                            iterations = iterations,
                                            learning_rate = learning_rate)
    ###########################################################
    ## ScoreCam
    def doScoreCam(self,
                   model,
                   img_array,
                   *,
                   max_N=-1):
        from .Algorithms.ScoreCam import ScoreCam
        return ScoreCam(self.__model,self.__preprocessorFunction)(model,
                                      img_array,
                                      max_N=max_N)

    ###########################################################
    ## Guided BackPropagation
    #https://github.com/nguyenhoa93/cnn-visualization-keras-tf2
    def doGuidedBackPropagation(self,img):
        from .Algorithms.GuidedGradientDescent import GuidedGradientDescent
        return GuidedGradientDescent(self.__model,self.__preprocessorFunction)(img)

    ###########################################################
    ## Image operations  
    def prepareImages(self,imgs,margin = 5):
        input_shape = (self.__model.input.shape[1],self.__model.input.shape[2] )# OBtenemos las dimensiones (width,height)
        from .ImageUtils import ImageUtils
        return ImageUtils.composeImages(imgs,input_shape,margin=margin)

    ##############################################
