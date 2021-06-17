import numpy as np
from .Handler import Handler
from .ImageUtils import ImageUtils

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, Callable, Tuple
class KernelHandler(Handler):
    def __init__(self, kernel, outputShape=(224,224)):
        super(KernelHandler, self).__init__(None,KernelHandler)
        self._outputShape : Tuple[int,int] = outputShape
        return

    def __call__(self,**kwargs):
        #Cargamos los argumentos esperados (que no tienen porque ser todos):
        who,dragMode,kernel=kwargs["who"],kwargs["dragMode"],kwargs["kernel"]
        #Proceso:
        if who in self._handlers: #Custom handlers
            kwargs["self"] = self
            kwargs["outputShape"] = self._outputShape
            return self._handlers[who](**kwargs)
        if dragMode:
            if kernel.shape[2] == 3:
                return self._handlers["3ShapeDragMode"](self,kernel),who
            else:
                return self._handlers["XShapeDragMode"](self,kernel),who
        else:
            if kernel.shape[2] == 3:
                return self._handlers["3ShapePlotMode"](self,kernel,who)
            else:
                return self._handlers["XShapePlotMode"](self,kernel,who)
        return
    

    ######################################
    ##HANDLERS:
    @staticmethod
    def RGBShapeDragMode(self,kernel : np.ndarray) ->np.ndarray:
        #Variables locales
        shape = self._outputShape 
        #Proceso
        result = ImageUtils.scaleValues(kernel,(0,255)).round().astype(np.uint8)
        import cv2
        result = cv2.resize(result,shape,interpolation=cv2.INTER_NEAREST)
        return result

    @staticmethod
    def XShapeDragMode(self,kernel : np.ndarray) ->np.ndarray:
        #Variables locales
        shape = self._outputShape
        #Proceso
        result = np.mean(kernel,axis=2)
        result = ImageUtils.scaleValues(result,(0,255)).round().astype(np.uint8)
        result = ImageUtils.grayToRGB(result)
        import cv2
        result = cv2.resize(result,shape,interpolation=cv2.INTER_NEAREST)
        return result

    @staticmethod
    def RGBShapePlotMode(self,kernel : np.ndarray,who:str)->np.ndarray:
        #Proceso
        result = ImageUtils.scaleValues(kernel,(0,255)).round().astype(np.uint8)
        self.mathPlot([result],[who],["gist_rainbow"])
        return

    @staticmethod
    def XShapePlotMode(self,kernel : np.ndarray,who:str)->np.ndarray:
        #Proceso
        result = np.mean(kernel,axis=2)
        who = "Average of {} channels".format(kernel.shape[2])
        self.mathPlot([result],[who],["gray"])
        return

   

 