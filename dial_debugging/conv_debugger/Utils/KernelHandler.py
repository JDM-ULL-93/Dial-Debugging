import numpy as np
from .Handler import Handler
from .ImageUtils import ImageUtils

class KernelHandler(Handler):
    def __init__(self, kernel, outputShape=(224,224)):
        Handler.__init__(self)
        self._outputShape = outputShape
        self.__handlers = {
             "3ShapeDragMode": self.RGBShapeDragMode,
             "XShapeDragMode": self.XShapeDragMode,
             "3ShapePlotMode" : self.RGBShapePlotMode,
             "XShapePlotMode" : self.XShapePlotMode
            }

        self.__kernel = None
        return


    def __call__(self,**kwargs):
        #Cargamos los argumentos esperados (que no tienen porque ser todos):
        who,dragMode,kernel=kwargs["who"],kwargs["dragMode"],kwargs["kernel"]
        #Proceso:
        self.__kernel = kernel
        if who in self.custom_handlers: #Custom handlers
            kwargs["outputShape"] = self._outputShape
            return self.__custom_handlers[who](**kwargs)
        if dragMode:
            if kernel.shape[2] == 3:
                return self.__handlers["3ShapeDragMode"](who)
            else:
                return self.__handlers["XShapeDragMode"](who)
        else:
            if kernel.shape[2] == 3:
                return self.__handlers["3ShapePlotMode"](who)
            else:
                return self.__handlers["XShapePlotMode"](who)
        return
    

    ######################################
    ##HANDLERS:
    def RGBShapeDragMode(self,who):
        #Variables locales
        kernel = self.__kernel
        shape = self._outputShape 
        #Proceso
        result = ImageUtils.scaleValues(kernel,(0,255)).round().astype(np.uint8)
        import cv2
        result = cv2.resize(result,shape,interpolation=cv2.INTER_NEAREST)
        return result,who

    def XShapeDragMode(self,who):
        #Variables locales
        kernel = self.__kernel
        shape = self._outputShape
        #Proceso
        result = np.mean(kernel,axis=2)
        result = ImageUtils.scaleValues(result,(0,255)).round().astype(np.uint8)
        result = ImageUtils.grayToRGB(result)
        import cv2
        result = cv2.resize(result,shape,interpolation=cv2.INTER_NEAREST)
        return result,who

    def RGBShapePlotMode(self,who):
        #Variables locales
        kernel = self.__kernel
        #Proceso
        result = ImageUtils.scaleValues(kernel,(0,255)).round().astype(np.uint8)
        self.mathPlot([result],[who],["gist_rainbow"])
        return

    def XShapePlotMode(self,who):
        #Variables locales
        kernel = self.__kernel
        #Proceso
        result = np.mean(kernel,axis=2)
        who = "Average of {} channels".format(kernel.shape[2])
        self.mathPlot([result],[who],["gray"])
        return

   

 