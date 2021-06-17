#from .ModelHandler import ModelHandler
import numpy as np
class Test:
    @staticmethod
    def dummy(**kwargs):
        return kwargs["img_array"][0].astype(np.uint8),"Dummy"

class HandlerConfiguration(object):
    """
        Singleton que almacena la configuración de cada clase Handler implementada.
        Cualquier edición en los handlers en OutputHandler y PredictionHandler se verá reflejado
        en el control 'select' del árbol EQTreeView para poder ser llamado desde él.
        Sin embargo, el nuevo handler añadido tiene como regla devolver 2 elementos:
            +Un array que pueda interpretarse como una imagen RGB
            +Una cadena que de titulo a la imagen anterior
        Actualmente no existe forma de ampliar esta tecnica desde interfaz aunque este preparado para ello.
    """
    handlers = None
    @staticmethod
    def getHandler(type):
        from .OutputHandler import OutputHandler
        from .PredictionHandler import PredictionHandler
        from .KernelHandler import KernelHandler
        HandlerConfiguration.handlers = {
        OutputHandler : {
             "Score-CAM": OutputHandler.doScoreCam,
             "Guided BackPropagation" : OutputHandler.doGuidedBackPropagation,
             "Dummy" : Test.dummy
            },
        PredictionHandler : {
                "Grad-CAM" : PredictionHandler.doGradCAM,
                "Vanilla Occlussion": PredictionHandler.doVanillaOcclusion,
                "Integrated Gradients": PredictionHandler.doIntegratedGradients,
                "Dummy" : Test.dummy
            },
        KernelHandler : {
             "3ShapeDragMode": KernelHandler.RGBShapeDragMode,
             "XShapeDragMode": KernelHandler.XShapeDragMode,
             "3ShapePlotMode" : KernelHandler.RGBShapePlotMode,
             "XShapePlotMode" : KernelHandler.XShapePlotMode
            }
        } if HandlerConfiguration.handlers is None else HandlerConfiguration.handlers
        return HandlerConfiguration.handlers[type]

    



class Handler(object):
    """
        Clase base que sirve de plantilla para los handlers que manejaran los datos de
        las capas para hacerles una operación de debugging
    """
    def __init__(self, defaultAction, type):
        self._handlers = HandlerConfiguration.getHandler(type)
        self.__defaultAction = defaultAction
        return


    def __call__(self,**kwargs):
        """
            Este metodo es el unico que debe ser llamado desde fuera y apartir de los argumentos,
            se decidirá la acción a realizar
        """
        raise NotImplementedError("Virtual function __call__ has not been overrided")
        return 

    def getHandler(self,who:str) -> 'Callable': 
        """
            De momento, la unica utilidad de este metodo es para poder preguntar por argumentos 
            opcionales desde el widget "conv_debugger_widget"
        """
        if who in self._handlers:
            return self._handlers[who]
        return self.__defaultAction

    def mathPlot(self,imgs : 'List[np.ndarray]' ,titles : 'List[str]', cmaps : 'List[str]', gridRows : int = 1, gridCols : int = 1,*, show : bool = True ) ->None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(gridRows, gridCols)
        for i in range(gridRows):
            for j in range(gridCols):
                x = i*gridCols+j
                if x > len(imgs): break
                if gridCols == 1 and gridRows == 1:
                    ax.imshow(imgs[x], cmap=cmaps[x])
                    ax.set_title(titles[x])
                elif gridCols == 1:
                    ax[i].imshow(imgs[x], cmap=cmaps[x])
                    ax[i].set_title(titles[x])
                elif gridRows == 1:
                    ax[j].imshow(imgs[x], cmap=cmaps[x])
                    ax[j].set_title(titles[x])
                else:
                    ax[i,j].imshow(imgs[x], cmap=cmaps[x])
                    ax[i,j].set_title(titles[x])
        if show: plt.show()
        return fig




