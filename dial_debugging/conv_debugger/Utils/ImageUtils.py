
import cv2
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
class ImageUtils(object):
    """description of class"""

    @staticmethod
    def normalize(img):
        """
            Lleva los valores al rango normal [0,1] 
        """
        import numpy as np
        min,max = np.min(img),np.max(img)
        if max == min:#Todos los valores son una cte.
            return img#np.zeros(img.shape)
        #Normalizamos al rango [0,1]
        return (img-np.min(img)) / np.ptp(img) 


    @staticmethod
    def scaleValues(img,range):
        #Normalize
        normalized = ImageUtils.normalize(img)
        #Scaled to desired scale
        scaled = normalized*(range[1]-range[0])+range[0]
        return scaled

    @staticmethod
    def processGradients(grads,std : float = 0.25):
        """
            Este metodo busca procesar la imagen para normalizarla, eliminar los valores outliers
            y poder visualizarla como imagen RGB al final pero mostrando maximizado los pixeles mas
            importantes y eliminando outliers.
        """
        #x = grads#.copy()
        #x -= x.mean() #Centramos media en 0
        #x /= (x.std() + 1e-07) #Colocamos la varianza en 1 (y añadimos un epsilon por si x.std() ya es 0)
        #x *= std #Hacemos que la varianza de los valores sea la especificada

        #import numpy as np
        # clip to [0, 1]
        #x += 0.5 #Desplazamos la media a 0.5
        #x = np.clip(x, 0, 1) #Nos cargamos los valores outliers

        # convertimos a valores dentro de un rango para visualizar imagen RGB
        x = ImageUtils.normalize(grads)
        x *= 255
        import numpy as np
        return x.astype(np.uint8)

    @staticmethod
    def randomArray(shape):
        import numpy as np
        return np.random.random(shape)
    @staticmethod
    def randomImage(shape):
        import numpy as np
        array = (255*ImageUtils.randomArray(shape)).round().astype(np.uint8)
        from PIL import Image
        return array,Image.fromarray(array, 'RGB')

    @staticmethod
    def grayToRGB(img):
        return ImageUtils.expandGrayChannels(img,3)

    @staticmethod
    def grayToCMYK(img):
        return ImageUtils.expandGrayChannels(img,4)

    @staticmethod
    def expandGrayChannels(img,channels):
        import numpy as np
        return np.array([[[val]*channels for val in row] for row in img], dtype=np.uint8)

    @staticmethod
    def composeImages(imgs, expectedDim, grid = -1, margin = 5):
        import numpy as np
        if grid == -1:
            if len(imgs) == 1: #Caso base
                grid = (1,1)
            else:
                cols = 2
                rows = int(np.round(len(imgs)/2 + 0.001))
                grid = (rows,cols) # Grid de n/2(height) x n/2(width) 
        elif grid[0]*grid[1] < len(img):
            raise Exception("Invalid shape. Shape is lesser than image shape array")
        #Se redimensionan todas las imagenes con diferente dimensiones:
        for i in range(0,len(imgs)):
            img = imgs[i]
            if expectedDim[0] != img.shape[0] or expectedDim[1] != img.shape[1]:
                import cv2
                imgs[i] = cv2.resize(img,expectedDim,interpolation=cv2.INTER_NEAREST)
        #Ahora que todas las imagenes tienen las mismas dimensiones, resulta más sencillo tratarlas:
        height,width,channels = (imgs[0].shape[0]+margin)*grid[0],(imgs[0].shape[1]+margin)*grid[1],img.shape[2]
        result = np.zeros((height, width, channels),dtype=np.uint8)
        #Asignación
        for i in range(0,len(imgs)):
            img = imgs[i]
            y,x = i//grid[1],i%grid[1]
            workingHeight,workingWidth = (imgs[0].shape[0]+margin)*y,(img.shape[1]+margin)*x
            result[
                workingHeight:workingHeight+img.shape[0],
                workingWidth:workingWidth+img.shape[1],
                :] = np.round(img).astype(np.uint8)
        return result

    @staticmethod
    def overlay(originalImg, overlay,emphasize:bool = True , hif:float = .5 , overlay_colormap = cv2.COLORMAP_JET):
        import numpy as np
        if emphasize: #La función sigmoide sirve para descartar/disminuir valores bajos y aumentar/potenciar valores altos
            def sigmoid(x, a, b, c):
                return c / (1 + np.exp(-a * (x-b)))
            overlay = sigmoid(overlay, 50, 0.5, 1)

        overlay = 255 * overlay
        overlay = cv2.resize(overlay, originalImg.shape[0:2])
        #ToFix: Debe salir al revés, rojo parte más importante
        overlay = 255 - overlay #El Fix. ¿Pero por qué?
        overlay = cv2.applyColorMap(overlay.round().astype(np.uint8), overlay_colormap) #cv2.cvtColor(cam3, cv2.COLOR_BGR2RGB)
        superimposed_img = overlay * hif + originalImg
        superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)
        return superimposed_img


