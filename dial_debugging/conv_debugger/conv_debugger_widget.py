
import os
import logging
import numpy as np

from typing import TYPE_CHECKING, Any, Dict, List, Optional
from pathlib import Path

import dependency_injector.providers as providers

from dial_core.utils import log
LOGGER = log.get_logger(__name__)

from PySide2.QtCore import (
    Qt,
    QRect
)
from PySide2.QtUiTools import loadUiType
from PySide2.QtGui import (
    QPixmap
   )

from PySide2.QtWidgets import (
    QWidget,
    QGraphicsScene,
    QPlainTextEdit,
    QHeaderView,
    QFileDialog,
    QAbstractItemView,
    QLabel,
    QGridLayout,
    QGroupBox
)

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import preprocessing
#from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

from .model_table import TableLayersModel
from .model_tree import (
    LayerTreeItem,
    LayersTreeModel
)
from .delegate_tree import LayerStyledDelegateItem

from .Utils import (
    ModelHandler,
    PredictionHandler,
    OutputHandler,
    KernelHandler,
    ImageUtils
    )

from .extended_widgets import EQGraphicsView,EQTreeView
#DONE: Mirar gradiente descents

#DONE: Trabajar en la visualización de varias ventanas.:
    #DONE: Mostrar predicciones top5 en histograma - Done (pero un donut mejor)
    #DONE: Mostrar multiples features maps
    #DONE: Mostrar multiples weights maps. Mostrar los pesos juntos como 1 imagen RGB (para kernels con 3 canales)

## DONE--> Alguna forma de pasarle una función de preprocesamiento para modelos no-estandar desde un nodo
## Para modelos estandar, el siguiente truco funciona: preprocess_input = getattr(getattr(tf.keras.applications,model_dicts[model_name]),"preprocess_input")

## ToDo--> Alguna forma de, atraves de otro nodo, decodificar las predicciones.
## Para predicciones de imagenet utilizar:
#       from tensorflow.keras.applications.imagenet_utils import decode_predictions 
#       decode_predictions(probs)

current_dir = os.path.dirname(os.path.abspath(__file__))
Form, Base = loadUiType(os.path.join(current_dir, "./ConvDebugger.ui"))


class ConvDebuggerWidget(Form, Base):
    def __init__(self, parent: QWidget = None):
        super(self.__class__, self).__init__(parent)
        self.setupUi(self)

        ############################################
        #DebugText
        self.handlerLogger = QPlainTextLoggerHandler(self.debugText)
        LOGGER.addHandler(self.handlerLogger)
        #ToDo: Añadirle un formatter personalizado para que incluya hora, minuto y segundo del día en el que sucedio...
        LOGGER.info("Initialized debugger console")

        #tableViewLayers
        self.tableViewmodel = TableLayersModel(self.tableViewLayers)
        self.tableViewLayers.setModel(self.tableViewmodel)
        self.tableViewLayers.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch) #Para que el header ocupe todo el ancho disponible.
        self.tableViewLayers.setSelectionBehavior(QAbstractItemView.SelectRows) #Para que la selección sea solo a filas

        #imageLoadButton
        self.imageLoadButton.clicked.connect(self.onImageLoadButtonClick)

        #layersLoadButton
        self.layersLoadButton.clicked.connect(self.onLayersLoadButtonClick)

        #predictButton
        #self.predictButton.clicked.connect(self.onPredictButtonClick)

        #treeLayerView
        self.treeLayerViewmodel = LayersTreeModel(self.treeLayerView,widget=self)#QStandardItemModel() 
        self.treeLayerView.setModel(self.treeLayerViewmodel)

        self.treeLayerItemDelegate = LayerStyledDelegateItem(self.treeLayerView)
        self.treeLayerView.setItemDelegate(self.treeLayerItemDelegate)
        #self.treeLayerView.selectionModel().selectionChanged.connect(self.ontreeLayerViewItemSelectionChanged)
        
        #Canvas
        self.scene = QGraphicsScene();
        self.canvas.setStyleSheet("border-width: 0px; border-style: solid;background: transparent");
        self.canvas.setScene(self.scene)

        #Canvas2
        self.scene2 = QGraphicsScene();
        self.canvas2.setScene(self.scene2)
        self.canvas2.dropEventHandler = self.onDropCanvas

        #loadImagesButton
        self.loadImagesButton.clicked.connect(self.onloadImagesButtonClick)
        ############################################
        #Atributos
        self._trained_model = None
        self._preprocessorFunction = None

        self._img = None
        self._img_array = None
        self._graphicWidgets = []  #Este elemento es necesario para no perder las referencias y que no crashee al limpiar la escena

        self._reminder = {} #Utilizado para que cuando se arrastren varios que requieran preguntar por argumentos, se pregunte 1 sola vez.

        ############################################

        return

    def set_preprocessorFunction(self,function): #processor_function
        self._preprocessorFunction = function
        LOGGER.info("Preprocessor function setted")
        return

    def set_trained_model(self,trained_model : Model): #processor_function
        self._trained_model = trained_model

        def dummyPreprocessorFunction(img):
            return img
        try:
            self._preprocessorFunction = getattr(getattr(tf.keras.applications,trained_model.name),"preprocess_input")
        except AttributeError as e:
            self._preprocessorFunction = dummyPreprocessorFunction

        self.tableViewmodel.loadLayers(trained_model.layers)
        LOGGER.info("Model loaded")

        self._img_array,self._img = ImageUtils.randomImage(self._trained_model.input.get_shape()[1:])
        self.loadImage(self.canvas,self._img)
        #self._img_array = self._preprocessorFunction(self._img_array)
        self._img_array = self._img_array[np.newaxis,...] #Extendemos para que el modelo keras pueda recibir este input

        self.treeLayerViewmodel.clear()
        self.doPrediction()
        self.prepareTreeViewRootItem(trained_model)
        return

    def prepareTreeViewRootItem(self,model):
        self.treeLayerViewmodel.setRootName(model.name)
        self.treeLayerView.expand(self.treeLayerViewmodel.createIndex(0,0,self.treeLayerViewmodel.rootItem))
        self.treeLayerViewmodel.rootItem.setEventFunction(self.onModelItemClick, [])
        return
    def loadPixMap(self,image):
        #No usar para procesar imagenes grandes
        from PIL.ImageQt import ImageQt
        qim = ImageQt(image)
        return QPixmap.fromImage(qim)

    def loadImage(self,canvas,img):
        pix = None
        if type(img) is np.array:
            pix = self.loadPixMap(Image.fromarray(img, 'RGB')).copy(0,0,img.shape[0],img.shape[1])
        else: #Hace falta la copia porque parece que Python no aguanta la referencia temporal y su Gargabe Collector lo manda a la basura inmedetiamente
            pix = self.loadPixMap(img).copy(0,0,img.width,img.height) 
        canvas.scene().setSceneRect(0, 0,canvas.width() , canvas.height())
        canvas.scene().addPixmap(pix.scaled(canvas.width(),canvas.height(),Qt.IgnoreAspectRatio,Qt.SmoothTransformation));
        return


    def askOptionalArguments(self, method , key_arg):
        """
            Metodo para recoger los argumentos opcionales del metodo y preguntar por unos nuevos.
            Se consideran opcionales aquellos argumentos que sean del tipo "keyword argument",
            es decir, para un metodo con la siguiente firma:
            test(a,b*,c,d=5)
            Es "keyword argument"  'd' , el argumento 'c' es "keyword-only argument"
            porque no tiene valor por defecto y por lo tanto será ignorado, importante asignarle
            un valor por defecto
        """
        #Recogemos los argumentos con algún tipado
        #typedArgs = method.__annotations__  #Out: {'<nameArg>' : <type>}
        optArgs = method.__kwdefaults__
        if optArgs is None: return {}
        args = {}
        if key_arg in self._reminder:
            return self._reminder[key_arg]
        count = 1
        import PySide2
        for key in optArgs:
            value, ok = PySide2.QtWidgets.QInputDialog().getInt(self, "Parametros {}/{}".format(count,len(optArgs)),
                                     "({}/{}) {}:".format(count,len(optArgs),key),optArgs[key])
            args[key] = value if ok else optArgs[key]
            count +=1
        self._reminder[key_arg] = args
        return args

    def setUpImageControl(self,pixmap,title:str="Title"):
        """
            Simplest Window with a title and an image inside to be 
            rendered inside scene of QGraphicScene
        """
        window = QGroupBox(title)
        #window.setWindowTitle(title)
        window.setFixedWidth(pixmap.width()+24)
        window.setFixedHeight(pixmap.height()+24)
        
        grid = QGridLayout()
        label = QLabel()
        label.setPixmap(pixmap)
        grid.addWidget(label, 0,0)

        window.setLayout(grid)
        self._graphicWidgets.append(window) #Hace falta mantener una referencia en todo momento, de otra forma, al limpiar, se intenta acceder a referencias nulas
        return window

    def loadImageFromDisk(self,imageName : str):
        image = preprocessing.image.load_img(imageName,
                                                        target_size=tuple(self._trained_model.input.shape)[1:3],
                                                        interpolation='bicubic') #Cargo imagen desde disco duro en memoria

        imageArray = preprocessing.image.img_to_array(image) #Lo transformo de PIL.Image a un numypy array
        #imageArray = self._preprocessorFunction(imageArray) #Le hago las transformaciones que el modelo de red neuronal espera
        imageArray = imageArray[np.newaxis,...] #Adecuo al formato que un modelo keras admite (minimo shape --> (-1,1))
        return image,imageArray

    def loadImages(self,images, canvas, imageModifier = None, params = {}):
        totalWidth = 0
        totalHeight= 0
        prevShape = None
        import PIL
        #Espero un generador de iterador(o una función que devuelva un iterador)y sino, es simplemente una lista:
        iterable = images() if callable(images) else images 
        count = 0
        for image in iterable:
            widgetTitle = ""
            if type(image) is tuple: #Entonces se espera que sea una lista que contenga 2 elementos, imagen y titulo de la ventana
                image,widgetTitle = image
            elif type(image) is str: #Es string suponemos que se nos pasa una ruta a fichero en disco
                _,image = self.loadImageFromDisk(image)
            if imageModifier is not None:
                image,widgetTitle = imageModifier(image,**params)#handler.createHeatmap(imageArray,0)

            #pixMap = self.loadPixMap(Image.fromarray(image, 'RGB')) #No puede ser... Para imagenes grandes crashea...
            #En su lugar lo pongo aqui mismo e ya:
            qim = PIL.ImageQt.ImageQt(PIL.Image.fromarray(image, 'RGB'))
            pixMap = QPixmap.fromImage(qim).copy(QRect()) #Esencial el copy, sino, crashea
            window = self.setUpImageControl(pixMap,title=widgetTitle)
            totalWidth += window.width() if (count < 2) else 0
            totalHeight += window.height() if (count%2 == 0) else 0
            proxy = canvas.scene().addWidget(window)
            if prevShape is not None:
                x = count//2 #Mostramos en matriz (n/2)x2
                y = count%2
                proxy.setPos((prevShape[1]+5)*y,(prevShape[0]+5)*x )
                #coords.setPos((prevShape[1]+5)*y,(prevShape[0]+5)*x)
            prevShape = image.shape
            count += 1
        canvas.scene().setSceneRect(0, 0,totalWidth+10 , totalHeight+10)
    #####################################
    ##loadImagesButtonClick methods
    def onloadImagesButtonClick(self,event):
        imagesNames,_ = QFileDialog.getOpenFileNames(self,"Open Image file", "", "Image Files(*.jfif *.jpg *.png)", options=QFileDialog.Options())
        if len(imagesNames) == 0: return
        handler = PredictionHandler(self._trained_model,self._preprocessorFunction)
        self.canvas2.scene().clear()
        self._graphicWidgets.clear() #Este elemento es necesario para no perder las referencias y que no crashee al limpiar la escena
        self.loadImages(imagesNames, self.canvas2, handler.createHeatmap, {'predictIndex':-1})
        return

    #####################################
    ##Canvas methods
    def onDropCanvas(self,event):
        treeItems = self.treeLayerView.selectionModel().selectedRows()
       
        self.canvas2.scene().clear()
        self._graphicWidgets.clear() #Este elemento es necesario para no perder las referencias y que no crashee al limpiar la escena
        def iteratorImages(treeItems = treeItems):
            """
                Este metodo nos permite ir cargando las imagenes a medida que vamos necesitandolas.
                De esta forma, no cargamos todas las imagenes y luego iteramos sobre ellas, saturando la memoria
                sino que cargamos una a una y las vamos descartando de la memoria una vez trabajada con ella.
            """
            n = len(treeItems)
            for i in range(n):
                yield treeItems[i].internalPointer().callEvent(True)
            return
        self.loadImages(iteratorImages,self.canvas2)
        self._reminder = {}
        return

    #####################################
    ##imageLoadButton methods
    def onImageLoadButtonClick(self, checked):
        imageName, _ = QFileDialog.getOpenFileName(self,"Open Image file", "", "Image Files(*.jfif *.jpg *.png)", options=QFileDialog.Options())
        if imageName != "":
            LOGGER.info("File Image '{}' opened".format(imageName))
            imageName = Path(imageName)
            self._img,self._img_array = self.loadImageFromDisk(imageName) 
            self.loadImage(self.canvas,self._img)

            self.treeLayerViewmodel.clear()
            self.doPrediction()
            self.prepareTreeViewRootItem(self._trained_model)
        return 0

    #####################################
    ##layersLoadButton methods
    def onLayersLoadButtonClick(self,checked):
        beginIndex = self.tableViewLayers.selectionModel().selectedRows()[0].row()
        endIndex = beginIndex + len(self.tableViewLayers.selectionModel().selectedRows())
        layers = self._trained_model.layers[beginIndex:endIndex]
        self.treeLayerViewmodel.loadLayers(layers)
        self.doPrediction()
        #self.treeLayerView.resizeColumnToContents(1)
        self.tabWidget.setCurrentIndex(1)
        self.treeLayerView.expand(self.treeLayerViewmodel.createIndex(0,0,self.treeLayerViewmodel.rootItem))
        #self.tableViewLayers.selectionModel().selectedRows() #-->[<PySide2.QtCore.QModelIndex...3F823C848>, <PySide2.QtCore.QMod...3F823C8C8>, <PySide2.QtCore.QMod...3F823C908>] 
        return

    #####################################
    ##predict methods
    def doPrediction(self):
        from datetime import datetime
        #child = LayerTreeItem(["Predictions",datetime.now().strftime("%d/%m/%Y %H:%M:%S")],self.treeLayerViewmodel.rootItem)
        child = LayerTreeItem(["Predictions",["Grad-CAM","Vanilla Occlussion","Integrated Gradients"]],self.treeLayerViewmodel.rootItem)

        handler = PredictionHandler(self._trained_model,self._preprocessorFunction)
        handler.dataHandler = lambda index,value: self.setUpPredictionData(index,value,child)
        handler.doPrediction(self._img_array)

        child.setEventFunction(self.onGroupPredictSelect,[handler])
        self.treeLayerViewmodel.rootItem.appendChild(child)
        self.treeLayerViewmodel.modelReset.emit()
        return
 
    def setUpPredictionData(self,index,value,parent):
        child = LayerTreeItem([str(index),str(value)],parent)
        child.setEventFunction(self.onSinglePredictionSelect,[index,parent.data])
        parent.appendChild(child)
        return
    #####################################
    ##treeLayerViewItems events

    def onModelItemClick(self,dragMode):
        """
        handler = ModelHandler(self._trained_model,self._preprocessorFunction)
        args = self.askOptionalArguments(handler.createIntegrated,"Integrated Gradient")
        imgs,titles = handler.createIntegrated(self._img_array,**args)
        if dragMode:
            return imgs[1],titles[1]
        else:
            handler.mathPlot(imgs,titles)
        """
        return

    def onGroupPredictSelect(self,handler,dragMode):
        self.scene2.clear()
        chart = handler.plotPrediction(self.canvas2.width(), self.canvas2.height(),5)
        self.scene2.addItem(chart)
        self.scene2.update()
        return

    #REFACTORIZED
    def onSinglePredictionSelect(self, predictionIndex, who,dragMode):
        # La idea adhoc esque el padre tendrá en la columna 2 el elemento seleccionado que viene
        # a ser el 1º elemento en la lista
        who = who(1)[0] #Truquito. Pasamos puntero al metodo que recoge el valor.
        handler = PredictionHandler(self._trained_model,self._preprocessorFunction)
        optArgs = self.askOptionalArguments(handler.getHandler(who),who)
        return handler(widget=self, 
                       who=who,
                       dragMode=dragMode,
                       img_array=self._img_array,
                       predictionIndex = predictionIndex,
                       **optArgs)

  
    #REFACTORIZED
    def onGroupOutputSelect(self,output,layerName,who,dragMode):
        who = who(1)[0]
        handler = OutputHandler(self._trained_model.inputs,[output],self._preprocessorFunction)
        optArgs = self.askOptionalArguments(handler.getHandler(who),who)
        return handler(widget=self, 
                       who=who,
                       dragMode=dragMode,
                       layerName=layerName,
                       classifier=self._trained_model,
                       img_array=self._img_array,
                       **optArgs)
    #REFACTORIZED
    def onSingleOutputSelect(self,output,outputIndex,who,dragMode):
        handler = OutputHandler(self._trained_model.inputs,[output], self._preprocessorFunction)
        optArgs = self.askOptionalArguments(handler.getHandler("FeatureMapAndGradientAscent"),"FeatureMapAndGradientAscent")
        return handler(widget=self,
                       who=who,
                       dragMode=dragMode,
                       img_array=self._img_array,
                       outputIndex=outputIndex,
                       **optArgs)

    #REFACTORIZED
    def onSingleKernelSelect(self,kernel,who,dragMode):
        handler = KernelHandler(kernel)
        return handler(widget=self,who=who,dragMode=dragMode,kernel=kernel)
   
        
    ################################################################################

#Estaria bien utilizar la clase LoggerTextboxWidget en dial-gui/dial-gui/widgets/log/logger_textbox.py
#Ocurre que al hacer el "promote" del objeto con LoggerTextboxWidget en QT Designer, me esta fallando...
#¿Puede ser que sea porque, la clase LoggerTextBoxWidget, hace herencia de 2 objetos, logger y QPlainTextEdit?
class QPlainTextLoggerHandler(logging.Handler):
    def __init__(self, textArea : QPlainTextEdit ):
        logging.Handler.__init__(self)
        self.textArea = textArea
        return
            
    def emit(self,record):
        self.textArea.appendPlainText(self.format(record))
        return


ConvDebuggerWidgetFactory = providers.Factory(ConvDebuggerWidget)
