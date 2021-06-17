


#from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

#from .extended_widgets import EQGraphicsView,EQTreeView
import PySide2
class Worker(PySide2.QtCore.QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @PySide2.QtCore.Slot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        self.fn(*self.args, **self.kwargs)


## ToDo--> Alguna forma de, atraves de otro nodo, decodificar las predicciones.
## Para predicciones de imagenet utilizar:
#       from tensorflow.keras.applications.imagenet_utils import decode_predictions 
#       decode_predictions(probs)



from typing import TYPE_CHECKING, Any, Dict, List, Optional,Callable, Tuple
from dial_core.utils import log
LOGGER = log.get_logger(__name__)

import numpy as np
import tensorflow
import PIL

from .model_tree import TreeItem
from PySide2.QtWidgets import (
    QGroupBox,
    QGridLayout,
    QLabel,
    )
from PySide2.QtGui import(
    QPixmap,
    )
from PySide2.QtCore import (
    QThreadPool,
    QPointF
    )
from .Utils import (
    ModelHandler,
    PredictionHandler,
    OutputHandler,
    KernelHandler,
    ImageUtils,
    HandlerConfiguration
    )

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
from PySide2.QtUiTools import loadUiType
Form, Base = loadUiType(os.path.join(current_dir, "./ConvVisualizer.ui"))
class ConvVisualizationWidget(Form, Base):

    onWindowComputed = PySide2.QtCore.Signal(object,object,object) #Señal utilizado para, desde un hilo, notificar a ConvDebuggerWidget como y donde debe añadir el controlador cargado
    def __init__(self, parent: PySide2.QtWidgets.QWidget = None):
        super(self.__class__, self).__init__(parent)
        self.setupUi(self)

        self.onWindowComputed.connect(self.addWidgetToCanvas)
        ############################################
        #Controles propios
        self.handlerLogger = QPlainTextLoggerHandler(self.debugText)
        LOGGER.addHandler(self.handlerLogger)
        #ToDo: Añadirle un formatter personalizado para que incluya hora, minuto y segundo del día en el que sucedio...
        LOGGER.info("Initialized debugger console")

        #tableViewLayers : PySide2.QtWidgets.QTableView
        from .model_table import TableLayersModel
        self.tableViewmodel = TableLayersModel(self.tableViewLayers) #TableLayersModel(PySide2.QtCore.QAbstractTableModel)
        self.tableViewLayers.setModel(self.tableViewmodel)
        self.tableViewLayers.horizontalHeader().setSectionResizeMode(PySide2.QtWidgets.QHeaderView.Stretch) #Para que el header ocupe todo el ancho disponible.
        self.tableViewLayers.setSelectionBehavior(PySide2.QtWidgets.QAbstractItemView.SelectRows) #Para que la selección sea solo a filas

        #imageLoadButton : PySide2.QtWidgets.QPushButton
        self.imageLoadButton.clicked.connect(self.onImageLoadButtonClick)

        #layersLoadButton : PySide2.QtWidgets.QPushButton
        self.layersLoadButton.clicked.connect(self.onLayersLoadButtonClick)

        #treeLayerView : extended_widgets.EQTreeView(PySide2.QtWidgets.QTreeView)
        from .model_tree import LayersTreeModel
        self.treeLayerViewmodel = LayersTreeModel(self.treeLayerView,widget=self)
        self.treeLayerView.setModel(self.treeLayerViewmodel)

        from .delegate_tree import LayerStyledDelegateItem
        self.treeLayerItemDelegate = LayerStyledDelegateItem(self.treeLayerView)
        self.treeLayerView.setItemDelegate(self.treeLayerItemDelegate)
        #self.treeLayerView.selectionModel().selectionChanged.connect(self.ontreeLayerViewItemSelectionChanged)
        
        from PySide2.QtWidgets import QGraphicsScene
        #canvas : PySide2.QtWidgets.QGraphicsView
        self.scene : PySide2.QtWidgets.QGraphicsScene = QGraphicsScene();
        self.canvas.setScene(self.scene)

        #canvas2 : extended_widgets.EQGraphicsView(PySide2.QtWidgets.QGraphicsView)
        self.scene2 : PySide2.QtWidgets.QGraphicsScene = QGraphicsScene();
        self.canvas2.setScene(self.scene2)
        self.canvas2.dropEventHandler = self.onDropCanvas

        #loadImagesButton : PySide2.QtWidgets.QPushButton
        self.loadImagesButton.clicked.connect(self.onloadImagesButtonClick)

        #progressbar : PySide2.QtWidgets.QProgressBar
        self.progressBar.reset()
        self.progressBar.valueChanged.connect(self.progressBar.setValue)
        ####################################################################################################################################
        ########################################################################################
        ############################################
        #Atributos
        self._trained_model : tensorflow.keras.Model = None
        self._preprocessorFunction : Callable = None

        self._img : PIL.Image.Image = None
        self._img_array : np.ndarray = None
        self._graphicWidgets : List[PySide2.QtWidgets.QGroupBox] = None  #Este elemento es necesario para no perder las referencias y que no crashee al limpiar la escena

        self._reminder : Dict[dict] = {} #Utilizado para que cuando se arrastren varios que requieran preguntar por argumentos, se pregunte 1 sola vez.

        self._n_elements : int = 0 #Utilizado para conocer de antemano el número total de elementos antes de cargarlos. Necesario para cargar en paralelo de imagenes
        self._master_threads_pool : PySide2.QtCore.QThreadPool = QThreadPool() #Para no bloquear el hilo principal.
        self._slave_threads_pool : PySide2.QtCore.QThreadPool = QThreadPool() #Creamos un 2º pool para poder esperar por él más dentro del 1º pool.
        ############################################

        return

    def set_preprocessorFunction(self,
                                 function : Callable) -> None: #processor_function --> Entrada "Custom Preprocessor"
        """
            Este metodo es llamado cada vez que el nodo recibe un input desde su entrada 'Custom Preprocessor'.
            Guarda el metodo pasado dentro del atributo '_preprocessorFunction' que será a su vez pasado a otras clases 
            responsables de generar las imagenes resultado del procesamiento de las diferentes tecnicas debugging implementadas.
            En ningún momento la imagen se preprocesa dentro de esta clase.
        """
        self._preprocessorFunction = function
        LOGGER.info("Preprocessor function has been setted")
        return

    def set_trained_model(self,
                          trained_model : tensorflow.keras.Model) -> None: #processor_function --> Entrada "Model"
        """
            Este metodo es llamado cada vez que el nodo recibe un input desde su entrada 'Model'.
            Es el responsable de cargar los datos basicos de las capas del modelo pasado como entrada en la tabla 'tableViewLayers' así
            como de generar una primera imagen aleatoria para poder trabajar con ella en el modelo.
        """
        self._trained_model = trained_model

        def dummyPreprocessorFunction(img : np.ndarray) -> np.ndarray : return img
        try:
            self._preprocessorFunction = getattr(getattr(tensorflow.keras.applications,trained_model.name),"preprocess_input")
        except AttributeError as e:
            self._preprocessorFunction = dummyPreprocessorFunction

        self.tableViewmodel.loadLayers(trained_model.layers)
        LOGGER.info("Model '{}' loaded".format(trained_model.name))

        self._img_array,self._img = ImageUtils.randomImage(self._trained_model.input.get_shape()[1:])
        self.loadImageToCanvas(self.canvas,self._img)
        #self._img_array = self._preprocessorFunction(self._img_array)
        self._img_array = self._img_array[np.newaxis,...] #Extendemos para que el modelo keras pueda recibir este input

        self.treeLayerViewmodel.clear()
        self.doPrediction()
        #self.prepareTreeViewRootItem(trained_model)
        return
    """
    def prepareTreeViewRootItem(self,model): #Sin uso. Anteriormente mostraba como raiz del árbol el nombre del propio modelo.
        self.treeLayerViewmodel.setRootName(model.name)
        self.treeLayerView.expand(self.treeLayerViewmodel.createIndex(0,0,self.treeLayerViewmodel.rootItem))
        self.treeLayerViewmodel.rootItem.setEventFunction(self.onModelItemClick, [])
        return
    """

    def addWidgetToCanvas(self,
                          window:PySide2.QtWidgets.QGroupBox,
                          canvas:PySide2.QtWidgets.QGraphicsView,
                          pos:PySide2.QtCore.QPointF) -> None:
        """
            Este metodo ad-hoc existe para que los hilos secundarios puedan dar la señal al hilo principal, mediante
            el signal 'onWindowComputed', de cargar las ventanas creadas en la escena canvas pasada como argumento.
            Este tipo de acciones no pueden realizarse fuera del hilo principal.
        """
        proxy = canvas.scene().addWidget(window)
        proxy.setPos(pos)
        self.treeLayerView.setEnabled(True)
        return

    def loadPixMap(self,
                   image:PIL.Image.Image) -> PySide2.QtGui.QPixmap:
        """
            Este metodo sirve como alias del proceso que se encarga de transformar una imagen en formato
            'PIL.Image.Image' en la misma pero en formato 'PySide2.QtGui.QPixmap' para que pueda ser añadida
            dentro de los diferentes widgets de QT con la invocación al metodo 'setPixmap'
        """
        #No usar para procesar imagenes grandes --> Overflow de la pila
        from PIL.ImageQt import ImageQt
        qim :PySide2.QtGui.QImage  = ImageQt(image)
        return QPixmap.fromImage(qim)

    def askOptionalArguments(self, 
                             method : Callable , 
                             key_arg : str) -> dict:
        """
            Metodo para recoger los argumentos opcionales del metodo y preguntar por unos nuevos.
            Se consideran opcionales aquellos argumentos que sean del tipo "keyword argument",
            es decir, para un metodo con la siguiente firma:
            test(a,b*,c,d=5)
            Es "keyword argument"  'd' , el argumento 'c' es "keyword-only argument"
            porque no tiene valor por defecto y por lo tanto será ignorado, importante asignarle
            un valor por defecto
        """
        optArgs : dict = method.__kwdefaults__ #Devuelve None si no tiene 'keyword-arguments' por defecto
        if optArgs is None: return {}
        args : dict = {}
        if key_arg in self._reminder:
            return self._reminder[key_arg]
        count = 1
        for key in optArgs:
            value,ok = PySide2.QtWidgets.QInputDialog().getInt(self, "Parametros {}/{}".format(count,len(optArgs)),
                                     "({}/{}) {}:".format(count,len(optArgs),key),optArgs[key])
            args[key] = value if ok else optArgs[key]
            count +=1
        self._reminder[key_arg] = args
        return args

    def setUpImageControl(self,
                          pixmap:PySide2.QtGui.QPixmap,
                          title:str="Title") -> PySide2.QtWidgets.QGroupBox:
        """
            This method created the simplest window with a title and an image inside to be 
            rendered inside a scene of QGraphicScene.
        """
        if self._graphicWidgets is None: #La idea es hacer el procedimiento "setUpImageControl" thread-safe
            self._graphicWidgets = [None]*self._n_elements #Importante. Este atributo debe estar seteado antes de empezar a procesar
        window = QGroupBox(title)
        #window.setWindowTitle(title)
        window.setFixedWidth(pixmap.width()+24)
        window.setFixedHeight(pixmap.height()+24)
        
        grid = QGridLayout(window)
        label = QLabel()
        label.setPixmap(pixmap)
        grid.addWidget(label, 0,0)

        window.setLayout(grid)
        self._graphicWidgets[self._graphicWidgets.count(None)-1] = window #Por lo tanto, empezamos insertando desde el final hacía el principio.
        #self._graphicWidgets.append(window) #Hace falta mantener una referencia en todo momento, de otra forma, al limpiar, se intenta acceder a referencias nulas
        return window

    def loadImageFromDisk(self,
                          imageName : str) -> Tuple[PIL.Image.Image,np.ndarray]:
        """
            Este metodo recibe como argumento una ruta a una imagen en disco y este metodo se encarga de traerlo a la memoria del programa,
            así como transforma la imagen en formato PIL.Image.Image en un np.ndarray, esto último para que la imagen pueda ser facilmente procesada
            para pasarsela a un modelo de red neuronal más adelante
        """
        from tensorflow.keras import preprocessing
        image : PIL.Image.Image = preprocessing.image.load_img(imageName,
                                             target_size=tuple(self._trained_model.input.shape)[1:3],
                                             interpolation='bicubic') #Cargo imagen desde disco duro en memoria

        imageArray : np.ndarray = preprocessing.image.img_to_array(image)
        #imageArray = self._preprocessorFunction(imageArray) #Le hago las transformaciones que el modelo de red neuronal espera
        imageArray = imageArray[np.newaxis,...] #Adecuo al formato que un modelo keras admite (minimo shape --> (-1,1))
        return image,imageArray

    def executeLoaders(self,
                   images: 'Callable o List[np.ndarray]',
                   canvas : PySide2.QtWidgets.QGraphicsView,
                   imageModifier : Callable = None,
                   params : dict = {}) ->None:
        """
            Metodo central encargado de la ejecución de las tecnicas de debugging de redes neuronales implementadas asi como de su visualización
        """
        self.treeLayerView.setEnabled(False)
        #Espero un generador de iterador(== una función que devuelva un iterador)y sino, es simplemente una lista:
        iterable = images() if callable(images) else images 
        
        self.progressBar.reset()
        self.progressBar.setFormat("Loading...");
        self.progressBar.setRange(0,self._n_elements+1)
        self.progressBar.setValue(1)

        #Empezamos procesando y guardando las imagenes
        def loadWindowProc(self,
                           item : 'Callable o str',
                           count: int,
                           imageModifier : Callable,
                           params : dict):
            image,widgetTitle = None,'Grad-CAM'
            if callable(item): #Función que se espera que devuelva 2 elementos, imagen y titulo de la ventana
                image,widgetTitle = item()
            elif type(item) is str: #String que se espera se trata de una ruta a fichero en disco
                _,image = self.loadImageFromDisk(item)
            if imageModifier is not None:
                image = imageModifier(image,**params)#handler.createHeatmap(imageArray,0)
            #pixMap = self.loadPixMap(Image.fromarray(image, 'RGB')) #No puede ser... Para imagenes grandes crashea...
            #Parece que lo anterior provoca un stackOverflow. Debo procesarlo aqui mismo:
            qim : PySide2.QtGui.QImage = PIL.ImageQt.ImageQt(PIL.Image.fromarray(image, 'RGB'))
            from PySide2.QtCore import QRect
            pixMap : PySide2.QtGui.QPixmap = QPixmap.fromImage(qim).copy(QRect()) #Esencial el copy, sino, crashea
            self.setUpImageControl(pixMap,title=widgetTitle)
            self.progressBar.valueChanged.emit(count)
            return 0
        ########################################################
        def imagesLoaderProc(self,
                             iterable : 'generator o Lst[str]',
                             canvas : PySide2.QtWidgets.QGraphicsView,
                             imageModifier : Callable = None,
                             params : dict = {}):
            count : int = 1
            #Hay operaciones que consumen mucha memoria y provocan que Tensorflow crashee. Dejarlo en 1 hasta que se encuentre solución:
            self._slave_threads_pool.setMaxThreadCount(1) 
            for item in iterable:
                count += 1
                windowLoader = Worker(loadWindowProc,self,item,count,imageModifier,params)
                self._slave_threads_pool.start(windowLoader) #Cargamos todos
                
            self._slave_threads_pool.waitForDone() #Esperamos hasta que todos hayan terminado...
            self._reminder = {} #Reinicializamos el "recordador de parametros opcionales" para que vuelva a preguntar después
            
            #Ahora viene el procedimiento de cargar las imagenes procesadas anteriormente al widget:
            totalWidth : int = 0
            totalHeight: int = 0
            prevShape : Tuple[int.int] = None
            count = 0
            for window in self._graphicWidgets:
                totalWidth += window.width() if (count < 2) else 0
                totalHeight += window.height() if (count%2 == 0) else 0
                pos = QPointF()
                if prevShape is not None:
                    x = count//2 #Mostramos en matriz (n/2)x2
                    y = count%2
                    pos = QPointF((prevShape[1]+5)*y,(prevShape[0]+5)*x )
                proxy = self.onWindowComputed.emit(window,canvas,pos)
                prevShape = tuple((window.height(),window.width()))
                count += 1
        
            self.progressBar.valueChanged.emit(self._n_elements+1)
            self.progressBar.setFormat("Finished Loading");
            canvas.scene().setSceneRect(0, 0,totalWidth+10 , totalHeight+10)
            return 0
        ########################################################
        imagesLoader = Worker(imagesLoaderProc,self,iterable,canvas,imageModifier, params)
        self._master_threads_pool.start(imagesLoader)
        return

    #####################################
    ##loadImagesButtonClick methods
    def onloadImagesButtonClick(self,
                                check : bool):
        """
            Este metodo es el que se ejecuta cuando le damos al boton que se encuentra debajo de nuestro árbol 'treeLayerView'.
            Se encarga de abrir una ventana modal para preguntar por las rutas de las imagenes a cargar así como de ejecutar los metodos
            responsables del procesamiento de la imagen en el modelo para obtener Grad-CAM
        """
        from PySide2.QtWidgets import QFileDialog
        imagesNames ,_ = QFileDialog.getOpenFileNames(self,"Open Image file", "", "Image Files(*.jfif *.jpg *.png)", options=QFileDialog.Options())
        if len(imagesNames) == 0: return
        handler = PredictionHandler(self._trained_model,self._preprocessorFunction)
        self.canvas2.scene().clear()
        if self._graphicWidgets is not None:
            self._graphicWidgets.clear() #Este elemento es necesario para no perder las referencias y que no crashee al limpiar la escena
            self._graphicWidgets = None
        self._n_elements = len(imagesNames)
        self.executeLoaders(imagesNames, self.canvas2, handler.doGradCAM, {'predictIndex':-1})
        return

    #####################################
    ##Canvas methods
    def loadImageToCanvas(self,
                  canvas:PySide2.QtWidgets.QGraphicsView,
                  img: 'Numpy.ndarray o PIL.Image.Image') -> None:
        """
            Este metodo es el responsable exclusivo de procesar la imagen pasada como argumento para cargarla
            en el canvas pasado como argumento
        """
        pix : PySide2.QtGui.QPixmap = None
        if type(img) is np.array:
            pix = self.loadPixMap(Image.fromarray(img, 'RGB')).copy(0,0,img.shape[0],img.shape[1])
        else: #Hace falta la copia porque parece que Python no aguanta la referencia temporal y su Gargabe Collector lo manda a la basura inmedetiamente
            pix = self.loadPixMap(img).copy(0,0,img.width,img.height) 
        canvas.scene().setSceneRect(0, 0,canvas.width() , canvas.height())
        from PySide2.QtCore import Qt
        canvas.scene().addPixmap(pix.scaled(canvas.width(),canvas.height(),Qt.IgnoreAspectRatio,Qt.SmoothTransformation));
        return

    def onDropCanvas(self,
                     event:PySide2.QtGui.QDropEvent) -> None:
        treeItems = self.treeLayerView.selectionModel().selectedRows()
       
        self.canvas2.scene().clear()
        if self._graphicWidgets is not None:
            self._graphicWidgets.clear() #Este elemento es necesario para no perder las referencias y que no crashee al limpiar la escena
            self._graphicWidgets = None
        self._n_elements = len(treeItems) #Util para setear el valor maximo de progressBar más adelante y para reservar espacio en _graphicWidgets para los hilos
        #Primero preguntamos por los argumentos opcionales:
        # (Esta realizado de esta forma para poder utilizar threading más adelante. Preguntamos mientras estemos en el hilo principal)
        for i in range(self._n_elements):
            treeItems[i].internalPointer().callEvent(executeMode=False)

        # Luego ejecutamos el propio algoritmo para recoger los valores.
        def iteratorImages(treeItems = treeItems):
            """
                Este metodo nos permite ir cargando las imagenes a medida que vamos necesitandolas.
                De esta forma, no cargamos todas las imagenes y luego iteramos sobre ellas, saturando la memoria
                sino que cargamos una a una y las vamos descartando de la memoria una vez trabajada con ella.
            """
            n : int = len(treeItems)
            for i in range(n):
                yield treeItems[i].internalPointer().callEvent #Para que sea el hilo el que ejecute el proceso, retornamos el metodo.
                #yield treeItems[i].internalPointer().callEvent()
            return
        self.executeLoaders(iteratorImages,self.canvas2) #Ponen en marcha los hilos para no bloquear el hilo principal en el proceso
        return

    #####################################
    ##imageLoadButton methods
    def onImageLoadButtonClick(self,
                               checked : bool) -> None:
        """
            Este metodo es el que se llama cuando se quiere cambiar la imagen de trabajo principal. Además de cargar la imagen en la clase,
            la procesa para cargarla en el canvas nº1 y, automaticamente, realiza la predicción del modelo sobre la misma y procesa y carga 
            el resultado en 'treeLayerView'
        """
        from PySide2.QtWidgets import QFileDialog
        imageName, _ = QFileDialog.getOpenFileName(self,"Open Image file", "", "Image Files(*.jfif *.jpg *.png)", options=QFileDialog.Options())
        if imageName != "":
            LOGGER.info("File Image '{}' opened".format(imageName))
            from pathlib import Path
            imageName = Path(imageName)
            self._img,self._img_array = self.loadImageFromDisk(imageName) 
            self.loadImageToCanvas(self.canvas,self._img)

            self.treeLayerViewmodel.clear()
            self.doPrediction()
            #self.prepareTreeViewRootItem(self._trained_model)
        return 0

    #####################################
    ##layersLoadButton methods
    def onLayersLoadButtonClick(self,
                                checked : bool) -> None:
        """
            Este metodo es el que se ejecuta cuando seleccionamos varias filas en la tabla y le damos al boton "layersLoadButton"
            Carga la información que se requiere de las capas en nuestro árbol 'treeLayerView'
        """
        layers : List[tensorflow.keras.layers.Layer] = [self._trained_model.layers[qmodelIndex.row()] for qmodelIndex in self.tableViewLayers.selectionModel().selectedRows()]
        self.treeLayerViewmodel.loadLayers(layers)
        self.doPrediction()
        #self.treeLayerView.resizeColumnToContents(1)
        self.tabWidget.setCurrentIndex(1)
        self.treeLayerView.expand(self.treeLayerViewmodel.createIndex(0,0,self.treeLayerViewmodel.rootItem))
        #self.tableViewLayers.selectionModel().selectedRows() #-->[<PySide2.QtCore.QModelIndex...3F823C848>, <PySide2.QtCore.QMod...3F823C8C8>, <PySide2.QtCore.QMod...3F823C908>] 
        return

    #####################################
    ##predict methods
    def doPrediction(self) -> None:
        """
            Este metodo es responsable de llamar a los metodos que generan los nodos que la aplicación utiliza para mostrar y guardar información
            sobre las predicciones del modelo así como de cargarlos en nuestro árbol 'treeLayerView'
        """
        #from datetime import datetime
        #child = LayerTreeItem(["Predictions",datetime.now().strftime("%d/%m/%Y %H:%M:%S")],self.treeLayerViewmodel.rootItem)
        from .model_tree import LayerTreeItem
        #child = LayerTreeItem(["Predictions",["Grad-CAM","Vanilla Occlussion","Integrated Gradients"]],self.treeLayerViewmodel.rootItem)
        child = LayerTreeItem(["Predictions","Debug Technique"],self.treeLayerViewmodel.rootItem)

        handler = PredictionHandler(self._trained_model,self._preprocessorFunction)
        handler.dataHandler = lambda index,value: self.setUpPredictionData(index,value,child)
        handler.doPrediction(self._img_array)

        child.setEventFunction(self.onGroupPredictSelect,[handler])
        self.treeLayerViewmodel.rootItem.appendChild(child)
        self.treeLayerViewmodel.modelReset.emit()
        return
 
    def setUpPredictionData(self,
                            index:int,
                            value:int,
                            parent:TreeItem) -> None:
        """
            Este es un metodo adhoc responsable de crear los nodos hijos del padre creado en 'doPrediction'
            Este metodo es llamado desde 'PredictionHandler' al asignarle como atributo 'dataHandler' este metodo 
            y ejecutar 'doPrediction'
        """
        from .model_tree import LayerTreeItem
        #child = LayerTreeItem([str(index),str(value)],parent)
        #child = LayerTreeItem(["{}:{}".format(index,value),["Grad-CAM","Vanilla Occlussion","Integrated Gradients"]],parent)
        options = list(HandlerConfiguration.getHandler(PredictionHandler).keys())
        child = LayerTreeItem(["{}:{}".format(index,value),options],parent)
        child.setEventFunction(self.onSinglePredictionSelect,[index,child.data]) #Fijarse que le estamos pasando como argumento 2º un puntero al metodo 'data'
        parent.appendChild(child)
        return
    #####################################
    ##treeLayerViewItems events

    def onModelItemClick(self,
                         dragMode : bool) -> None: #NOT USED
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

    def onGroupPredictSelect(self,
                             handler : PredictionHandler,
                             *,
                             dragMode : bool=True,
                             executeMode : bool=True,
                             **kwargs) -> None:
        self.scene2.clear()
        chart = handler.plotPrediction(self.canvas2.width(), self.canvas2.height(),5)
        self.scene2.addItem(chart)
        self.scene2.update()
        return

    #REFACTORIZED
    def onSinglePredictionSelect(self,
                                 predictionIndex : int,
                                 who : Callable,
                                 *,
                                 dragMode : bool = True,
                                 executeMode : bool = True,
                                 **kwargs) -> 'np.ndarray o dict':
        # Lo que ocurre adelante es que el argumento who tiene un puntero al metodo del nodo 'data' y por la forma de procesar
        # la edición, el elemento que aparece seleccionado en el select en la 2º columna es siempre el primero.
        who = who(1)[0] #Truquito. Pasamos puntero al metodo que recoge el valor.
        handler = PredictionHandler(self._trained_model,self._preprocessorFunction)
        if executeMode is True:
            optArgs = self.askOptionalArguments(handler.getHandler(who),who)
            return handler(widget=self, 
                           who=who,
                           dragMode=dragMode,
                           img_array=self._img_array,
                           predictionIndex = predictionIndex,
                           **optArgs)
        else:
            return self.askOptionalArguments(handler.getHandler(who),who)

  
    #REFACTORIZED
    def onGroupOutputSelect(self,
                            output : tensorflow.keras.layers.Layer,
                            layerName : str,
                            who : Callable,
                            *,
                            dragMode : bool = True,
                            executeMode : bool = True,
                            **kwargs) -> 'np.ndarray o dict':
        who = who(1)[0]
        handler = OutputHandler(self._trained_model.inputs,[output],self._preprocessorFunction)
        if executeMode is True:
            optArgs = self.askOptionalArguments(handler.getHandler(who),who)
            return handler(widget=self, 
                           who=who,
                           dragMode=dragMode,
                           layerName=layerName,
                           classifier=self._trained_model,
                           img_array=self._img_array,
                           **optArgs)
        else:
            return self.askOptionalArguments(handler.getHandler(who),who)
    #REFACTORIZED
    def onSingleOutputSelect(self,
                             output : tensorflow.keras.layers.Layer,
                             outputIndex : int ,
                             who : Callable,
                             *,
                             dragMode : bool =True,
                             executeMode : bool =True,
                             **kwargs) -> 'np.ndarray o dict':
        handler = OutputHandler(self._trained_model.inputs,[output], self._preprocessorFunction)
        if executeMode is True:
            optArgs : dict = self.askOptionalArguments(handler.getHandler("FeatureMapAndGradientAscent"),"FeatureMapAndGradientAscent")
            return handler(widget=self,
                           who=who,
                           dragMode=dragMode,
                           img_array=self._img_array,
                           outputIndex=outputIndex,
                           **optArgs)
        else:
            return self.askOptionalArguments(handler.getHandler("FeatureMapAndGradientAscent"),"FeatureMapAndGradientAscent")

    #REFACTORIZED
    def onSingleKernelSelect(self,
                             kernel : np.ndarray,
                             who : str,
                             *,
                             dragMode : bool = True,
                             executeMode : bool = True,
                             **kwargs) -> 'np.ndarray o dict':
        handler = KernelHandler(kernel)
        if executeMode is True:
            return handler(widget=self,who=who,dragMode=dragMode,kernel=kernel)
        else:
            def dummy(): return
            return self.askOptionalArguments(dummy,"Dummy")

   
        
    ################################################################################

#Estaria bien utilizar la clase LoggerTextboxWidget en dial-gui/dial-gui/widgets/log/logger_textbox.py
#Ocurre que al hacer el "promote" del objeto con LoggerTextboxWidget en QT Designer, me esta fallando...
#¿Puede ser que sea porque, la clase LoggerTextBoxWidget, hace herencia de 2 objetos, logger y QPlainTextEdit?
import logging
class QPlainTextLoggerHandler(logging.Handler):
    def __init__(self, textArea : PySide2.QtWidgets.QPlainTextEdit ):
        logging.Handler.__init__(self)
        self.textArea : PySide2.QtWidgets.QPlainTextEdit  = textArea
        return
            
    def emit(self,record : logging.LogRecord) -> None:
        self.textArea.appendPlainText(self.format(record))
        return

import dependency_injector.providers as providers
ConvVisualizationWidgetFactory = providers.Factory(ConvVisualizationWidget)
