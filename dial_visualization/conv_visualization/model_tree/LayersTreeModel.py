from PySide2 import (
    QtCore,
    QtWidgets
)
from tensorflow import keras
from .LayerTreeItem import LayerTreeItem

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..Utils import (
    PredictionHandler,
    OutputHandler,
    KernelHandler,
    HandlerConfiguration
    )
class LayersTreeModel(QtCore.QAbstractItemModel):#https://doc.qt.io/qt-5/qtwidgets-itemviews-simpletreemodel-example.html
    def __init__(self, parent: Optional[QtCore.QModelIndex] = None, widget : Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.__widget : QWidget = widget

        self.__rootItem = LayerTreeItem(["Name", "Description"])
        #Descomentar para que aparezca el nombre del modelo como raiz del resto:
        #self.__rootName = "Modelo"
        #self.__parentRoot = LayerTreeItem([self.getRootName],self.__rootItem)
        #self.__parentRoot.appendChild(LayerTreeItem(["Dummy"],self.__parentRoot))
        #self.__rootItem.appendChild(self.__parentRoot)
        self.__layers : List[keras.layers.Layer] = []
        return

    @property
    def rootItem(self):
        return self.__rootItem

    """
    def getRootName(self):
        return self.__rootName
    def setRootName(self,val):
        self.__rootName = val
    """

    def clear(self):
        self.rootItem.clearChilds()
        self.modelReset.emit()
        return
    def columnCount(self, parent : LayerTreeItem) -> int:
        return self.__rootItem.columnCount()
     
    def rowCount(self, parent : LayerTreeItem) -> int:
        if parent.column() > 0:
            return 0

        if not parent.isValid():
            parentItem = self.__rootItem
        else:
            parentItem = parent.internalPointer()

        return parentItem.childCount()

    def flags(self, index : QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        if not index.isValid():
            return QtCore.Qt.NoItemFlags
        item =  index.internalPointer()
        flags = QtCore.Qt.ItemIsSelectable |  QtCore.Qt.ItemIsDragEnabled if item.hasEvent() else 0
        isEditable = QtCore.Qt.ItemIsEditable if type(item.data(index.column())) is list else 0
        return QtCore.Qt.ItemIsEnabled | flags | isEditable

    def headerData(self, section : int , orientation : QtCore.Qt.Orientation, role : int):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self.__rootItem.data(section)

        return None
    #######################################
    def index(self, row : int, column : int, parent : QtCore.QModelIndex = QtCore.QModelIndex()) ->QtCore.QModelIndex:
        if not self.hasIndex(row, column, parent):
            return QtCore.QModelIndex()

        if not parent.isValid():
            parentItem = self.__rootItem
        else:
            parentItem = parent.internalPointer()

        childItem = parentItem.child(row)
        if childItem:
            return self.createIndex(row, column, childItem)
        else:
            return QtCore.QModelIndex()

    def parent(self, index : QtCore.QModelIndex) -> QtCore.QModelIndex:
        if not index.isValid():
            return QtCore.QModelIndex()

        childItem = index.internalPointer()
        parentItem = childItem.parent()
        if parentItem == self.__rootItem:
            return QtCore.QModelIndex()

        return self.createIndex(parentItem.row(), 0, parentItem)
    ###############################################
    def data(self, index : QtCore.QModelIndex , role : int = QtCore.Qt.DisplayRole):

        if not index.isValid():
            return None

        if role != QtCore.Qt.DisplayRole:
            return None

        item = index.internalPointer()
        #if index.column() < item.columnCount():
        actualData = item.data(index.column())
        if type(actualData) is list:
            actualData = actualData[0]
        return actualData
        #return None


    def loadLayers(self,layers: List[keras.layers.Layer]):
        """
        Set a new `layers` array.
        """
        self.__layers = layers
        self.rootItem.clearChilds()
        self.setupModelData(layers,self.rootItem)
        # Model has been reset, redraw view
        self.modelReset.emit()
        return

    def setupModelData(self, layers: List[keras.layers.Layer], parent: LayerTreeItem):
        """
            Este metodo es el responsable de crear y cargar en el widget la información de la capa pasada como argumento.
            Dado que es mucha información, los padres de varios nodos implementan una carga diferida que permite aliviar la carga
            de trabajo de forma que solo se cargue la información cuando se necesita. Esta carga diferida funciona gracias a la sobrecarga
            de los metodos :
                'canFetchMore': Llamado por el controlador, pregunta si hay mas datos que cargar, si le pasamos True el mismo controlador llama a:
                'fetchMore' : Llamado por el controlador y es responsabilidad del programador indicar en este metodo como se van a cargar los datos que faltan.
        """
        for layer in layers:
            child = LayerTreeItem([layer.name,str(type(layer).__name__)],parent)
            if hasattr(layer,'bias'):
                childBias = LayerTreeItem(["Bias","{}".format(layer.bias.shape[0])],child)
                childBias.setDelayedLoadFunction(self.setupBiasData,[layer.bias,childBias])
                childBias.appendChild(LayerTreeItem(["Dummy"],childBias)) #Para poder ejecutar el lazyLoad
                #self.setupBiasData(layer.bias,childBias)
                child.appendChild(childBias)

            if hasattr(layer,'kernel'):
                height, width, channels,_ = layer.kernel.shape
                childKernel = LayerTreeItem(["Kernel","{}x{}x{}".format(height,width,channels)],child)
                childKernel.setDelayedLoadFunction(self.setupKernelData,[layer.kernel,childKernel])
                #childKernel.setEventFunction(self.__widget.onGroupKernelSelect,[layer.kernel.numpy()])
                childKernel.appendChild(LayerTreeItem(["Dummy"],childKernel)) #Para poder ejecutar el lazyLoad
                #self.setupKernelData(layer.kernel,childKernel)
                child.appendChild(childKernel)

            #childOutput = LayerTreeItem(["Output",["Guided BackPropagation","Score-CAM"]],child)
            childOutput = LayerTreeItem(["Output", list(HandlerConfiguration.getHandler(OutputHandler).keys())],child)
            childOutput.setDelayedLoadFunction(self.setupOutputData,[layer.output,childOutput])
            childOutput.appendChild(LayerTreeItem(["Dummy"],childOutput)) #Para poder ejecutar el lazyLoad
            childOutput.setEventFunction(self.__widget.onGroupOutputSelect,[layer.output, layer.name, childOutput.data])
            #childOutput.setEventFunction(self.__widget.onGroupOutputSelect,[layer.output, "Output of:{} - Fast-ScoreCam".format(layer.name)])
            child.appendChild(childOutput)

            parent.appendChild(child)

        return

    def setupOutputData(self,outputData : 'KerasTensor',parent : LayerTreeItem) -> None:
        """
            Metodo para carga diferida de los hijos del nodo padre "Output"
        """
        outputName = "output_{}" #Not Used
        for i in range(outputData.shape[3]):
            #child = LayerTreeItem([i,"Feature Map+Gradient Ascent"],parent) #ToDo: Que el segundo elemento sea un dropView, para seleccionar el tipo de muestreo...
            child = LayerTreeItem([i],parent) #ToDo: Que el segundo elemento sea un dropView, para seleccionar el tipo de muestreo...
            child.setEventFunction(self.__widget.onSingleOutputSelect,[outputData,i,"{}. Index:{} - Feature Map & Gradient Descent".format(outputData.name,i)])
            parent.appendChild(child)
        return
    def setupBiasData(self,biasData : 'ResourceVariable',parent : LayerTreeItem) -> None:
        """
            Metodo para carga diferida de los hijos del nodo padre "Bias"
        """
        biasName = "bias_{}" #Not Used
        for i in range(biasData.shape[0]):
            child = LayerTreeItem([i, str(biasData[i].numpy())],parent)
            parent.appendChild(child)
        return

    def setupKernelData(self,kernelData : 'ResourceVariable',parent : LayerTreeItem) -> None:
        """
            Metodo para carga diferida de los hijos del nodo padre "Kernel"
        """
        kernelName = "kernel_{}" #Not Used
        if kernelData.shape.ndims == 4: #Suponemos Convolucional. El unico caso que nos importa por ahora
            for i in range(kernelData.shape[-1]): #Número salidas
                #By tensorflow convention, for convolutional kernels, their shape-format is (kernel_height, kernel_width, input_channels, output_channels)
                child = LayerTreeItem([i],parent)
                child.setDelayedLoadFunction(self.setupKernelChannels,[kernelData,i,child])
                child.setEventFunction(self.__widget.onSingleKernelSelect,[ kernelData[:,:,:,i].numpy(), "{}.Index:{}".format(kernelData.name,i) ])
                child.appendChild(LayerTreeItem(["Dummy"],child)) #Para poder ejecutar el lazyLoad
                #self.setupKernelChannels(kernelData,child)
                parent.appendChild(child)
        return

    def setupKernelChannels(self,kernelData : 'ResourceVariable',outputIndex : int ,parent : LayerTreeItem ) -> None:
        for c in range(kernelData.shape[-2]):#Número canales
            child = LayerTreeItem([c,str(kernelData[:,:,c,outputIndex].numpy()).strip()],parent)
            parent.appendChild(child)
        return

    #######################################################################################################################
    ###Lazy Load main functions###
    def canFetchMore(self, parent : QtCore.QModelIndex) -> bool:
        """
            Cada ocasión que se abre una lista, es llamado este metodo para preguntar si el elemento padre
            tiene hijos que aún no han sido cargados
        """
        if not parent.isValid():
            return False
        item = parent.internalPointer()
        return item.isLoadDelayed()

    def fetchMore(self,parent : QtCore.QModelIndex):
        """
            Este metodo es llamado cuando canFetchMore devuelve True
        """
        item = parent.internalPointer()
        item.clearChilds() #Eliminamos el dummy adhoc añadido anteriormente para poder alcanzar este punto
        item.callDelayedLoadFunction()
        return
############################################ToDo--> Implement it:https://stackoverflow.com/a/47851155
