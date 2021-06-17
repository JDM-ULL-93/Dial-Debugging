
#import PySide2
from typing import TYPE_CHECKING, Any, Dict, List, Optional,Callable, Tuple
#class TreeItem(PySide2.QtWidgets.QTreeWidget):
class TreeItem():
    def __init__(self, data : List, parent : 'TreeItem' = None):
        self.parentItem : 'TreeItem' = parent
        self.itemData : List = data
        self.childItems : List['TreeItem'] = []
        return
    ###################################
    #Clear STUFF
    def __del__(self):
        self.clearChilds()
        self.itemData = None
        return
    def clearChilds(self):
        for childItem in self.childItems:
            del childItem
        self.childItems.clear()
        return
    ###################################
    def appendChild(self, item : "TreeItem") ->None:
        self.childItems.append(item)
        return

    def child(self, row : int) -> 'TreeItem':
        return self.childItems[row]

    def childCount(self) -> int:
        return len(self.childItems)

    def columnCount(self) -> int:
        return len(self.itemData)

    def data(self, column : int) -> str:
        try:
            data = self.itemData[column]
            if callable(data):
                return data()
            else:
                return data
        except IndexError:
            return None

    def parent(self) -> 'TreeItem':
        return self.parentItem

    def row(self) -> int:
        if self.parentItem:
            return self.parentItem.childItems.index(self)
        return 0
#######################################

class LayerTreeItem(TreeItem):
    def __init__(self, data : List, parent : TreeItem = None, layer :'Keras.Layer'= None):
        super(LayerTreeItem, self).__init__(data,parent)
        #self.layer = layer

        self.delayedFunction : Callable = None
        self.delayedFunctionArgs : List = []
        self.delayedLoad : bool = False

        self.eventFunction : Callable = None
        self.eventFunctionArgs : List = []
        return

    def __del__(self):
        super(LayerTreeItem, self).__del__()
        self.layer = None
        self.delayedFunction = None
        self.delayedFunctionArgs = None
        self.eventFunction = None
        self.eventFunctionArgs = None
        return
    #####LAZY LOAD#####
    def setDelayedLoadFunction(self,function : Callable, args : List):
        self.delayedFunction = function
        self.delayedFunctionArgs = args
        self.delayedLoad = True
        return
    def callDelayedLoadFunction(self):
        self.delayedFunction(*self.delayedFunctionArgs)
        self.delayedLoad = False
        return
    def isLoadDelayed(self):
        return self.delayedLoad

    #####EVENT HANDLER#####
    def setEventFunction(self,eventFunction : Callable, args : List):
        self.eventFunction = eventFunction
        self.eventFunctionArgs = args
        return
    def callEvent(self,dragMode:bool=True,executeMode:bool=True,**kwargs):
        if self.eventFunction is not None:
             return self.eventFunction(*self.eventFunctionArgs,dragMode=dragMode,executeMode=executeMode,**kwargs)
        return None
    def hasEvent(self)->bool:
        return self.eventFunction is not None

    #####OTHERS#####
    #def getLayer(self,Layer):
        #return self.layer

#############################################
