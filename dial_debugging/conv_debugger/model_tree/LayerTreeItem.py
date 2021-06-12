import PySide2

class TreeItem(PySide2.QtWidgets.QTreeWidget):
    def __init__(self, data, parent=None):
        self.parentItem = parent
        self.itemData = data
        self.childItems = []
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
    def appendChild(self, item : "TreeItem"):
        self.childItems.append(item)
        return

    def child(self, row : int):
        return self.childItems[row]

    def childCount(self):
        return len(self.childItems)

    def columnCount(self):
        return len(self.itemData)

    def data(self, column : int):
        try:
            data = self.itemData[column]
            if callable(data):
                return data()
            else:
                return data
        except IndexError:
            return None

    def parent(self):
        return self.parentItem

    def row(self):
        if self.parentItem:
            return self.parentItem.childItems.index(self)
        return 0
#######################################

from tensorflow import keras
class LayerTreeItem(TreeItem):
    def __init__(self, data, parent=None, layer = None):
        super(LayerTreeItem, self).__init__(data,parent)
        self.layer = layer

        self.delayedFunction = None
        self.delayedFunctionArgs = []
        self.delayedLoad = False

        self.eventFunction = None
        self.eventFunctionArgs = []
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
    def setDelayedLoadFunction(self,function, args):
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
    def setEventFunction(self,eventFunction, args):
        self.eventFunction = eventFunction
        self.eventFunctionArgs = args
        return
    def callEvent(self, dragMode : bool):
        if self.eventFunction is not None:
             return self.eventFunction(*self.eventFunctionArgs, dragMode)
        return None
    def hasEvent(self)->bool:
        return self.eventFunction is not None

    #####OTHERS#####
    def getLayer(self,Layer):
        return self.layer
#############################################
