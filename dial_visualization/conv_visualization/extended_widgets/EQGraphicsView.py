import PySide2
from .EQTreeView import EQTreeView

#Done:https://stackoverflow.com/questions/62258969/drag-item-from-qtreewidget-to-qgraphicsview
from typing import TYPE_CHECKING, Any, Dict, List, Optional,Callable, Tuple
class EQGraphicsView(PySide2.QtWidgets.QGraphicsView):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self.__dropEventHandler : Callable = None
        return

    @property
    def dropEventHandler(self) ->Callable:
        return self.__dropEventHandler

    @dropEventHandler.setter
    def dropEventHandler(self,value : Callable):
        self.__dropEventHandler = value
        return

    def dragEnterEvent(self, event : PySide2.QtGui.QDragEnterEvent):
        if type(event.source()) == EQTreeView:#PySide2.QtWidgets.QTreeView:
            event.acceptProposedAction()
        return

    def dragMoveEvent(self, event : PySide2.QtGui.QDragMoveEvent):
        event.acceptProposedAction()
        return

    def dropEvent(self,event : PySide2.QtGui.QDropEvent ):
        if self.__dropEventHandler is not None:
            self.__dropEventHandler(event)
        return
