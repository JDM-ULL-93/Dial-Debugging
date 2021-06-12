import PySide2
from .EQTreeView import EQTreeView

#ToDo:https://stackoverflow.com/questions/62258969/drag-item-from-qtreewidget-to-qgraphicsview
class EQGraphicsView(PySide2.QtWidgets.QGraphicsView):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self.__dropEventHandler = None
        return

    @property
    def dropEventHandler(self):
        return self.__dropEventHandler

    @dropEventHandler.setter
    def dropEventHandler(self,value):
        self.__dropEventHandler = value
        return

    def dragEnterEvent(self, event):
        if type(event.source()) == EQTreeView:#PySide2.QtWidgets.QTreeView:
            event.acceptProposedAction()
        return

    def dragMoveEvent(self, event : 'QDragMoveEvent'):
        event.acceptProposedAction()
        return

    def dropEvent(self,event : 'QDropEvent'):
        if self.__dropEventHandler is not None:
            self.__dropEventHandler(event)
        return
