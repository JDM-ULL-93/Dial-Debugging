import PySide2

#ToDo: Implement custom mimeData:
#https://doc.qt.io/qt-5/dnd.html#dragging
class EQTreeView(PySide2.QtWidgets.QTreeView):
    def __init__(self,parent = None):
        super().__init__(parent)
        self.setSelectionMode(PySide2.QtWidgets.QAbstractItemView.ContiguousSelection)
        self.setDragDropMode(PySide2.QtWidgets.QAbstractItemView.DragOnly)
        self.setDragEnabled(True)
        self.setEditTriggers(PySide2.QtWidgets.QAbstractItemView.CurrentChanged)

        #self.__dragStartPosition = None
        #self.__draggedItem = None
        return

    def mouseDoubleClickEvent(self, event : PySide2.QtGui.QMouseEvent):
        if len(self.selectionModel().selectedRows()) > 0:
            index : PySide2.QtCore.QModelIndex = self.indexAt(event.pos())
            if (int(index.flags()) & PySide2.QtCore.Qt.ItemFlag.ItemIsEditable): #Entonces priorizamos el evento que produce el evento 'editar'(por defecto 'mousedoubleclick') sobre el de llamar al evento.
                super().mouseDoubleClickEvent(event)
            elif index.internalPointer().hasEvent():
                item = index.internalPointer()
                item.callEvent(dragMode=False)
        return

    ################################################################################################################################################
    ################################################################################################################################################
    ##################METODOS NO USADOS. Sirven para documentar como es todo el proceso involucrado en el evento 'DragItem' por debajo.
    def mousePressEvent(self,event : PySide2.QtGui.QMouseEvent):
        super().mousePressEvent(event)
        #if event.button() == PySide2.QtCore.Qt.LeftButton:
            #self.__dragStartPosition = event.pos()
            #self.__draggedItem = self.indexAt(event.pos()) #Para recoger de memoria el item sobre el que el cursor se encuentra
        return
    def mouseMoveEvent(self,event : PySide2.QtGui.QMouseEvent):
        super().mouseMoveEvent(event)
        return
        #Checks:
        """ #Originalmente tenia pensado que me hacia falta modificar el MimeData interno,lo cual,
            #entre otras cosas, obligaba a re-implementar el efecto de arrastrar un item,luego me di cuenta que no hacia falta
            #para nada

        if not(event.buttons() & PySide2.QtCore.Qt.LeftButton):
            return#Entre los botones pulsados no se encuentra el boton izquierdo
        if (event.pos() - self.__dragStartPosition).manhattanLength() < PySide2.QtWidgets.QApplication.startDragDistance():
            return#La distancia para considerar que se esta arrastrando desde el comienzo al mayor excede a la marca de 10
        #Actions:
        ## Aqui emulamos el arrastrar tradicional
        drag = PySide2.QtGui.QDrag(self)
        mimeData = PySide2.QtCore.QMimeData()

        #mimeData.setData("application/pickle-stream",self.serialize())
        mimeData.setText("Hola Mundo")
        drag.setMimeData(mimeData)

        pixmap, offset = self.__renderSelectedItemsToPixmap(event)
        drag.setPixmap(pixmap)
        drag.setHotSpot(offset) #Posición del cursor
        dropAction = drag.exec_(PySide2.QtCore.Qt.CopyAction | PySide2.QtCore.Qt.MoveAction)
        return
        """
    def __getItemColumnsCount(self,item):
        count = 0
        while item.siblingAtColumn(count).column() != -1:
            count += 1
        return count
    def __getRowItemSize(self, item):
        numColumns = self.__getItemColumnsCount(item)
        itemRect = self.visualRect(item)
        size = PySide2.QtCore.QSize(0,0)
        for i in range(numColumns):
            itemRect = self.visualRect(item)
            size.setHeight(itemRect.height()*len(self.selectionModel().selectedRows()))
            size.setWidth(size.width()+itemRect.width())
        return size

    def __renderSelectedItemsToPixmap(self,event):
        targetItemFromToRender = self.selectionModel().selectedRows()[0].siblingAtColumn(0) #Trabajamos siempre con el 1º elemento de la 1º fila
        itemRect = self.visualRect(targetItemFromToRender) #Nos sirve para conocer la posición  x e y en la que esta localizado (ignoramos el resto)
        #itemSize = self.sizeHintForIndex(self.__draggedItem)
        headerSize = self.header().size() #Por defecto no tiene encuenta el offset que la cabecera produce, lo seleccionamos para obtener el offset de altura
        itemSize = self.__getRowItemSize(targetItemFromToRender)#Obtenemos la suma de anchos de cada item de cada columna de la fila
        offset =  PySide2.QtCore.QPoint(event.pos().x()-itemRect.x(),event.pos().y()-itemRect.y())

        pixmap = PySide2.QtGui.QPixmap(itemSize)
        self.render(pixmap,
                   sourceRegion=PySide2.QtGui.QRegion(itemRect.x(),
                                                      itemRect.y()+headerSize.height(),
                                                      itemSize.width(),
                                                      itemSize.height())
                   )
        return pixmap, offset #Devuelve tambien la posición sobre la que el cursor debe situarse respecto al pixmap generado.

