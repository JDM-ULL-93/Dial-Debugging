import PySide2
from typing import TYPE_CHECKING, Any, Dict, List, Optional

class LayerStyledDelegateItem(PySide2.QtWidgets.QStyledItemDelegate):
    def __init__(self, parent: Optional[PySide2.QtCore.QModelIndex] = None):
        super().__init__(parent)
        self.blockSignals(True) #Importante para que se pueda abrir el combobox sin que se destruya al momento
        return

    #Llamado el resto del tiempo que no se esta editando el control
    def paint(self,painter : PySide2.QtGui.QPainter, option : PySide2.QtWidgets.QStyleOptionViewItem, index : PySide2.QtCore.QModelIndex):
        item = index.internalPointer()
        data = item.data(index.column())
        if type(data) is list:
            comboBoxStyleOption = PySide2.QtWidgets.QStyleOptionComboBox()
            #comboBoxStyleOption.initFrom(PySide2.QtWidgets.QComboBox())
            comboBoxStyleOption.rect = option.rect;
            comboBoxStyleOption.currentText = data[0]
            comboBoxStyleOption.state = PySide2.QtWidgets.QStyle.State_Enabled
            PySide2.QtWidgets.QApplication.style().drawComplexControl(PySide2.QtWidgets.QStyle.CC_ComboBox,
                                                                     comboBoxStyleOption, painter);
            PySide2.QtWidgets.QApplication.style().drawControl(PySide2.QtWidgets.QStyle.CE_ComboBoxLabel,
                                                               comboBoxStyleOption, painter )
        else:
            super().paint(painter,option,index) #Comportamiento por defecto: Dibujar texto plano
        return

    #Called when a field is starting to be edited (before input control is printed)
    def createEditor(self,parent : PySide2.QtWidgets.QWidget,  option :PySide2.QtWidgets.QStyleOptionViewItem , index : PySide2.QtCore.QModelIndex ):
        item = index.internalPointer()
        data = item.data(index.column())
        if type(data) is list: #Solo nos interesa cambiar el comportamiento y dibujar otro input cuando se trate de una lista
            cellComboBox = PySide2.QtWidgets.QComboBox(parent)
            for value in data:
                cellComboBox.addItem(value)
            return cellComboBox #return super().createEditor(parent,option,index)
        else:
            return super().createEditor(parent,option,index) #Comportamiento por defecto

    def setEditorData(self,editor : PySide2.QtWidgets.QWidget,index : PySide2.QtCore.QModelIndex):
        #editor.setCurrentText("Una prueba");
        return
    #https://stackoverflow.com/questions/48892319/using-a-qcombobox-delegate-item-within-a-qtreeview
    #Called after editing is finished to set up model data
    def setModelData(self,editor : PySide2.QtWidgets.QWidget , model : PySide2.QtCore.QAbstractItemModel, index : PySide2.QtCore.QModelIndex):
        item = index.internalPointer()
        data = item.data(index.column())
        if  type(data) is list: #Caso combobox
            data.remove(editor.currentText()) #El elemento seleccionado será el que se encuentre en la 1º posición de la lista
            data.insert(0,editor.currentText())
        else: #Otros casos
            super.setModelData(editor,model,index)
        return

