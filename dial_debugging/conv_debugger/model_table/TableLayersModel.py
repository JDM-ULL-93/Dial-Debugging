from PySide2 import QtCore

from tensorflow import keras

from typing import TYPE_CHECKING, Any, Dict, List, Optional


class TableLayersModel(QtCore.QAbstractTableModel): #https://doc.qt.io/qt-5/modelview.html
    from enum import IntEnum
    class Column(IntEnum):
        Type = 0
        Name = 1
        Input_Shape = 2
        Output_Shape = 3

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.__layers: List[keras.layers.Layer] = []
        return

    def loadLayers(self,layers: List[keras.layers.Layer]):
        """
        Set a new `layers` array.
        """
        self.__layers = layers

        # Model has been reset, redraw view
        self.modelReset.emit()
        return

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        """
        Return the number of rows.
        """
        return len(self.__layers)

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:
        """
        Return the number of columns.
        """
        return len(self.Column);

    def headerData(
        self, section: int, orientation: QtCore.Qt.Orientation, role: int = QtCore.Qt.DisplayRole
    ) -> Optional[str]:
        """
        Return the name of the headers.
        Horizontal header   -> Column names
        Vertical header     -> Row nº
        """
        if role != QtCore.Qt.DisplayRole:
            return None

        # Column header must have their respective names
        if orientation == QtCore.Qt.Horizontal:
            return str(self.Column(section).name)

        # Row header will have the row number as name
        if orientation == QtCore.Qt.Vertical:
            return str(section)

        return None
    def flags(self, index : QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        if not index.isValid():
            return QtCore.Qt.NoItemFlags
        row = index.row()
        layer = self.__layers[row]
        flags = QtCore.Qt.NoItemFlags
        if layer.input_spec is not None and (layer.input_spec.ndim == 4 or layer.input_spec.min_ndim == 4):
            flags = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        return flags

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole) -> Optional[Any]:
        """ Returns data depending on the specified role.

        Args:
            index: Index representing the item on the table.
            role: Data access role.

        Returns:
            The data of the `index` item for `role` or `None`.
        """
        if role == QtCore.Qt.DisplayRole:
            row = index.row()
            col = index.column()
            layer = self.__layers[row]
           
            if col == self.Column.Type:
                return str(type(layer).__name__)
            elif col == self.Column.Name:
                return str(layer.name)
            elif col == self.Column.Input_Shape:
                return str(layer.input_shape)
            elif col == self.Column.Output_Shape:
                return str(layer.output_shape)
            return None

        # Center align all text
        if role == QtCore.Qt.TextAlignmentRole:
            return (QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        return None
############################################
