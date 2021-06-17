
import os
import PySide2
import dependency_injector.providers as providers

current_dir = os.path.dirname(os.path.abspath(__file__))
Form, Base = PySide2.QtUiTools.loadUiType(os.path.join(current_dir, "./PreProcessorLoader.ui"))

class PreProcessorLoaderWidget(Form, Base):
    def __init__(self, parent: PySide2.QtWidgets.QWidget = None):
        super(self.__class__, self).__init__(parent)
        self.setupUi(self)
        
        self.buttonSelectScript.clicked.connect(self.onSelectScriptClick)
        self.__dict = {}
        return

    def send_preprocessor_function(self):
        key = self.comboBoxFunctions.itemText(self.comboBoxFunctions.currentIndex())
        return self.__dict[key]

    def onSelectScriptClick(self,event):
        options =  PySide2.QtWidgets.QFileDialog.Options()
        scriptPath, _ = PySide2.QtWidgets.QFileDialog.getOpenFileName(self,"Open Script file", "", "Python Files(*.py)", options=options)
        if scriptPath != '':
            self.lineScriptPath.setText(scriptPath)
            with open(scriptPath,'r') as scriptFile:
                input = scriptFile.read()
                exec(input,globals(),self.__dict)
                self.comboBoxFunctions.clear()
                for key in self.__dict:
                    if callable(self.__dict[key]):
                        self.comboBoxFunctions.addItem(str(key))
            
            return

PreProcessorLoaderWidgetFactory = providers.Factory(PreProcessorLoaderWidget)