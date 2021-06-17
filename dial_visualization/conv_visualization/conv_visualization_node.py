from typing import TYPE_CHECKING, List

import dependency_injector.providers as providers
from dial_core.node_editor import Node

from .conv_visualization_widget import ConvVisualizationWidgetFactory
from .conv_visualization_widget import ConvVisualizationWidget

from tensorflow.keras import Model
from typing import Callable

class ConvVisualizationNode(Node):
    def __init__(
        self, conv_debugger_widget: ConvVisualizationWidget,
    ):
        super().__init__(
            title="Convolutional Debugger", inner_widget=conv_debugger_widget,
        )

        self.add_input_port("Model", port_type=Model)
        self.inputs["Model"].set_processor_function(
            self.inner_widget.set_trained_model
        )

        self.add_input_port("Custom Preprocessor", port_type=Callable)
        self.inputs["Custom Preprocessor"].set_processor_function(
            self.inner_widget.set_preprocessorFunction
        )
        return

    #def __reduce__(self): #Esto es para uso interno de pickle, para serializar objetos sin necesidad de instanciarlos
        #return (HyperparametersConfigNode, (self.inner_widget,), super().__getstate__())


ConvVisualizationNodeFactory = providers.Factory(
    ConvVisualizationNode,
    conv_debugger_widget=ConvVisualizationWidgetFactory,
)

