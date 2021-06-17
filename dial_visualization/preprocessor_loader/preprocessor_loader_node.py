from dial_core.node_editor import Node

from .preprocessor_loader_widget import PreProcessorLoaderWidgetFactory
from .preprocessor_loader_widget import PreProcessorLoaderWidget

import dependency_injector.providers as providers
from tensorflow.keras import Model
from typing import TYPE_CHECKING,Callable

class PreProcessorLoaderNode(Node):
    def __init__(
        self, preprocessor_loader_widget: PreProcessorLoaderWidget,
    ):
        super().__init__(
            title="Image Preprocessor Loader", inner_widget=preprocessor_loader_widget,
        )

        self.add_output_port(name="Preprocessor Function", port_type=Callable)
        self.outputs["Preprocessor Function"].set_generator_function(
            self.inner_widget.send_preprocessor_function
        )
        return

    #def __reduce__(self): #Esto es para uso interno de pickle, para serializar objetos sin necesidad de instanciarlos
        #return (HyperparametersConfigNode, (self.inner_widget,), super().__getstate__())


PreProcessorLoaderNodeFactory = providers.Factory(
    PreProcessorLoaderNode,
    preprocessor_loader_widget=PreProcessorLoaderWidgetFactory,
)
