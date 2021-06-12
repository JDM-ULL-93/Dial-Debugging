# vim: ft=python fileencoding=utf-8 sts=4 sw=4 et:

"""This package has the basic nodes that can be placed on the Node Editor.

From editing datasets to compiling models, this nodes should satisfy most of the needs
when working with classical Deep Learning problems.
"""

from dial_core.node_editor import NodeRegistrySingleton
from dial_core.notebook import NodeCellsRegistrySingleton


from conv_debugger import(
    ConvDebuggerNode,
    ConvDebuggerNodeCells,
    ConvDebuggerNodeFactory
)

from preprocessor_loader import(
    PreProcessorLoaderNode,
    PreProcessorLoaderNodeFactory
)


def load_plugin():
    node_registry = NodeRegistrySingleton()

    # Register Node
    node_registry.register_node(
        "Debugging/Convolutional Debugger", ConvDebuggerNodeFactory
     )

    node_registry.register_node(
        "Debugging/Image Preprocessor Loader", PreProcessorLoaderNodeFactory
     )

    # Register Notebook Transformers
    node_cells_registry = NodeCellsRegistrySingleton()
    node_cells_registry.register_transformer(
        ConvDebuggerNode, ConvDebuggerNodeCells
    )


def unload_plugin():
    node_registry = NodeRegistrySingleton()

    # Unregister Nodes
    node_registry.unregister_node("Debugging/Convolutional Debugger")
    node_registry.unregister_node("Debugging/Image Preprocessor Loader")

    # Unregister Notebook Transformers
    node_registry.unregister_node(ConvDebuggerNodeCells)


#load_plugin()

__all__ = [
    "load_plugin",
    "unload_plugin",
]
