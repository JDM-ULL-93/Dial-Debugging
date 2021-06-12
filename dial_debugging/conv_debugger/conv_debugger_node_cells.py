import nbformat as nbf
from dial_core.notebook import NodeCells


class ConvDebuggerNodeCells(NodeCells):
    """The ConvDebuggerNodeCells class generates a block of code corresponding
    to the hyperparameters dictionary."""

    def _body_cells(self):
        return [nbf.v4.new_code_cell("# TODO: Implement later")]
