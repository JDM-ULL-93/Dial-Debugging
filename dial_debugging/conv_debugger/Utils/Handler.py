class Handler(object):
    """
        Clase base que sirve de plantilla para los handlers que manejaran los datos de
        las capas para hacerles una operación de debugging
    """
    def __init__(self):
        self.custom_handlers = {}
        return

    def addCustomHandler(self,key:str,function):
        if not callable(function):
            raise Exception("function argument must be callable")
        self.custom_handlers[key] = function
    def deleteCustomHandler(self,key:str):
        if key in self.custom_handlers:
            del self.custom_handlers[key]
        return

    def __call__(self,**kwargs):
        """
            Este metodo es el unico que debe ser llamado desde fuera y apartir de los argumentos,
            se decidirá la acción a realizar
        """
        raise NotImplementedError("Virtual function __call__ has not been overrided")
        return 

    def mathPlot(self,imgs ,titles, cmaps, gridRows = 1, gridCols = 1 ):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(gridRows, gridCols)
        for i in range(gridRows):
            for j in range(gridCols):
                x = i*gridCols+j
                if x > len(imgs): break
                if gridCols == 1 and gridRows == 1:
                    ax.imshow(imgs[x], cmap=cmaps[x])
                    ax.set_title(titles[x])
                elif gridCols == 1:
                    ax[i].imshow(imgs[x], cmap=cmaps[x])
                    ax[i].set_title(titles[x])
                elif gridRows == 1:
                    ax[j].imshow(imgs[x], cmap=cmaps[x])
                    ax[j].set_title(titles[x])
                else:
                    ax[i,j].imshow(imgs[x], cmap=cmaps[x])
                    ax[i,j].set_title(titles[x])
        plt.show()
        return



