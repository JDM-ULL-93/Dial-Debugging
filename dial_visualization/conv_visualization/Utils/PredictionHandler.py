import numpy as np
import tensorflow as tf

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, Callable
from .Handler import Handler
class PredictionHandler(Handler):
    def __init__(self, model : tf.keras.Model, preprocessorFunction : Callable):
        super(PredictionHandler, self).__init__(None,PredictionHandler)
        self.__model : tf.keras.Model = model
        self.__preprocessor : Callable = preprocessorFunction
        self.__predictions : np.ndarray = None
        self.__dataHandler : Callable = None
        return

    @property
    def predictions(self):
         return self.__predictions

    @property
    def dataHandler(self) -> None:
        return self.__dataHandler
    @dataHandler.setter
    def dataHandler(self,value : Callable) -> None:
        self.__dataHandler = value
        return

    def doPrediction(self,img_array : np.ndarray) -> None:
        processed = self.__preprocessor(img_array.copy())
        self.__predictions = self.__orderPrediction(self.__model(processed)[0].numpy())
        return

    def __orderPrediction(self,predictions : np.ndarray) -> np.ndarray:
        result = []
        while predictions[np.argmax(predictions)] != -1:
            index = np.argmax(predictions)
            value = predictions[index]
            if self.__dataHandler is not None:
                self.__dataHandler(index,value)
            predictions[index] = -1
            result.append([index,value])
        return result

    #https://doc.qt.io/qt-5/qtcharts-examples.html
    def plotPrediction(self, width : int, height : int, topN : int) ->'QtCharts.QChart':
        from PySide2.QtCharts import QtCharts
        series = QtCharts.QPieSeries()
        series.setHoleSize(0.35)

        sum = self.__predictions[0][1]
        slice = QtCharts.QPieSlice()
        percentage = round(self.__predictions[0][1]*100,2)
        label = "{}[{}%]".format(self.__predictions[0][0],percentage)
        slice = series.append(label, percentage)
        slice.setExploded()                                            
        slice.setLabelVisible()                                        
        
        for prediction in self.__predictions[1:topN]:
            sum+=prediction[1]
            #percentage = round(prediction[1]*100,2)
            #label = "{}[{}%]".format(prediction[0],percentage)
            label = "{}".format(prediction[0])
            serie = series.append(label, percentage)
            serie.setLabelVisible()           
          
        slice = QtCharts.QPieSlice()
        percentage = round((1-sum)*100,2)
        label = "Rest[{}%]".format(percentage)
        slice = series.append(label, percentage)
        slice.setExploded()                                            
        slice.setLabelVisible() 

        chart = QtCharts.QChart()
        #chart.legend().hide()
        chart.addSeries(series)

        chart.setAnimationOptions(QtCharts.QChart.SeriesAnimations)
        chart.setTitle("Prediction. Top 5")
        chart.setTheme(QtCharts.QChart.ChartThemeQt)
        chart.resize(width,height)
        return chart


    def __call__(self,**kwargs):
        #Cargamos los argumentos esperados (que no tienen porque ser todos):
        who,dragMode=kwargs["who"],kwargs["dragMode"]
        #Proceso:
        if who in self._handlers:
            img_array,predictionIndex = kwargs["img_array"],kwargs["predictionIndex"]
            heatMap = None
            if who == "Grad-CAM":
                heatMap = PredictionHandler.doGradCAM(self,
                                                      img_array,
                                                      predictionIndex)
            elif who == "Vanilla Occlussion":
                occluding_size,occluding_pixel,occluding_stride = kwargs["occluding_size"],kwargs["occluding_pixel"],kwargs["occluding_stride"]
                heatMap = PredictionHandler.doVanillaOcclusion(self,
                                                               img_array,
                                                               predictionIndex,
                                                               occluding_size=occluding_size,
                                                               occluding_pixel=occluding_pixel,
                                                               occluding_stride=occluding_stride)
            elif who =="Integrated Gradients":
                num_steps,batch_size = kwargs["num_steps"],kwargs["batch_size"]
                heatMap = PredictionHandler.doIntegratedGradients(self,
                                                                  img_array,
                                                                  predictionIndex,
                                                                  num_steps=num_steps,
                                                                  batch_size=batch_size)
            else: #Custom one
                kwargs["self"] = self
                kwargs["model"] = self.__model
                kwargs["preprocessorFunction"] = self.__preprocessor
                return self._handlers[who](**kwargs)
            who = "Prediction Index: {} - {}".format(predictionIndex,who)
            if dragMode:
                return heatMap,who
            else:
                self.mathPlot([heatMap],[who],[None],1,1)
        return
    ################################################################
    ##GradCAM
    @staticmethod
    def doGradCAM(self,
                  img_array : np.ndarray,
                  predictIndex : int) ->np.ndarray:
        from .Algorithms.GradCAM import GradCAM
        return GradCAM(self.__model,self.__preprocessor)(img_array,predictIndex)
    
    ################################################################
    ##vanillaOcclusion
    @staticmethod
    def doVanillaOcclusion(self,
                         img_array : np.ndarray, 
                         class_index : int,
                         *,
                         occluding_size:int=8,
                         occluding_pixel:int=0,
                         occluding_stride:int=8) ->np.ndarray:

        from .Algorithms.VanillaOclussion import VanillaOclussion
        return VanillaOclussion(self.__model,self.__preprocessor)(img_array,
                                                                  class_index,
                                                                  occluding_size=occluding_size,
                                                                  occluding_pixel=occluding_pixel,
                                                                  occluding_stride=occluding_stride)
    ################################################################
    ##Integrated Gradients
    @staticmethod
    def doIntegratedGradients(self,
                              img_array : np.ndarray,
                              class_index : int,
                              *,
                              num_steps : int = 50,
                              batch_size : int = 32) ->np.ndarray:
        from .Algorithms.IntegratedGradients import IntegratedGradients
        return IntegratedGradients(self.__model,self.__preprocessor)(img_array,
                                                                     class_index,
                                                                     num_steps = num_steps,
                                                                     batch_size = batch_size)

    ########################################################################















