import datetime
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
from frank.config import Config
from tensorflow.keras import Model


class ModelOuputHelper:
    '''
    處理神經網路模型的各種資料輸出，包含圖形、模型和文檔
    '''

    def __init__(self,
                 model: Model,
                 root_directory=None
                 ) -> None:
        if(model is None):
            raise Exception("please check model")
        self.model = model

        rootFolder = root_directory or 'result'

        self.rootFolder = Path(rootFolder) / self.model.name / \
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

        self._createFolder()

    def _createFolder(self):
        self.weightsFolder = self.rootFolder / 'weights'
        self.cfgFolder = self.rootFolder
        self.modelArchFolder = self.rootFolder / 'arch'
        self.resultFolder = self.rootFolder / 'result'
        self.tfBoardFolder = self.resultFolder / 'logs'
        self.myResultFolder = self.resultFolder / 'myResult'

        self.weightsFolder.mkdir(parents=True, exist_ok=True)
        self.cfgFolder.mkdir(parents=True, exist_ok=True)
        self.modelArchFolder.mkdir(parents=True, exist_ok=True)
        self.resultFolder.mkdir(parents=True, exist_ok=True)
        self.tfBoardFolder.mkdir(parents=True, exist_ok=True)
        self.myResultFolder.mkdir(parents=True, exist_ok=True)

    def saveTrainProcessImg(self, history=None) -> None:
        '''
        將訓練過程用 matplotlib.pyplot 畫成圖表
        :param history  傳入 model.fit() 的回傳值
        '''
        modelHistory = history
        history = history.history
        if(history is None):
            return
        plt.figure(figsize=(15, 5))

        self.__pltOnePlot('loss', (1, 2, 1),
                          [
            [history['loss'], '-'],
            [history['val_loss'], '--'],
        ], loc_mini='upper right')
        self.__pltOnePlot('accuracy', (1, 2, 2),
                          [
            [history['accuracy'], '-'],
            [history['val_accuracy'], '--'],
        ])

        plt.savefig((self.myResultFolder / 'train-progress.jpg').__str__())
        plt.show()

        print('drawTrainProcess... Done')

    def __pltOnePlot(self, title, pos, plotDatas: list, loc_mini: str = 'upper left'):
        '''
        pos:
            ex: (1,2,1)

        plotDatas:
            ex:
            [
                [[...] ,'--'],
                [[...] ,'-'],
            ]

        loc_mini: 'upper left' or 'upper right'
        '''

        plt.subplot(*pos)
        plt.title(title)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid(True, linestyle="--", color='gray', linewidth='0.5')
        xticks_start = 0
        xticks_end = 0
        yticks_start = sys.maxsize
        yticks_end = 0

        for datas, sign in plotDatas:
            xticks_end = max(xticks_end, len(datas))
            yticks_end = max(max(datas), yticks_end)
            yticks_start = min(min(datas), yticks_start)

            plt.plot(datas, sign)

        plt.legend(['train', 'test'], loc=loc_mini)
        plt.xlim([xticks_start, xticks_end])
        plt.ylim([yticks_start, yticks_end])

        x_range = max(10, (xticks_start + xticks_end) / 20)
        x_tick_list = np.arange(xticks_start, xticks_end, x_range)
        x_tick_list = np.append(x_tick_list, xticks_end)
        plt.xticks(x_tick_list, rotation=90)

        y_range = (yticks_start + yticks_end) / 10
        y_tick_list = np.arange(yticks_start, yticks_end - y_range, y_range)
        y_tick_list = np.append(y_tick_list, yticks_end)
        plt.yticks(y_tick_list)

    def saveTrainHistory(self, history):
        '''
        儲存 train 產生的 history 以備不時之需
        '''
        modelHistory = history
        history = history.history
        path = self.myResultFolder / 'trainHistory.json'
        with path.open('w') as f:
            json.dump(history, f, ensure_ascii=False, indent=4)

        print('saveTrainHistory... Done')

    # def saveModel(self):
    #     '''
    #     儲存 model 到預設位置(目前是儲存所有的資料)
    #     '''
    #     self.model.save(self.train_result_dir.__str__())

    #     print('saveModel... Done')

    def seveModelArchitecture(self) -> None:
        '''
        儲存 model:
            1. 文字 summary
            2. 圖片 keras.utils.plot_model(simple & complete)
        '''
        self.__drawModelImg()
        self.__saveModelTxT()

    def __drawModelImg(self):
        '''
        使用 keras.utils.plot_model 畫出模型架構圖
        '''
        keras.utils.plot_model(
            self.model.build_graph(),
            to_file=(self.modelArchFolder /
                     'simple-model-architecture.png').__str__(),
            show_shapes=False,
            expand_nested=True,
        )
        keras.utils.plot_model(
            self.model.build_graph(),
            to_file=(self.modelArchFolder /
                     'complete-model-architecture.png').__str__(),
            show_shapes=True,
            expand_nested=True,
        )
        print('saveModelImg... Done')

    def __saveModelTxT(self):
        '''
        儲存 model.summary()
        '''
        path = self.modelArchFolder / 'model-architecture.txt'
        with path.open('w') as f:
            self.model.build_graph().summary(print_fn=lambda x: print(x, file=f))

        print('saveModelTxT... Done')
