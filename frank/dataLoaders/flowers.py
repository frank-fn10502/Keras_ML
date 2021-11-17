from .interface import IDataset
import tensorflow as tf
import scipy.io
from ..config import Config


class Flowers102(IDataset):
    '''
    ** 使用 102 個 class 的版本 **
    採用 flowers 資料集(https://www.robots.ox.ac.uk/~vgg/data/flowers/)
        1. Consisting of 102 different categories of flowers common to the UK
        2. Each class consists of between 40 and 258 images
    '''
    def __init__(self, 
                 cfg : Config, 
                 info = False,
                 labelPath = 'dataset/flowers/imagelabels.mat', 
                 imagePath = 'dataset/flowers/') -> None:

        super().__init__()
        if(info):
            print(f"dataset: {self.__doc__} \n")
            
        self.cfg = cfg
        self.info = info
        self.labelPath = labelPath
        self.imagePath = imagePath

        #目前下載的資料集是這樣處理
        self.__labels = scipy.io.loadmat(self.labelPath)['labels'][0].tolist()
        self.__labelMode = 'int'

        self.__batchSize = 32
        self.__imgSize = (256,256)
        self.__seed = 2021
        self.__split = .2

    def tocategorical(self) -> 'Flowers102':
        '''
        label 轉換成 one-hot 編碼(train_y 和 test_y)
        '''
        pre_labels = self.__labels[0]

        self.__labels = list(
            map(
                lambda x: [int(i) for i in x],
                tf.keras.utils.to_categorical(self.__labels)
            )
        )

        if self.info:
            print("one-hot encoder:")
            print(f"\tindex: 0 ,pre: {pre_labels} ,after:{self.__labels[0]}")

        return self

    def Done(self) -> IDataset:
        batchSize = self.cfg.getCfgData('dataLoader', 'batch_size', self.__batchSize)
        imgSize = self.cfg.getCfgData('dataLoader', 'input_shape', (*self.__imgSize ,3))[0:2]
        validationSplit = self.cfg.getCfgData('dataLoader', 'validation', self.__split)
        seed = self.cfg.getCfgData('dataLoader', 'seed', self.__seed)

        # get dataset from dir
        self.trainData = tf.keras.preprocessing.image_dataset_from_directory(
            directory=self.imagePath,
            labels=self.__labels,
            batch_size=batchSize,
            label_mode=self.__labelMode,
            image_size=imgSize,
            # Set seed to ensure the same split when loading testing data.
            seed=seed,
            validation_split=validationSplit,
            subset='training',
            shuffle=True,
            interpolation="nearest",
            crop_to_aspect_ratio=True)
            
        self.validationData = tf.keras.preprocessing.image_dataset_from_directory(
            directory=self.imagePath,
            labels=self.__labels,
            batch_size=batchSize,
            label_mode=self.__labelMode,
            image_size=imgSize,
            seed=seed,
            validation_split=validationSplit,
            subset='validation',
            shuffle=True,
            interpolation="nearest",
            crop_to_aspect_ratio=True)

        return self