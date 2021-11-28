from .interface import IDataset
import tensorflow as tf
from tensorflow import keras
import numpy as np
from ..config import Config


class IkerasDataset(IDataset):
    '''
    將 keras 內建的資料集轉換成 python generate 以統一 model.fit 的參數
    - train_x, train_y, test_x, test_y 轉換成
        - self.trainData
        - self.validationData
    '''

    def __init__(self, cfg: Config, info: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.info = info
        if(info):
            print(f"dataset: {self.__doc__} \n")

    def Done(self) -> 'IkerasDataset':
        '''
        將資料統一成 trainData, validationData
        - 會自動回傳所需的 batchsize (cfg 檔)
        '''
        if self.info:
            print(
                f"train_x:{self._train_x.shape} \ntrain_y:{self._train_y.shape} \ntest_x:{self._test_x.shape} \ntest_y:{self._test_y.shape}")

        def getDatas(datas, labels, batchSize) -> tf.data.Dataset:
            d = tf.data.Dataset.from_tensor_slices((datas, labels))
            return d.shuffle(batchSize * 4).batch(batchSize)

        batchSize = self.cfg.getCfgData('dataLoader', 'batch_size')
        self.trainData = getDatas(self._train_x, self._train_y, batchSize)
        self.validationData = getDatas(self._test_x, self._test_y, batchSize)
        self.batchSize = batchSize

        self.inputShape = self._train_x[0].shape
        self.classes = self._train_y[0].size

        return self

    def addChannel(self) -> 'IkerasDataset':
        return self

    def tocategorical(self) -> 'IkerasDataset':
        '''
        label 轉換成 one-hot 編碼(train_y 和 test_y)
        '''
        pre_train_y = self._train_y[0]
        pre_test_y = self._test_y[0]

        self._train_y = tf.keras.utils.to_categorical(
            self._train_y, dtype="uint8")
        self._test_y = tf.keras.utils.to_categorical(
            self._test_y, dtype="uint8")

        if self.info:
            print("one-hot encoder:")
            print(f"\tindex: 0 ,pre: {pre_train_y} ,after:{self._train_y[0]}")
            print(
                f"\tindex: 0 ,pre: {pre_test_y} ,after:{self._test_y[0]}\n{'-'*10}")

        return self


class MNIST(IkerasDataset):
    '''
    使用 tensorflow.keras 取得的 MNIST 資料集(https://keras.io/api/datasets/mnist/) \n
    This is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images
    '''

    def __init__(self, cfg: Config, info: bool = False) -> None:
        self.dataset = keras.datasets.mnist
        (self._train_x, self._train_y), \
            (self._test_x, self._test_y) = self.dataset.load_data()
        super().__init__(cfg, info)

    def addChannel(self) -> 'MNIST':
        self._train_x = np.expand_dims(self._train_x, 3)
        self._test_x = np.expand_dims(self._test_x, 3)
        return self


class CIFAR10(IkerasDataset):
    '''
    使用 tensorflow.keras 取得的 CIFAR10 資料集(https://keras.io/api/datasets/cifar10/) \n
    The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. \n
    There are 50000 training images and 10000 test images.
    '''

    def __init__(self, cfg: Config, info: bool = False) -> None:
        self.dataset = keras.datasets.cifar10
        (self._train_x, self._train_y), \
            (self._test_x, self._test_y) = self.dataset.load_data()
        super().__init__(cfg, info)


class CIFAR100(IkerasDataset):
    '''
    使用 tensorflow.keras 取得的 CIFAR10 資料集(https://keras.io/api/datasets/cifar100/) \n
    This dataset is just like the CIFAR-10, except it has 100 classes containing 600 images each. \n
    There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 super()classes. \n
    Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).\n
    EX: Superclass:(aquatic mammals) \n
        Classes: (beaver, dolphin, otter, seal, whale)
    '''

    def __init__(self, cfg: Config, label_mode: str = "fine", info: bool = False) -> None:
        self.dataset = keras.datasets.cifar100
        (self._train_x, self._train_y), \
            (self._test_x, self._test_y) = self.dataset.load_data(
                label_mode=label_mode)
        super().__init__(cfg, info)
