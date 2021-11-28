import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, activations
from ..config import Config
from ..layers.myLayer import DistortImage, InceptionBlock, ResidualBottleneckBlock
from ..layers import efficientv2
# from ..layers.efficientv2 import StochasticDropout


class FrankModel(Model):
    def __init__(self, name, cfg: Config, input_shape=(32, 32, 1), classes=10):
        super().__init__(name=name)
        self.classes = cfg.getCfgData('model', 'classes', classes)
        self.inputShape = cfg.getCfgData(
            'dataLoader', 'input_shape', input_shape)

    def build_graph(self):
        '''
        為了印出完整的架構
        '''
        inputs = keras.Input(shape=(self.inputShape))
        return Model(inputs=inputs, outputs=self.call(inputs))


class LeNet(FrankModel):
    def __init__(self, config: Config, input_shape=(32, 32, 1), classes=10, name="frank.LeNet") -> None:
        super().__init__(name, config, input_shape, classes)

        self._build()

    def _build(self):
        self.layer_scale = layers.Rescaling(scale=1. / 255)
        self.conv1 = layers.Conv2D(kernel_size=(
            5, 5), filters=6, strides=1, activation='tanh')
        self.conv2 = layers.Conv2D(kernel_size=(
            5, 5), filters=16, strides=1, activation='tanh')

        self.maxPool1 = layers.MaxPooling2D(pool_size=2, strides=2)
        self.maxPool2 = layers.MaxPooling2D(pool_size=2, strides=2)

        self.flatten = layers.Flatten(data_format='channels_last')
        self.dense1 = layers.Dense(120, activation='tanh')
        self.dense2 = layers.Dense(84, activation='tanh')
        self.denseOutput = layers.Dense(
            self.classes, activation=activations.softmax)

    def call(self, inputs, training=False):

        # 數值縮成 0 ~ 1 之間
        x = self.layer_scale(inputs)

        # ( [(32 - 5) / 1] + 1 ) * 6 = 28 * 6 ==> 輸出 28 * 28 * 6
        x = self.conv1(x)
        # ( [(28 - 2) / 2] + 1 ) * 6 = 14 * 6 ==> 輸出 14 * 14 * 6
        x = self.maxPool1(x)

        # ( [(14 - 5) / 1] + 1 ) * 16 = 10 * 16 ==> 輸出 10 * 10 * 16
        x = self.conv2(x)
        # ( [(10 - 2) / 2] + 1 ) * 16 = 5 * 16 ==> 輸出 5 * 5 * 16
        x = self.maxPool2(x)

        # 2D --> 1D
        x = self.flatten(x)

        x = self.dense1(x)
        x = self.dense2(x)

        return self.denseOutput(x)


class AlexNet(FrankModel):
    def __init__(self, config: Config, input_shape=(None, None, 3), classes=10, name="frank.AlexNet") -> None:
        super().__init__(name, config, input_shape, classes)

        self._build(config)

    def _build(self, config: Config):
        resize = config.getCfgData('model', 'resize', (224, 224))

        self.layer_scale = layers.Rescaling(scale=1. / 255)
        self.layer_resizing = layers.Resizing(*resize,
                                              interpolation='nearest')

        def getConv(k, f, s, p='valid'):
            return layers.Conv2D(
                kernel_size=k, filters=f, strides=s, padding=p, activation='relu')

        self.conv1 = getConv(11, 96, 4)
        self.moxPool1 = layers.MaxPool2D(pool_size=3, strides=2)
        self.BN1 = layers.BatchNormalization()

        self.conv2 = getConv(5, 256, 2, 'same')
        self.moxPool2 = layers.MaxPool2D(pool_size=3, strides=2)
        self.BN2 = layers.BatchNormalization()

        self.conv3 = getConv(3, 384, 1, 'same')
        self.conv4 = getConv(3, 384, 1, 'same')
        self.conv5 = getConv(3, 256, 1, 'same')

        self.maxPool3 = layers.MaxPool2D(pool_size=3, strides=2)

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(4096, activation='relu')
        self.dropout1 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(4096, activation='relu')
        self.dropout2 = layers.Dropout(0.5)

        self.outputs = layers.Dense(
            self.classes, activation=activations.softmax)

    def call(self, inputs, training=False):

        # preprocessing layer
        x = self.layer_scale(inputs)
        x = self.layer_resizing(x)

        x = self.__buildConv(x)
        outputs = self.__buildFC(x, training)

        return outputs

    def __buildConv(self, x):
        # conv1
        #   (227 - 11) / 4 + 1 --> 55 * 55 * 96
        x = self.conv1(x)

        #   (55 - 3) / 2 + 1  --> 27 * 27 * 96
        x = self.moxPool1(x)
        x = self.BN1(x)  # 本來應該要用 LRN

        # conv2
        #   27 * 27 * 256
        x = self.conv2(x)

        # (27 - 3) / 2 + 1 --> 13 * 13 * 256
        x = self.moxPool2(x)
        x = self.BN2(x)  # 本來應該要用 LRN

        # conv3 - 4 - 5
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # (13 - 3) / 2 + 1 --> 6 * 6 * 256
        x = self.maxPool3(x)

        return x

    def __buildFC(self, x, training: bool):
        x = self.flatten(x)
        # FC6
        x = self.dense1(x)
        x = self.dropout1(x, training=training)

        # FC7
        x = self.dense2(x)
        x = self.dropout2(x, training=training)

        # FC8
        # x = layers.Dense(1000)(x)
        # x = layers.Dropout(0.5)(x)
        x = self.outputs(x)

        return x


class VGG16(FrankModel):
    def __init__(self, config: Config, input_shape=(None, None, 3), classes=10, flexImgSize=False, name="frank.VGG16") -> None:
        super().__init__(name, config, input_shape, classes)

        self.flexImgSize = config.getCfgData(
            'model', 'flexImgSize', flexImgSize)

        self._build(config)

    def _build(self, config: Config):
        resize = config.getCfgData('model', 'resize', (224, 224))
        self.layer_scale = layers.Rescaling(scale=1. / 255)
        self.layer_resizing = layers.Resizing(*resize,
                                              interpolation='nearest')

        def getConv(k, f, s=1, p='same', a='relu'):
            return layers.Conv2D(
                kernel_size=k, filters=f, strides=s, padding=p, activation=a)

        self.conv1 = getConv(3, 64)
        self.conv2 = getConv(3, 64)
        self.maxPool1 = layers.MaxPool2D(pool_size=2, strides=2)

        self.conv3 = getConv(3, 128)
        self.conv4 = getConv(3, 128)
        self.maxPool2 = layers.MaxPool2D(pool_size=2, strides=2)

        self.conv5 = getConv(3, 256)
        self.conv6 = getConv(3, 256)
        self.conv7 = getConv(3, 256)
        self.maxPool3 = layers.MaxPool2D(pool_size=2, strides=2)

        self.conv8 = getConv(3, 512)
        self.conv9 = getConv(3, 512)
        self.conv10 = getConv(3, 512)
        self.maxPool4 = layers.MaxPool2D(pool_size=2, strides=2)

        self.conv11 = getConv(3, 512)
        self.conv12 = getConv(3, 512)
        self.conv13 = getConv(3, 512)
        self.maxPool5 = layers.MaxPool2D(pool_size=2, strides=2)

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(4096, activation='relu')
        self.dropout1 = layers.Dropout(0.5)

        self.dense2 = layers.Dense(4096, activation='relu')
        self.dropout2 = layers.Dropout(0.5)

        self.dense3 = layers.Dense(1000)
        self.dropout3 = layers.Dropout(0.5)

        self.last_conv1 = getConv(7, 4096, p='valid', a=None)
        self.last_dropout1 = layers.Dropout(0.5)

        self.last_conv2 = getConv(1, 4096, p='valid', a=None)
        self.last_dropout2 = layers.Dropout(0.5)

        self.last_conv3 = getConv(1, 1000, p='valid', a=None)
        self.last_dropout3 = layers.Dropout(0.5)

        self.globalAvgPool = layers.GlobalAveragePooling2D(
            data_format='channels_last')

        self.outputs = layers.Dense(
            self.classes, activation=activations.softmax)

    def call(self, inputs, training=False):
        x = self.layer_scale(inputs)
        x = self.layer_resizing(x)

        x = self.__buildConv(x)

        if self.flexImgSize:
            x = self.__buildLastConv(x, training)
        else:
            x = self.__buildFC(x, training)

        outputs = self.outputs(x)

        return outputs

    def __buildConv(self, x):
        # conv1
        x = self.conv1(x)
        # conv2
        x = self.conv2(x)
        # (224 - 2) / 2 + 1 --> 112 * 112 * 64
        x = self.maxPool1(x)

        # conv3
        x = self.conv3(x)
        # conv4
        x = self.conv4(x)
        # (112 - 2) / 2 + 1 --> 56 * 56 * 128
        x = self.maxPool2(x)

        # conv5
        x = self.conv5(x)
        # conv6
        x = self.conv6(x)
        # conv7
        x = self.conv7(x)
        # (56 - 2) / 2 + 1 --> 28 * 28 * 256
        x = self.maxPool3(x)

        # conv8
        x = self.conv8(x)
        # conv9
        x = self.conv9(x)
        # conv10
        x = self.conv10(x)
        # (28 - 2) / 2 + 1 --> 14 * 14 * 512
        x = self.maxPool4(x)

        # conv11
        x = self.conv11(x)
        # conv12
        x = self.conv12(x)
        # conv13
        x = self.conv13(x)
        # (14 - 2) / 2 + 1 --> 7 * 7 * 512
        x = self.maxPool5(x)

        return x

    def __buildFC(self, x, training):
        # FC1
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x, training)

        # FC2
        x = self.dense2(x)
        x = self.dropout2(x, training)

        # FC3
        x = self.dense3(x)
        x = self.dropout3(x, training)

        return x

    def __buildLastConv(self, x, training):
        # conv1
        x = self.last_conv1(x)
        x = self.last_dropout1(x, training)

        # conv2
        x = self.last_conv2(x)
        x = self.last_dropout2(x, training)

        # conv3
        x = self.last_conv3(x)
        x = self.last_dropout3(x, training)

        # https://keras.io/api/layers/pooling_layers/global_average_pooling2d/
        x = self.globalAvgPool(x)

        return x


class InceptionV1(FrankModel):
    def __init__(self, config: Config, input_shape=(None, None, 3), classes=10, name="frank.InceptionV1") -> None:
        super().__init__(name, config, input_shape, classes)

        self._build(config)

    def _build(self, config: Config):
        resize = config.getCfgData('model', 'resize', (224, 224))

        self.layer_scale = layers.Rescaling(scale=1. / 255)
        self.layer_resizing = layers.Resizing(*resize,
                                              interpolation='nearest')

        self.conv1 = layers.Conv2D(
            kernel_size=7, strides=2, filters=64, activation='relu', padding='same')
        self.pool1 = layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        self.conv2 = layers.Conv2D(
            kernel_size=3, strides=1, filters=192, padding='same', activation='relu')
        self.pool2 = layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        self.inception1 = InceptionBlock(64, (96, 128), (16, 32), 32)
        self.inception2 = InceptionBlock(128, (128, 192), (32, 96), 64)

        self.pool3 = layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        self.inception3 = InceptionBlock(192, (96, 208), (16, 48), 64)
        self.inception4 = InceptionBlock(160, (112, 224), (24, 64), 64)
        self.inception5 = InceptionBlock(128, (128, 256), (24, 64), 64)
        self.inception6 = InceptionBlock(112, (144, 288), (32, 64), 64)
        self.inception7 = InceptionBlock(256, (160, 320), (32, 128), 128)

        self.pool4 = layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        self.inception8 = InceptionBlock(256, (160, 320), (32, 128), 128)
        self.inception9 = InceptionBlock(384, (192, 384), (48, 128), 128)

        self.globalPool = layers.GlobalAveragePooling2D(
            data_format='channels_last')
        self.dropout = layers.Dropout(.4)

        self.outputs = layers.Dense(
            self.classes, activation=activations.softmax)

    def call(self, inputs, training=False):
        x = self.layer_scale(inputs)
        x = self.layer_resizing(x)

        x = self.conv1(x)
        # https://keras.io/api/layers/pooling_layers/max_pooling2d/
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.inception1(x)
        x = self.inception2(x)

        x = self.pool3(x)

        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        x = self.inception7(x)

        x = self.pool4(x)

        x = self.inception8(x)
        x = self.inception9(x)

        x = self.globalPool(x)
        x = self.dropout(x, training)

        outputs = self.outputs(x)

        return outputs


class ResNet50(FrankModel):
    def __init__(self, config: Config, input_shape=(None, None, 3), classes=10, name="frank.ResNet50") -> None:
        super().__init__(name, config, input_shape, classes)

        self._build(config)

    def _build(self, config: Config):
        resize = config.getCfgData('model', 'resize', (224, 224))

        self.layer_scale = layers.Rescaling(scale=1. / 255)
        self.layer_resizing = layers.Resizing(*resize,
                                              interpolation='nearest')

        self.conv1 = layers.Conv2D(kernel_size=(
            7, 7), filters=64, strides=2, padding='same')
        self.pool1 = layers.MaxPool2D(
            pool_size=(3, 3), strides=2, padding='same')

        self._residualblockList = []
        self._residualblockList.append(ResidualBottleneckBlock(
            64, 64, 256, changeShortcutChannel=True))
        self._residualblockList.extend(
            [ResidualBottleneckBlock(64, 64, 256) for i in range(2)])

        self._residualblockList.append(
            ResidualBottleneckBlock(128, 128, 512, needDownSample=True))
        self._residualblockList.extend(
            [ResidualBottleneckBlock(128, 128, 512) for i in range(3)])

        self._residualblockList.append(
            ResidualBottleneckBlock(256, 256, 1024, needDownSample=True))
        self._residualblockList.extend(
            [ResidualBottleneckBlock(256, 256, 1024) for i in range(5)])

        self._residualblockList.append(
            ResidualBottleneckBlock(512, 512, 2048, needDownSample=True))
        self._residualblockList.extend(
            [ResidualBottleneckBlock(512, 512, 2048) for i in range(3)])

        self.globalPool = layers.GlobalAveragePooling2D()
        self.outputs = layers.Dense(
            self.classes, activation=activations.softmax)

    def call(self, inputs, training=False):
        x = self.layer_scale(inputs)
        x = self.layer_resizing(x)

        x = self.conv1(x)
        x = self.pool1(x)

        for block in self._residualblockList:
            x = block(x)

        x = self.globalPool(x)

        return self.outputs(x)


class EfficientNetV2_S(FrankModel):
    def __init__(self, config: Config, input_shape=(None, None, 3), classes=10, name="frank.EfficientNetV2_S") -> None:
        super().__init__(name, config, input_shape, classes)
        self.dropoutRate = .2

        self._build(config)

    def _build(self, config: Config):
        resize = config.getCfgData('model', 'resize', (128, 128))

        self.layer_scale = layers.Rescaling(scale=1. / 255)
        self.layer_resizing = layers.Resizing(*resize,
                                              interpolation='nearest')
        self.distortImage = DistortImage()

        self.conv1 = efficientv2.BN_ConvBlock(stride=2, filters=24)

        def getStride(i, s):
            return s if i == 0 else 1

        def getInOutFilter(j, i, o):
            return (i, o) if j == 0 else (o, o)

        self.FusedBlockList = []
        self.FusedBlockList.extend([efficientv2.Fused_MBConvBlock(
            *getInOutFilter(i, 24, 24), 1, 3, getStride(i, 1)) for i in range(2)])
        self.FusedBlockList.extend([efficientv2.Fused_MBConvBlock(
            *getInOutFilter(i, 24, 38), 4, 3, getStride(i, 2)) for i in range(4)])
        self.FusedBlockList.extend([efficientv2.Fused_MBConvBlock(
            *getInOutFilter(i, 48, 64), 4, 3, getStride(i, 2)) for i in range(4)])

        self.MB_BlockList = []
        self.MB_BlockList.extend([efficientv2.MBConvBlock(
            *getInOutFilter(i, 64, 128), 4, 3, getStride(i, 2), .25) for i in range(6)])
        self.MB_BlockList.extend([efficientv2.MBConvBlock(
            *getInOutFilter(i, 128, 160), 6, 3, getStride(i, 1), .25) for i in range(9)])
        self.MB_BlockList.extend([efficientv2.MBConvBlock(
            *getInOutFilter(i, 160, 256), 6, 3, getStride(i, 2), .25) for i in range(15)])

        self.conv2 = efficientv2.BN_ConvBlock(
            kernel_size=1, stride=1, filters=1280)
        self.globalAvgPool = layers.GlobalAvgPool2D()
        self.dropout = layers.Dropout(self.dropoutRate)
        self.outputs = layers.Dense(
            self.classes, activation=activations.softmax)

    def call(self, inputs, training=False):
        x = inputs
        # x = self.distortImage(inputs)
        x = self.layer_resizing(x)

        x = self.layer_scale(x)

        # stem 輸出 channel 和 下一層的 input channel 相同
        x = self.conv1(x)

        # body
        for block in self.FusedBlockList:
            x = block(x)

        for block in self.MB_BlockList:
            x = block(x)

        # head
        x = self.conv2(x)
        x = self.globalAvgPool(x)
        x = self.dropout(x, training)

        # output
        return self.outputs(x)
