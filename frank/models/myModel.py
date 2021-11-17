import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, activations
from ..config import Config
from ..layers.myLayer import DistortImage, StochasticDropout, Inception, ResidualBottleneckBlock

class FrankModel(Model):
    def __init__(self, name, cfg : Config, input_shape=(32, 32, 1), classes = 10):
        super().__init__(name = name)
        self.classes = cfg.getCfgData('model', 'classes', classes)
        self.inputShape = cfg.getCfgData('dataLoader', 'input_shape', input_shape)

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
        self.layer_scale = layers.Rescaling(scale=1./255)
        self.conv1 = layers.Conv2D(kernel_size=(5, 5), filters=6, strides=1, activation='tanh')
        self.conv2 = layers.Conv2D(kernel_size=(5, 5), filters=16, strides=1, activation='tanh')

        self.maxPool1 = layers.MaxPooling2D(pool_size=2, strides=2)
        self.maxPool2 = layers.MaxPooling2D(pool_size=2, strides=2)

        self.flatten = layers.Flatten(data_format='channels_last')
        self.dense1 = layers.Dense(120, activation='tanh')
        self.dense2 = layers.Dense(84, activation='tanh')
        self.denseOutput = layers.Dense(self.classes, activation=activations.softmax)

    def call(self ,inputs, training=False):

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

        self.layer_scale = layers.Rescaling(scale=1./255)
        self.layer_resizing = layers.Resizing(*resize,
                                              interpolation='nearest')

        getConv = lambda k, f, s, p = 'valid' : layers.Conv2D(kernel_size=k, filters=f, strides=s, padding=p, activation='relu')

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

        self.outputs = layers.Dense(self.classes, activation=activations.softmax)

    def call(self, inputs, training=False):

        #preprocessing layer
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
        x = self.BN1(x) #本來應該要用 LRN


        # conv2
        #   27 * 27 * 256
        x = self.conv2(x)

        #   (27 - 3) / 2 + 1 --> 13 * 13 * 256
        x = self.moxPool2(x)
        x = self.BN2(x) #本來應該要用 LRN


        #conv3 - 4 - 5
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # (13 - 3) / 2 + 1 --> 6 * 6 * 256
        x = self.maxPool3(x)

        return x

    def __buildFC(self, x, training : bool):
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

        self.flexImgSize = config.getCfgData('model', 'flexImgSize', flexImgSize)

        self._build(config)

    def _build(self, config: Config):
        resize = config.getCfgData('model', 'resize', (224, 224))
        self.layer_scale = layers.Rescaling(scale=1./255)
        self.layer_resizing = layers.Resizing(*resize,
                                              interpolation='nearest')

        getConv = lambda k, f, s = 1, p = 'same', a = 'relu' : layers.Conv2D(kernel_size=k, filters=f, strides=s, padding=p, activation=a)

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

        self.globalAvgPool = layers.GlobalAveragePooling2D(data_format='channels_last')

        self.outputs = layers.Dense(self.classes, activation=activations.softmax)

    def call(self, inputs, training=False):
        x = self.layer_scale(inputs)
        x = self.layer_resizing(x)

        x = self.__buildConv(x)

        if self.flexImgSize: x = self.__buildLastConv(x, training)
        else: x = self.__buildFC(x, training)
        
        outputs = self.outputs(x)

        return outputs

    def __buildConv(self, x):
        #conv1
        x = self.conv1(x)
        #conv2
        x = self.conv2(x)
        #(224 - 2) / 2 + 1 --> 112 * 112 * 64
        x = self.maxPool1(x)

        #conv3
        x = self.conv3(x)
        #conv4
        x = self.conv4(x)
        #(112 - 2) / 2 + 1 --> 56 * 56 * 128
        x = self.maxPool2(x)

        #conv5
        x = self.conv5(x)
        #conv6
        x = self.conv6(x)
        #conv7
        x = self.conv7(x)
        #(56 - 2) / 2 + 1 --> 28 * 28 * 256
        x = self.maxPool3(x)

        #conv8
        x = self.conv8(x)
        #conv9
        x = self.conv9(x)
        #conv10
        x = self.conv10(x)
        #(28 - 2) / 2 + 1 --> 14 * 14 * 512
        x = self.maxPool4(x)

        #conv11
        x = self.conv11(x)
        #conv12
        x = self.conv12(x)
        #conv13
        x = self.conv13(x)
        #(14 - 2) / 2 + 1 --> 7 * 7 * 512
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

        #FC3
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

        #https://keras.io/api/layers/pooling_layers/global_average_pooling2d/
        x = self.globalAvgPool(x)

        return x

class InceptionV1(FrankModel):
    def __init__(self, config: Config, input_shape=(None, None, 3), classes=10, name="frank.InceptionV1") -> None:
        super().__init__(name, config, input_shape, classes)

        self._build(config)

    def _build(self, config: Config):
        resize = config.getCfgData('model', 'resize', (224, 224))

        self.layer_scale = layers.Rescaling(scale=1./255)
        self.layer_resizing = layers.Resizing(*resize,
                                              interpolation='nearest')

        self.conv1 = layers.Conv2D(kernel_size=7 ,strides=2 ,filters=64 ,activation='relu' ,padding='same')
        self.pool1 = layers.MaxPool2D(pool_size=3 ,strides=2, padding='same')
        self.conv2 = layers.Conv2D(kernel_size=3, strides=1, filters=192, padding='same', activation='relu')
        self.pool2 = layers.MaxPool2D(pool_size=3 ,strides=2, padding='same')

        self.inception1 = Inception(64 ,(96,128) ,(16 ,32) ,32)
        self.inception2 = Inception(128 ,(128,192) ,(32 ,96) ,64)

        self.pool3 = layers.MaxPool2D(pool_size=3 ,strides=2, padding='same')

        self.inception3 = Inception(192 ,(96,208) ,(16 ,48) ,64)
        self.inception4 = Inception(160 ,(112,224) ,(24 ,64) ,64)
        self.inception5 = Inception(128 ,(128,256) ,(24 ,64) ,64)
        self.inception6 = Inception(112 ,(144,288) ,(32 ,64) ,64)
        self.inception7 = Inception(256 ,(160,320) ,(32 ,128) ,128)

        self.pool4 = layers.MaxPool2D(pool_size=3 ,strides=2, padding='same')

        self.inception8 = Inception(256 ,(160,320) ,(32 ,128) ,128)
        self.inception9 = Inception(384 ,(192,384) ,(48 ,128) ,128)
        
        self.globalPool = layers.GlobalAveragePooling2D(data_format='channels_last')
        self.dropout = layers.Dropout(.4)

        self.outputs = layers.Dense(self.classes, activation=activations.softmax)

    def call(self, inputs, training=False):
        x = self.layer_scale(inputs)
        x = self.layer_resizing(x)
        
        x = self.conv1(x)
        #https://keras.io/api/layers/pooling_layers/max_pooling2d/
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

        self.layer_scale = layers.Rescaling(scale=1./255)
        self.layer_resizing = layers.Resizing(*resize,
                                              interpolation='nearest')

        self.conv1 = layers.Conv2D(kernel_size=(7, 7), filters=64, strides=2, padding='same')
        self.pool1 = layers.MaxPool2D(pool_size=(3,3),strides=2 ,padding='same')
        
        self._residualblockList = []
        self._residualblockList.append(ResidualBottleneckBlock(64, 64, 256, changeShortcutChannel=True))
        self._residualblockList.extend([ResidualBottleneckBlock(64, 64, 256) for i in range(2)])

        self._residualblockList.append(ResidualBottleneckBlock(128, 128, 512, needDownSample=True))
        self._residualblockList.extend([ResidualBottleneckBlock(128, 128, 512) for i in range(3)])

        self._residualblockList.append(ResidualBottleneckBlock(256, 256, 1024, needDownSample=True))
        self._residualblockList.extend([ResidualBottleneckBlock(256, 256, 1024) for i in range(5)])

        self._residualblockList.append(ResidualBottleneckBlock(512, 512, 2048, needDownSample=True))
        self._residualblockList.extend([ResidualBottleneckBlock(512, 512, 2048) for i in range(3)])

        self.globalPool = layers.GlobalAveragePooling2D()
        self.outputs = layers.Dense(self.classes, activation=activations.softmax)

    def call(self, inputs, training=False):
        x = self.layer_scale(inputs) 
        x = self.layer_resizing(x)

        x = self.conv1(x)
        x = self.pool1(x)

        for block in self._residualblockList:
            x = block(x)

        x = self.globalPool(x)

        return self.outputs(x)

class EfficientNetV2_S(Model):
    def __init__(self, config: Config, input_shape=(None, None, 3), classes=10, name="frank.EfficientNetV2_S") -> None:
        super().__init__(name = name)
        self.classes = config.getCfgData('model', 'classes', classes)
        
        self._build(config)

    def _build(self, config: Config):
        self.layer_scale = layers.Rescaling(scale=1./255)
        self.distortImage = DistortImage()

    def call(self, inputs, training=False):
        x = self.distortImage(inputs)

        x = self.layer_scale(x)

        #stem 輸出channel 和 下一層的 input channel 相同
        x = self.__conv_BN_silu_(x ,stride=2 ,filters=24) 

        #
        x = self.__Fused_MBConv(x, 24, 24, 1, 3, 1, 2)
        x = self.__Fused_MBConv(x, 24, 38, 4, 3, 2, 4)
        x = self.__Fused_MBConv(x, 48, 64, 4, 3, 2, 4)
        x = self.__MBConv(x, 64, 128, 4, 3, 2, .25, 6)
        x = self.__MBConv(x, 128, 160, 6, 3, 1, .25, 9)
        x = self.__MBConv(x, 160, 256, 6, 3, 2, .25, 15)

        # head
        x = self.__conv_BN_silu_(x ,kernel_size=1 ,stride=1 ,filters=1280) 
        x = layers.GlobalAvgPool2D()(x)
        x = layers.Dropout(self.dropout)(x)


        #output
        outputs = layers.Dense(self.classes, activation=activations.softmax)(x)
        return outputs        

    def __MBConv(self, inputs, input_filters, output_filters, expansion_ratio, kernel_size, strides, se_ratio, number_layers):
        x = inputs
        for i in range(number_layers):
            x = self.__conv_BN_silu_(x, input_filters * expansion_ratio ,kernel_size=1)
            x = self.__depthwiseconv_BN_silu_(x ,kernel_size=kernel_size,stride=strides)

            x = self.__se(x, input_filters * se_ratio, input_filters * expansion_ratio)

            x = layers.Conv2D(kernel_size=1 ,strides=1 ,filters=output_filters,padding='same')(x)
            x = layers.BatchNormalization()(x)

            x = self.__residual(inputs ,x ,strides ,input_filters ,output_filters )

            strides = 1 # 根據 code 決定的
            input_filters = output_filters # 根據 code 決定的
            inputs = x
            
        return x
        
    def __Fused_MBConv(self, inputs, input_filters, output_filters , expansion_ratio, kernel_size, strides, number_layers):
        x = inputs
        for i in range(number_layers):
            if(expansion_ratio != 1):
                x = self.__conv_BN_silu_(
                    x, input_filters * expansion_ratio, stride=strides, kernel_size=kernel_size)

            #se 似乎沒有 se block

            k_s = 1 if expansion_ratio != 1 else kernel_size
            s = 1 if expansion_ratio != 1 else strides
            x = layers.Conv2D(kernel_size=k_s ,strides=s ,filters= output_filters ,padding='same')(x)
            x = layers.BatchNormalization()(x)

            x = self.__residual(inputs, x, strides, input_filters, output_filters)

            strides = 1 # 根據 code 決定的
            input_filters = output_filters # 根據 code 決定的
            inputs = x

        return x

    def __se(self ,x ,filters ,output_filters):
        # 預設不用
        # x = layers.AveragePooling2D()(x) 應該要用 avg_pooling

        x = layers.Conv2D(kernel_size=1 ,strides= 1 ,filters=filters, padding='same')(x)
        x = tf.nn.silu(x)

        x = layers.Conv2D(kernel_size=1 ,strides=1 ,filters=output_filters, padding='same')(x)
        x = activations.sigmoid(x)

        return x

    def __conv_BN_silu_(self, x ,filters, kernel_size=(3, 3), stride=1 ,padding='same'):
        x = layers.Conv2D(kernel_size=kernel_size,
                          filters=filters, strides=stride ,padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.silu(x)


        return x
    
    def __depthwiseconv_BN_silu_(self, x , kernel_size=(3, 3), stride=1 ,padding='same'):
        x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride ,padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.silu(x)


        return x

    def __residual(self, inputs, x, strides , input_filters ,output_filters):
        if (strides == 1 and input_filters == output_filters):
            # 在 keras conv2d 中 paddind == 'same' : 公式 output_size = w / s
            # 在 effencientnet 中 input 和 output 維度都要相同才做相加(code 這樣寫)

            # x = layers.Dropout(self.dropout)(x) 不太一樣


            drop = StochasticDropout()
            x = drop(x) #需要繼承 layers.Layer
            x = layers.Add()([inputs ,x])

        return x
