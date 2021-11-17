import tensorflow as tf
from tensorflow.keras import layers
from .other.autoaugment import distort_image_with_randaugment



# tensorflow.python.framework.ops.EagerTensor' object has no attribute '_keras_history'
#為了解決以上的錯誤(主要是畫圖時會需要用到該屬性)
class StochasticDropout(layers.Layer):
    # from effNetV2 code #survival_prob 預設 0.8 def drop_connect(inputs, is_training = True, survival_prob = 0.8):
    def call(self, inputs ,training=None):
        """Drop the entire conv with given survival probability."""
        # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
        if not training:
            return inputs

        survival_prob = 0.8
        # Compute tensor.
        batch_size = tf.shape(inputs)[0]
        random_tensor = survival_prob
        random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
        binary_tensor = tf.floor(random_tensor)
        # Unlike conventional way that multiply survival_prob at test time, here we
        # divide survival_prob at training time, such that no addition compute is
        # needed at test time.
        output = inputs / survival_prob * binary_tensor
        return output

# 使用 google effNetv2 的 code 包裝成 keras.layers
class DistortImage(layers.Layer):
    '''
    random augment
    '''
    def __init__(self, totalEpoch = 0):
        super().__init__()
        self.numLayers = 2

        self.magnitude = 5
        self.__minMagnitude = 5
        self.__maxMagnitude = 15
        self.imgSize = (128 ,128)
        self.__minImgSize = (128 ,128)
        self.__maxImgSize = (300 ,300)

        self.totalEpoch = totalEpoch
        self.setStages(4)

        self.layer_resizing = layers.Resizing(*self.imgSize, interpolation='nearest')

    def setStages(self ,n):
        self.__stages = n
        self.__numPerStage = self.totalEpoch // self.__stages
        self.__magnitudePerStage = (self.__maxMagnitude - self.__minMagnitude) / self.__stages

        self.__imgSizePerStage = (self.__maxImgSize[0] - self.__minImgSize[0]) / self.__stages

    def setNewMagnitude(self ,currentEpoch):
        # (得到現在的 stage) * 每一個 stage 的增長數量 + 基礎的值
        self.magnitude = (currentEpoch // self.__numPerStage) * self.__magnitudePerStage + self.__minMagnitude

    def setResizing(self ,currentEpoch):
        self.imgSize = (int((currentEpoch // self.__numPerStage) * self.__imgSizePerStage + self.__minImgSize[0]),
                        int((currentEpoch // self.__numPerStage) * self.__imgSizePerStage + self.__minImgSize[1]))

        self.layer_resizing = layers.Resizing(*self.imgSize, interpolation='nearest')
        
    def testFunc(self, inputs):
        inputs = self.layer_resizing(inputs) # 先 resizing 到目標大小
        func = distort_image_with_randaugment
        return tf.map_fn(lambda img: func(img, self.numLayers, self.magnitude), inputs, dtype=tf.float32)

    def call(self, inputs ,training=None):
        if training:
            inputs = self.layer_resizing(inputs) # 先 resizing 到目標大小
            func = distort_image_with_randaugment
            
            return tf.map_fn(lambda img: func(img, self.numLayers, self.magnitude), inputs, dtype=tf.float32)
        else:
            return inputs

class Inception(layers.Layer):
    def __init__(self, *depth):
        super().__init__()
        self._build(depth)

    def _build(self, depth):
        conv1x1 ,(c3x3_r ,c3x3) ,(c5x5_r,c5x5) ,c1x1_pool = depth

        getConv = lambda k, f, : layers.Conv2D(kernel_size=k, filters=f, padding='same', activation='relu')

        self.b1x1 = getConv(1, conv1x1)
        
        self.b3x3_r = getConv(1, c3x3_r)
        self.b3x3 = getConv(3, c3x3)

        self.b5x5_r = getConv(1, c5x5_r)
        self.b5x5 = getConv(5, c5x5)

        self.pool = layers.MaxPool2D(pool_size=3 ,strides=1, padding='same')

        self.b1x1_pool = getConv(1, c1x1_pool)

    def call(self, inputs):
        b1x1 = self.b1x1(inputs)
        b3x3 = self.b3x3(self.b3x3_r(inputs))
        b5x5 = self.b5x5(self.b5x5_r(inputs))
        pool = self.b1x1_pool(self.pool(inputs))

        return layers.concatenate([b1x1, b3x3, b5x5 ,pool],axis=3)

class ResNet50_BN_Conv(layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), stride=1):
        super().__init__()
        self.BN = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv = layers.Conv2D(kernel_size=kernel_size, filters=filters, strides=stride ,padding='same')

    def call(self, inputs):
        return self.conv(self.relu(self.BN(inputs)))

class ResidualBottleneckBlock(layers.Layer):
    def __init__(self, *f, needDownSample=False, changeShortcutChannel=False):
        super().__init__()
        self.needDownSample = needDownSample
        self.changeShortcutChannel = changeShortcutChannel
        self._build(f)

    def _build(self, f):
        (cf1 ,cf2 ,cf3) = f

        self.conv1 = ResNet50_BN_Conv(kernel_size=(1, 1), filters=cf1, stride=2 if self.needDownSample else 1)
        self.conv2 = ResNet50_BN_Conv(filters=cf2)
        self.conv3 = ResNet50_BN_Conv(kernel_size=(1, 1), filters=cf3)

        self.reshapeConv = ResNet50_BN_Conv( kernel_size=(1,1), filters=cf3, stride=2 if self.needDownSample else 1)

    def call(self, inputs):
        x = self.conv3(self.conv2(self.conv1(inputs)))

        if self.needDownSample or self.changeShortcutChannel:
        #filters 要使用最終的輸出才可相加
            inputs = self.reshapeConv(inputs)

        return layers.Add()([inputs ,x])