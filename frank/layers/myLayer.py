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
