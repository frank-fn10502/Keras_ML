import tensorflow as tf
from tensorflow.keras import layers


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

class BN_ConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), stride=1 ,padding='same'):
        super().__init__()
        self._conv = layers.Conv2D(kernel_size=kernel_size, filters=filters, strides=stride ,padding=padding)
        self._BN = layers.BatchNormalization()
        self._silu = tf.nn.silu

    def call(self, inputs):
        return self._silu(self._BN(self._conv(inputs)))

class BN_DepthwiseConv(layers.Layer):
    def __init__(self, kernel_size=(3, 3), stride=1 ,padding='same'):
        super().__init__()
        self._conv = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride ,padding=padding)
        self._BN = layers.BatchNormalization()
        self._silu = tf.nn.silu

    def call(self, inputs):
        return self._silu(self._BN(self._conv(inputs)))

class SE(layers.Layer):
    def __init__(self, filters ,output_filters):
        super().__init__()
        self._conv1 = layers.Conv2D(kernel_size=1 ,strides= 1 ,filters=filters, padding='same')
        self._silu = tf.nn.silu

        self._conv2 = layers.Conv2D(kernel_size=1 ,strides=1 ,filters=output_filters, padding='same')
        self._act = tf.keras.activations.sigmoid
        
    def call(self, inputs):
        return self._act(self._conv2(self._silu(self._conv1(inputs))))

class MBConvBlock(layers.Layer):
    def __init__(self, input_filters, output_filters, expansion_ratio, kernel_size, strides, se_ratio):
        super().__init__()
        self.conv1 = BN_ConvBlock(input_filters * expansion_ratio ,kernel_size=1)
        self.conv2 = BN_DepthwiseConv(kernel_size=kernel_size,stride=strides)
        self.se = SE(input_filters * se_ratio, input_filters * expansion_ratio)

        self.conv3 = layers.Conv2D(kernel_size=1 ,strides=1 ,filters=output_filters,padding='same')

        self.BN = layers.BatchNormalization()

        self._drop = StochasticDropout()

        self.strides = strides
        self.input_filters = input_filters
        self.output_filters = output_filters

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.se(x)
        x = self.conv3(x)
        x = self.BN(x)

        x = self._residual(x ,inputs)

        return x

    def _residual(self, x, inputs):
        if (self.strides == 1 and self.input_filters == self.output_filters):
            x = self._drop(x)
            x = layers.Add()([x, inputs])            

        return x

class Fused_MBConvBlock(layers.Layer):
    def __init__(self, input_filters, output_filters , expansion_ratio, kernel_size, strides):
        super().__init__()
        self.conv1 = BN_ConvBlock(input_filters * expansion_ratio, stride=strides, kernel_size=kernel_size)

        k_s = 1 if expansion_ratio != 1 else kernel_size
        s = 1 if expansion_ratio != 1 else strides

        self.conv2 = layers.Conv2D(kernel_size=k_s ,strides=s ,filters= output_filters ,padding='same')
        self.BN = layers.BatchNormalization()

        self._drop = StochasticDropout()

        self.expansion_ratio = expansion_ratio

        self.strides = strides
        self.input_filters = input_filters
        self.output_filters = output_filters

    def call(self ,inputs):
        x = inputs
        if(self.expansion_ratio != 1):
            x = self.conv1(x)

        #se

        x = self.conv2(x)
        x = self.BN(x)

        x = self._residual(x ,inputs)

        return x

    def _residual(self, x, inputs):
        if (self.strides == 1 and self.input_filters == self.output_filters):
            x = self._drop(x)
            x = layers.Add()([inputs ,x])            

        return x