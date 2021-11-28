import tensorflow as tf
import os


from frank.models.myModel import LeNet, AlexNet, VGG16, InceptionV1, ResNet50, EfficientNetV2_S
from frank.config import Config
from pathlib import Path
from frank.dataLoaders.kerasDataset import MNIST, CIFAR10
from frank.dataLoaders.flowers import Flowers102
from utils.outputs import ModelOuputHelper

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# tf.config.experimental_run_functions_eagerly(True)

# 從 cfg 得到 training 設定值
# epoch, init_learning_rate, optimizer, lr_schedule, loss, 甚至 metrics
# --------------------
cfg = Config()

epoch = cfg.getCfgData('train', 'epoch', 2)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01,
                                    momentum=0.9)
loss = 'categorical_crossentropy',
metrics = ['accuracy']
# --------------------

# dataset
# --------------------
dataLoader = MNIST(cfg, info=True).tocategorical().addChannel().Done()
# --------------------

# model prepare
# --------------------
model = EfficientNetV2_S(cfg, dataLoader.inputShape, dataLoader.classes)

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics
)
# --------------------

outputHelper = ModelOuputHelper(model=model, root_directory='temp')

cfg.setCommient('dataLoader', dataLoader.__doc__)  # 紀錄 dataset
cfg.saveConfig(savePath=Path(outputHelper.cfgFolder))
outputHelper.seveModelArchitecture()

# training
history = \
    model.fit(
        x=dataLoader.trainData,
        epochs=epoch,
        validation_data=dataLoader.validationData,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(outputHelper.weightsFolder / '{epoch:02d}-{val_accuracy:.3f}.h5',
                                               save_weights_only=True,
                                               monitor='val_accuracy',
                                               mode='max',
                                               save_best_only=True),
            tf.keras.callbacks.TensorBoard(
                log_dir=outputHelper.tfBoardFolder, histogram_freq=1)
        ]
    )

outputHelper.saveTrainProcessImg(history)
outputHelper.saveTrainHistory(history)
