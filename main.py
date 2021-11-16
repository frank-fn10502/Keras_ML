import tensorflow as tf


from frank.models.myModel import LeNet, AlexNet, EfficientNetV2_S
from frank.config import Config
from pathlib import Path
from frank.dataLoaders.kerasDataset import MNIST
from frank.dataLoaders.flowers import Flowers102
from utils.outputs import ModelOuputHelper

# tf.config.experimental_run_functions_eagerly(True)

# 從 cfg 得到 training 設定值
# epoch, init_learning_rate, optimizer, lr_schedule, loss, 甚至 metrics
#--------------------
cfg = Config()

epoch = cfg.getCfgData('train', 'epoch', 10)
optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, 
                                  momentum=0.9)
loss='categorical_crossentropy',
metrics=['accuracy']
#--------------------

#dataset
#--------------------
dataLoader = MNIST(cfg, info=True).tocategorical().addChannel().Done()
#--------------------

#model prepare
#--------------------
model = LeNet(cfg, dataLoader.inputShape, dataLoader.classes)

# inputs = tf.keras.Input(shape=dataLoader.inputShape)
# outputs = myModel(inputs)
# model = tf.keras.Model(inputs=inputs, outputs=outputs, name=myModel.name)
# model.build(input_shape=dataLoader.inputShape)

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics
)
#--------------------

outputHelper = ModelOuputHelper(model=model, root_directory='temp')

cfg.saveConfig(savePath=Path(outputHelper.cfgFolder))
#dataset 應該也要記錄 (outputHelper
outputHelper.seveModelArchitecture()

#training
model.fit(
    x = dataLoader.trainData,
    epochs=epoch,
    validation_data=dataLoader.validationData,
    # steps_per_epoch = 10,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(outputHelper.weightsFolder / '{epoch:03d}-{val_accuracy:.3f}.h5',
                                           save_weights_only=True,
                                           monitor='val_accuracy',
                                           mode='max',
                                           save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=outputHelper.tfBoardFolder, histogram_freq=1)
    ]
)

outputHelper.saveTrainProcessImg()
outputHelper.saveTrainHistory()