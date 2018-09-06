from keras.applications import DenseNet121
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
import glob
import os
import numpy as np
import scipy.misc as misc
from keras.utils import np_utils
from mainpath import ROOT_DIR
from config import Config
from resnet import ResnetBuilder

keras_tb = os.path.join(ROOT_DIR, 'keras', 'tblogs')
ckpt_keras = os.path.join(ROOT_DIR, 'keras', 'ckpt')
hyper = Config.backbone
tensorboard = TensorBoard(os.path.join(keras_tb, hyper))
early_stopping = EarlyStopping(monitor='val_loss',patience=5, min_delta=0.001)
lr_reduce = ReduceLROnPlateau(monitor='loss',patience=20,verbose=1)
checkpoint = ModelCheckpoint(os.path.join('ckpt', hyper + '.h5'), period=1,save_best_only=True, monitor='val_acc')
callbacks = [tensorboard,lr_reduce,checkpoint]
batch_size = 16

base_model = ResnetBuilder.build_resnet_50(input_shape=(224,224,3), num_outputs=8, block_fun='resnet')

optimizer = SGD(lr=0.01, momentum=0.9, decay=0.00001)
train_gen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, rotation_range=10,
                               width_shift_range=0.1, height_shift_range=0.1, shear_range=10, zoom_range=0.1,
                               fill_mode='reflect', horizontal_flip=True, vertical_flip=True)
val_gen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
train_data = train_gen.flow_from_directory(os.path.join(ROOT_DIR, 'train'), target_size=(224,224), batch_size=batch_size)
val_data = val_gen.flow_from_directory(os.path.join(ROOT_DIR, 'val'), target_size=(224,224), batch_size=batch_size)

train_step_per_epoch = len(train_data)/batch_size
valid_step_per_epoch = len(val_data)/batch_size

base_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
base_model.fit_generator(train_data, steps_per_epoch=train_step_per_epoch,
                         epochs=200,callbacks=callbacks,
                         validation_data=val_data,validation_steps=valid_step_per_epoch)