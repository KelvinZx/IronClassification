from keras.applications import DenseNet121
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
import glob
import os
import numpy as np
import scipy.misc as misc
from keras.utils import np_utils


tensorboard = TensorBoard('./logs/DenseNet_fine_tune2')
#early_stopping = EarlyStopping(monitor='val_loss',patience=ES)
lr_reduce = ReduceLROnPlateau(monitor='loss',patience=20,verbose=1)
checkpoint = ModelCheckpoint('./ckpt/DenseNet_stain.h5',period=1,save_best_only=True, monitor='val_acc')
callbacks = [tensorboard,lr_reduce,checkpoint]

base_model = DenseNet121(include_top=False,input_shape=(224,224,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(8, activation='softmax')(x)
model = Model(inputs=base_model.inputs, outputs=x)


train_gen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, rotation_range=10,
                               width_shift_range=0.1, height_shift_range=0.1, shear_range=10, zoom_range=0.1,
                               fill_mode='reflect', horizontal_flip=True, vertical_flip=True)
val_gen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
train_data = train_gen.flow_from_directory('./zht_train', target_size=(224,224), batch_size=16)
val_data = val_gen.flow_from_directory('./zht_val', target_size=(224,224), batch_size=16)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
model.fit_generator(train_data, steps_per_epoch=88,epochs=200,callbacks=callbacks,validation_data=val_data,validation_steps=13)

'''
path = './val'
class_list = os.listdir(path)
print(class_list)
data = []
label = []
for i, curr_cls in enumerate(class_list, 0):
    img_list = glob.glob(os.path.join(path, curr_cls, '*'))
    for j in img_list:
        img = misc.imread(j)
        img = misc.imresize(img, (224, 224))
        img = img - 128.
        img = img / 128.
        data.append(img)
        label.append(i)
data = np.array(data)
label = np.array(label)
label = np_utils.to_categorical(label, num_classes=8)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
model.fit(x=data, y=label, batch_size=16, epochs=200, validation_split=0.3, callbacks=callbacks)
'''