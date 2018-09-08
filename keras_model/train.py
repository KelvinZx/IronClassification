from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
#from mainpath import ROOT_DIR
from keras_config import Config
from resnet import ResnetBuilder
from keras_loss import cls_cross_entropy

import os

ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith('keras_model'):
    ROOT_DIR = os.path.dirname(ROOT_DIR)
keras_tb = os.path.join(ROOT_DIR, 'keras_model', 'tblogs')
ckpt_keras = os.path.join(ROOT_DIR, 'keras_model', 'ckpt')
weights_keras = os.path.join(ROOT_DIR, 'keras_model', 'model_weights')
json_keras = os.path.join(ROOT_DIR, 'keras_model', 'json')
OPEN_DIR = os.path.join(ROOT_DIR, 'breasKHis_patient')
TRAIN_OPEN_DIR = os.path.join(OPEN_DIR, 'train')
VALID_OPEN_DIR = os.path.join(OPEN_DIR, 'val')
TEST_OPEN_DIR = os.path.join(OPEN_DIR, 'test')

hyper = Config.backbone
tensorboard = TensorBoard(os.path.join(keras_tb, hyper))
early_stopping = EarlyStopping(monitor='val_loss',patience=5, min_delta=0.001)
lr_reduce = ReduceLROnPlateau(monitor='loss',patience=20,verbose=1)
checkpoint = ModelCheckpoint(os.path.join(ROOT_DIR, 'keras_model', 'ckpt', hyper + '.h5'), period=1,save_best_only=True, monitor='val_acc')
callbacks = [tensorboard,lr_reduce,checkpoint]
batch_size = Config.image_per_gpu * Config.gpu_count

num_output = 8
if Config.backbone == 'resnet50':
    base_model = ResnetBuilder.build_resnet_50(input_shape=(224,224,3), num_outputs=num_output, block_fun='resnet')
if Config.backbone == 'resnet101':
    base_model = ResnetBuilder.build_resnet_101(input_shape=(224, 224, 3), num_outputs=num_output, block_fun='resnet')
if Config.backbone == 'resnet152':
    base_model = ResnetBuilder.build_resnet_152(input_shape=(224, 224, 3), num_outputs=num_output, block_fun='resnet')

if Config.backbone == 'xresnet50':
    base_model = ResnetBuilder.build_resnet_50(input_shape=(224,224,3), num_outputs=num_output, block_fun='xresnet')
if Config.backbone == 'xresnet101':
    base_model = ResnetBuilder.build_resnet_101(input_shape=(224, 224, 3), num_outputs=num_output, block_fun='xresnet')
if Config.backbone == 'xresnet152':
    base_model = ResnetBuilder.build_resnet_152(input_shape=(224, 224, 3), num_outputs=num_output, block_fun='xresnet')

if Config.backbone == 'dresnet50':
    base_model = ResnetBuilder.build_resnet_50(input_shape=(224,224,3), num_outputs=num_output, block_fun='dresnet')
if Config.backbone == 'dresnet101':
    base_model = ResnetBuilder.build_resnet_101(input_shape=(224, 224, 3), num_outputs=num_output, block_fun='dresnet')
if Config.backbone == 'dresnet152':
    base_model = ResnetBuilder.build_resnet_152(input_shape=(224, 224, 3), num_outputs=num_output, block_fun='dresnet')
base_model.summary()


loss = cls_cross_entropy()
optimizer = SGD(lr=Config.lr, momentum=0.9, decay=0.00001, nesterov=True)
train_gen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, rotation_range=10,
                               width_shift_range=0.1, height_shift_range=0.1, shear_range=10, zoom_range=0.1,
                               fill_mode='reflect', horizontal_flip=True, vertical_flip=True)
val_gen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
train_data = train_gen.flow_from_directory(TRAIN_OPEN_DIR, target_size=(224,224), batch_size=batch_size, shuffle=True, color_mode='rgb')
val_data = val_gen.flow_from_directory(VALID_OPEN_DIR, target_size=(224,224), batch_size=batch_size, shuffle=True, color_mode='rgb')

train_step_per_epoch = len(train_data)/batch_size
valid_step_per_epoch = len(val_data)/batch_size

base_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
base_model.fit_generator(train_data, steps_per_epoch=train_step_per_epoch,
                         epochs=200,callbacks=callbacks,
                         validation_data=val_data,validation_steps=valid_step_per_epoch)

model_json = base_model.to_json()
with open(os.path.join(json_keras, hyper + '.json'), 'w') as json_file:
    json_file.write(model_json)
base_model.save_weights(os.path.join(weights_keras, hyper + '.h5'))
print(hyper + 'has been saved')