from keras.layers.convolutional import Conv2D
from keras.layers import Activation, Add
from keras.layers.normalization import BatchNormalization
import numpy as np
###########################################
# ResNet Graph
###########################################

def _deep_res_bottleneck(inputs, f, kernel_size, stage, block, step, d_rate=1, stride=(1,1)):
    x = Conv2D(filters=f, kernel_size=kernel_size, padding='same', strides=stride, dilation_rate=d_rate,
               name=str(stage) + '_' + str(block) + '_rescomb_conv_'+str(step))(inputs)
    #print('stage: {} block: {} step: {} shape: {}'.format(stage, block, step, x.shape))
    x = BatchNormalization(name=str(stage) + '_' + str(block) + '_rescomb_BN_'+str(step))(x)
    x = Activation('relu', name=str(stage) + '_' + str(block) + '_rescomb_act_'+str(step))(x)
    return x


def resnet_identity_block(mid_f, output_f, stage, block, inputs, trainable=True):
    """
    :param f: number of filters
    :param stage: stage of residual blocks
    :param block: ith module
    :param trainable: freeze layer if false
    """
    x_shortcut = inputs
    x = _deep_res_bottleneck(inputs, mid_f, (1, 1), stage=stage, block=block, step=1)
    x = _deep_res_bottleneck(x, mid_f, (3, 3), stage=stage, block=block, step=2)
    x = Conv2D(filters=output_f, kernel_size=(1, 1), padding='same',
               name=str(stage) + '_' + str(block) + '_idblock_conv_3')(x)
    x = BatchNormalization(name=str(stage) + '_' + str(block) + '_idblock_BN_3',
                           trainable=trainable)(x)

    x_add = Add(name=str(stage) + '_' + str(block) + '_idblock_add', trainable=trainable)([x, x_shortcut])
    x_idblock_output = Activation('relu', name=str(stage) + '_' + str(block) + '_idblock_act_outout',
                                  trainable=trainable)(x_add)
    return x_idblock_output


def resnet_convolution_block(mid_f, output_f, stage, block, inputs, stride=(2,2), trainable=True):
    """
    :param f: number of filters
    :param stage: stage of residual blocks
    :param block: ith module
    """
    x = _deep_res_bottleneck(inputs, mid_f, (1, 1), stage=stage, block=block, step=1, stride=stride)
    x = _deep_res_bottleneck(x, mid_f, (3, 3), stage=stage, block=block, step=2)
    x = Conv2D(filters=output_f, kernel_size=(1, 1), padding='same',
               name=str(stage) + '_' + str(block) + '_idblock_conv_3')(x)
    x = BatchNormalization(name=str(stage) + '_' + str(block) + '_idblock_BN_3',
                           trainable=trainable)(x)


    x_shortcut = Conv2D(output_f, kernel_size=(1, 1), strides=stride, padding='same',
                        name=str(stage) + '_' + str(block) + '_convblock_shortcut_conv',
                        trainable=trainable)(inputs)
    x_shortcut = BatchNormalization(name=str(stage) + '_' + str(block) + '_convblock_shortcut_BN_1',
                                    trainable=trainable)(x_shortcut)
    x_add = Add(name=str(stage) + '_' + str(block) + '_convblock_add',
                trainable=trainable)([x, x_shortcut])
    x_convblock_output = Activation('relu', name=str(stage) + '_' + str(block) + '_convblock_act_output',
                                    trainable=trainable)(x_add)
    return x_convblock_output


###################
# ResNeXT
###################

def _resneXt_bottleneck(inputs, mid_f, output_f, stage, block, width, d_rate=1, stride=(1,1)):
    x = _deep_res_bottleneck(inputs, mid_f, (1, 1), stage=stage, block=block, step=100+width, stride=stride)
    x = _deep_res_bottleneck(x, mid_f, (3, 3), d_rate=d_rate, stage=stage, block=block, step=200+width)
    x = Conv2D(filters=output_f, kernel_size=(1, 1), padding='same',
               name=str(stage) + '_' + str(block) + '_resneXtbot_conv_width' + str(width))(x)
    x = BatchNormalization(name=str(stage) + '_' + str(block) + '_resneXtbot_BN_'+ str(width))(x)
    return x


def xresneXt_identity_block(mid_f, output_f, stage, block, inputs, width=32, trainable=True):
    """
    :param f: number of filters
    :param stage: stage of residual blocks
    :param block: ith module
    :param trainable: freeze layer if false
    """
    x_shortcut = inputs
    f_perblock = int(output_f / (2 * width))
    x_add = inputs
    for i in range(width):
        newx = _resneXt_bottleneck(inputs, f_perblock, output_f, stage, block, width=i)
        x_add = Add(name=str(stage) + '_' + str(block) + '_idblock_add_width' + str(i), trainable=trainable)([x_add, newx])

    x_final_add = Add(name=str(stage) + '_' + str(block) + 'final_idblock_add', trainable=trainable)([x_add, x_shortcut])
    x_idblock_output = Activation('relu', name=str(stage) + '_' + str(block) + '_idblock_act_outout',
                                  trainable=trainable)(x_final_add)
    return x_idblock_output


def xresneXt_convolution_block(mid_f, output_f, stage, block, inputs, width=32, stride=(2,2),trainable=True):
    """
    :param f: number of filters
    :param stage: stage of residual blocks
    :param block: ith module
    :param trainable: freeze layer if false
    """
    x_shortcut = inputs

    f_perblock = int(output_f / (2*width))
    for i in range(width):
        if i == 0:
            newx = _resneXt_bottleneck(inputs, f_perblock, output_f, stage, block, width=i, stride=stride)
            x_add = newx
        else:
            newx = _resneXt_bottleneck(inputs, f_perblock, output_f, stage, block, width=i, stride=stride)
            x_add = Add(name=str(stage) + '_' + str(block) + '_convblock_add_width' + str(i), trainable=trainable)([x_add, newx])

    x_shortcut = Conv2D(output_f, kernel_size=(1, 1), strides=stride, padding='same',
                        name=str(stage) + '_' + str(block) + '_convblock_shortcut_conv',
                        trainable=trainable)(x_shortcut)
    x_shortcut = BatchNormalization(name=str(stage) + '_' + str(block) + '_convblock_shortcut_BN_1',
                                    trainable=trainable)(x_shortcut)

    x_final_add = Add(name=str(stage) + '_' + str(block) + 'final_convblock_add', trainable=trainable)([x_add, x_shortcut])
    x_convblock_output = Activation('relu', name=str(stage) + '_' + str(block) + '_convblock_act_outout',
                                  trainable=trainable)(x_final_add)
    return x_convblock_output


def dresneXt_identity_block(mid_f, output_f, stage, block, inputs, d_rate=2, width=32, trainable=True):
    """
    :param f: number of filters
    :param stage: stage of residual blocks
    :param block: ith module
    :param trainable: freeze layer if false
    """
    x_shortcut = inputs

    x_add = inputs
    f_perblock = int(output_f / (2 * width))
    for i in range(width):
        newx = _resneXt_bottleneck(inputs,f_perblock, output_f, stage, block, d_rate=d_rate, width=i)
        x_add = Add(name=str(stage) + '_' + str(block) + '_idblock_add_width' + str(i), trainable=trainable)([x_add, newx])

    x_final_add = Add(name=str(stage) + '_' + str(block) + '_idblock_add', trainable=trainable)([x_add, x_shortcut])
    x_idblock_output = Activation('relu', name=str(stage) + '_' + str(block) + '_idblock_act_output',
                                  trainable=trainable)(x_final_add)
    return x_idblock_output


def dresneXt_convolution_block(mid_f, output_f, stage, block, inputs, d_rate=2, width=32, stride=(2, 2), trainable=True):
    """
    :param f: number of filters
    :param stage: stage of residual blocks
    :param block: ith module
    :param trainable: freeze layer if false
    """
    x_shortcut = inputs
    f_perblock = int(output_f / (2 * width))
    for i in range(width):
        if i == 0:
            newx = _resneXt_bottleneck(inputs, f_perblock, output_f, stage, block, stride=stride, d_rate=d_rate, width=i)
            x_add = newx
        else:
            newx = _resneXt_bottleneck(inputs, f_perblock, output_f, stage, block, stride=stride, d_rate=d_rate, width=i)
            x_add = Add(name=str(stage) + '_' + str(block) + '_convblock_add_width'+str(i) , trainable=trainable)([x_add, newx])

    x_shortcut = Conv2D(output_f, kernel_size=(1, 1), strides=stride, padding='same',
                        name=str(stage) + '_' + str(block) + '_convblock_shortcut_conv',
                        trainable=trainable)(x_shortcut)
    x_shortcut = BatchNormalization(name=str(stage) + '_' + str(block) + '_convblock_shortcut_BN_1',
                                    trainable=trainable)(x_shortcut)

    x_final_add = Add(name=str(stage) + '_' + str(block) + 'final_convblock_add', trainable=trainable)([x_add, x_shortcut])
    x_convblock_output = Activation('relu', name=str(stage) + '_' + str(block) + '_convblock_act_outout',
                                  trainable=trainable)(x_final_add)
    return x_convblock_output