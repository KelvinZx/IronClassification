from __future__ import division

import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D
from keras.layers import Activation
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from variant_res_module import resnet_convolution_block, resnet_identity_block, xresneXt_convolution_block, xresneXt_identity_block,\
    dresneXt_convolution_block, dresneXt_identity_block
import os

ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith('keras_model'):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization()(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _residual_block(input, id_block, conv_block, mid_f, output_f, repetitions, stage, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """

    for i in range(repetitions):
        if i == 0 and is_first_layer is True:
            input = conv_block(mid_f, output_f, stage, i, input, stride=(1, 1))
        elif i == 0 and is_first_layer is False:
            input = conv_block(mid_f, output_f, stage, i, input)
        else:
            input = id_block(mid_f, output_f, stage, i, input)
    return input



def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, repetitions, mid_f = [64, 128, 256, 512], output_f=[256, 512, 1024, 2048], block_fn='resnet'):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        #if K.image_dim_ordering() == 'tf':
        #    input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        if block_fn == 'xresnet':
            id_block = xresneXt_identity_block
            conv_block = xresneXt_convolution_block
        elif block_fn == 'dresnet':
            id_block = dresneXt_identity_block
            conv_block = dresneXt_convolution_block
        else:
            id_block = resnet_identity_block
            conv_block = resnet_convolution_block

        print('the input shape: {}'.format(input_shape))
        input = Input(shape=input_shape)
        # initial building block
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        #pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = conv1
        #filters = 64
        print('input before reisdual', block.shape)
        for i, r in enumerate(repetitions):
            if i == 0:
                block = _residual_block(block, id_block=id_block, conv_block=conv_block, stage=i,
                                        mid_f=mid_f[i], output_f=output_f[i], repetitions=r, is_first_layer=True)
            else:
                block = _residual_block(block, id_block=id_block, conv_block=conv_block, stage=i,
                                        mid_f=mid_f[i], output_f=output_f[i], repetitions=r, is_first_layer=False)


        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)

        pool2 = AveragePooling2D(pool_size=(block_shape[1], block_shape[2]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(num_outputs)(flatten1)
        dense = Activation('softmax', name='Softmax')(dense)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs, block_fun):
        return ResnetBuilder.build(input_shape, num_outputs, [2, 2, 2, 2], block_fn=block_fun)

    @staticmethod
    def build_resnet_34(input_shape, num_outputs, block_fun):
        return ResnetBuilder.build(input_shape, num_outputs, [3, 4, 6, 3], block_fn=block_fun)

    @staticmethod
    def build_resnet_50(input_shape, num_outputs, block_fun):
        return ResnetBuilder.build(input_shape, num_outputs, [3, 4, 6, 3], block_fn=block_fun)

    @staticmethod
    def build_resnet_101(input_shape, num_outputs, block_fun):
        return ResnetBuilder.build(input_shape, num_outputs, [3, 4, 23, 3], block_fn=block_fun)

    @staticmethod
    def build_resnet_152(input_shape, num_outputs, block_fun):
        return ResnetBuilder.build(input_shape, num_outputs, [3, 8, 36, 3], block_fn=block_fun)