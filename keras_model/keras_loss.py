import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
epsilon = 1e-7
cls_threshold = 0.8

def cls_cross_entropy():
    #print(weights.shape)
    #weights = tf.convert_to_tensor(weights, np.float32)#K.variable(weights)
    def _cls_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1-epsilon)
        smoother = K.pow((1 - y_pred), 0.5)
        result = -K.mean(smoother * K.log(y_pred) * y_true)
        return result
    return _cls_loss
