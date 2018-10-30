import tensorflow as tf
import numpy as np

def conv_layer(x, filters, kernel_size, strides, padding='SAME', use_bias=True):
    return tf.layers.conv2d(x, filters, kernel_size, strides, padding, use_bias=use_bias)

def max_pool(x, side_l, stride, padding='SAME'):
    """
    Performs max pooling on given input.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param side_l: int, the side length of the pooling window for each dimension.
    :param stride: int, the stride of the sliding window for each dimension.
    :param padding: str, either 'SAME' or 'VALID',
                         the type of padding algorithm to use.
    :return: tf.Tensor.
    """
    return tf.nn.max_pool(x, ksize=[1, side_l, side_l, 1],
                          strides=[1, stride, stride, 1], padding=padding)

def batchNormalization(x, is_train):
    """
    Add a new batchNormalization layer.
    :param x: tf.Tensor, shape: (N, H, W, C) or (N, D)
    :param is_train: tf.placeholder(bool), if True, train mode, else, test mode
    :return: tf.Tensor.
    """
    return tf.layers.batch_normalization(x, training=is_train, momentum=0.99, epsilon=0.001, center=True, scale=True)

def conv_bn_relu(x, filters, kernel_size, is_train, strides=(1, 1), padding='SAME', relu=True):
    """
    Add conv + bn + Relu layers.
    see conv_layer and batchNormalization function
    """
    conv = conv_layer(x, filters, kernel_size, strides, padding, use_bias=False)
    bn = batchNormalization(conv, is_train)
    if relu:
        return tf.nn.relu(bn)
    else:
        return bn

def build_head_loc(x, num_anchors, depth=4, name='head_loc'):
    head = x
    with tf.variable_scope(name):
        for _ in range(depth):
            head = tf.nn.relu(conv_layer(head, 256, (3, 3), (1, 1)))
        output_channels = num_anchors * 4
        head = conv_layer(head, output_channels, (3, 3), (1, 1))
    return head

def build_head_cls(x, num_anchors, num_classes, depth=4, prior_probs=0.01, name='head_cls'):
    head = x
    with tf.variable_scope(name):
        for _ in range(depth):
            head = tf.nn.relu(conv_layer(head, 256, (3, 3), (1, 1)))
        output_channels = num_anchors * num_classes
        bias = np.zeros((num_classes, 1, 1), dtype=np.float32)
        bias[0] = np.log((num_classes - 1) * (1 - prior_probs) / (prior_probs))
        bias = np.vstack([bias for _ in range(num_anchors)])
        biases = tf.get_variable('biases', [num_anchors * num_classes], tf.float32,\
                                tf.constant_initializer(value=bias))
        head = conv_layer(head, output_channels, (3, 3), (1, 1), use_bias=False) + biases
    return head

def resize_to_target(x, target):
    size = (tf.shape(target)[1], tf.shape(target)[2])
    x = tf.image.resize_bilinear(x, size)
    return tf.cast(x, x.dtype)

def residual(x, input_channels, output_channels, is_train, strides=(1, 1), name='residual_block', st=True):
    """
    Build series of residual blocks
    """
    shortcut = x
    with tf.variable_scope(name):
        x = conv_bn_relu(x, input_channels, (1, 1), is_train, strides=(1, 1), padding='SAME')
        x = conv_bn_relu(x, input_channels, (3, 3), is_train, strides=strides, padding='SAME')
        x = conv_bn_relu(x, output_channels, (1, 1), is_train, strides=(1, 1), padding='SAME', relu=False)

        if strides != (1, 1) or st is not True:
            shortcut = conv_bn_relu(shortcut, output_channels, (1, 1), is_train, strides=strides, padding='SAME', relu=False)

        x = tf.add(x, shortcut)
        x = tf.nn.relu(x)

    return x

