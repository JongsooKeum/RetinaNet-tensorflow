from builders import frontend_builder
import tensorflow as tf
from abc import abstractmethod, ABCMeta
from models.layers import build_head_cls, build_head_loc, conv_layer, resize_to_target

class DetectNet(metaclass=ABCMeta):
    """Base class for Convolutional Neural Networks for detection."""

    def __init__(self, input_shape, num_classes, **kwargs):
        """
        model initializer
        :param input_shape: tuple, shape (H, W, C)
        :param num_classes: int, total number of classes
        """
        self.X = tf.placeholder(tf.float32, [None] + input_shape)
        self.is_train = tf.placeholder(tf.bool)
        self.num_classes = num_classes
        self.d = self._build_model(**kwargs)
        self.logits = self.d['logits']
        self.loss = self._build_loss(**kwargs)

    @abstractmethod
    def _build_model(self, **kwargs):
        """
        Build model.
        This should be implemented.
        """
        pass

    @abstractmethod
    def _build_loss(self, **kwargs):
        """
        build loss function for the model training.
        This should be implemented.
        """
        pass

    @abstractmethod
    def predict(self, sess, dataset, verbose=False, **kwargs):
        """
        Predict bounding box.
        This should be implemented.
        """
        pass

class RetinaNet(DetectNet):
    """RetinaNet Class"""

    def __init__(self, input_shape, num_classes, anchors, **kwargs):
        #self.anchors = anchors_for_shape(input_shape[-2]) if anchors is None else anchors
        super(RetinaNet, self).__init__(input_shape, num_classes, **kwargs)

    def _build_model(self, **kwargs):
        d = dict()
        num_classes = self.num_classes
        pretrain = kwargs.pop('pretrain', True)
        frontend = kwargs.pop('frontend', 'ResNet50')
        num_anchors = kwargs.pop('num_anchors', 9)

        if pretrain:
            logits, end_points, frontend_scope, init_fn = frontend_builder.build_frontend(self.X, frontend)
            convs = [end_points['pool5'], end_points['pool4'], end_points['pool3'], end_points['pool2']]
        else:
            #TODO build convNet
            raise NotImplementedError("Build own convNet!")

        with tf.variable_scope('layer5'):
            d['s_5'] = conv_layer(convs[0], 256, (1, 1), (1, 1))
            d['cls_head5'] = build_head_cls(d['s_5'], num_anchors, num_classes+1)
            d['loc_head5'] = build_head_loc(d['s_5'], num_anchors)
            d['flat_cls_head5'] = tf.reshape(d['cls_head5'], (tf.shape(d['cls_head5'])[0], -1, num_classes+1))
            d['flat_loc_head5'] = tf.reshape(d['loc_head5'], (tf.shape(d['loc_head5'])[0], -1, 4))

        with tf.variable_scope('layer6'):
            d['s_6'] = conv_layer(d['s_5'], 256, (3, 3), (2, 2))
            d['cls_head6'] = build_head_cls(d['s_6'], num_anchors, num_classes+1)
            d['loc_head6'] = build_head_loc(d['s_6'], num_anchors)
            d['flat_cls_head6'] = tf.reshape(d['cls_head6'], (tf.shape(d['cls_head6'])[0], -1, num_classes+1))
            d['flat_loc_head6'] = tf.reshape(d['loc_head6'], (tf.shape(d['loc_head6'])[0], -1, 4))

        with tf.variable_scope('layer7'):
            d['s_7'] = conv_layer(tf.nn.relu(d['s_6']), 256, (3, 3), (2, 2))
            d['cls_head7'] = build_head_cls(d['s_7'], num_anchors, num_classes+1)
            d['loc_head7'] = build_head_loc(d['s_7'], num_anchors)
            d['flat_cls_head7'] = tf.reshape(d['cls_head7'], (tf.shape(d['cls_head7'])[0], -1, num_classes+1))
            d['flat_loc_head7'] = tf.reshape(d['loc_head7'], (tf.shape(d['loc_head7'])[0], -1, 4))

        with tf.variable_scope('layer4'):
            d['up4'] = resize_to_target(d['s_5'], convs[1])
            d['s_4'] = conv_layer(convs[1], 256, (1, 1), (1, 1)) + d['up4']
            d['cls_head4'] = build_head_cls(d['s_4'], num_anchors, num_classes+1)
            d['loc_head4'] = build_head_loc(d['s_4'], num_anchors)
            d['flat_cls_head4'] = tf.reshape(d['cls_head4'], (tf.shape(d['cls_head4'])[0], -1, num_classes+1))
            d['flat_loc_head4'] = tf.reshape(d['loc_head4'], (tf.shape(d['loc_head4'])[0], -1, 4))

        with tf.variable_scope('layer3'):
            d['up3'] = resize_to_target(d['s_4'], convs[2])
            d['s_3'] = conv_layer(convs[2], 256, (1, 1), (1, 1)) + d['up3']
            d['cls_head3'] = build_head_cls(d['s_3'], num_anchors, num_classes+1)
            d['loc_head3'] = build_head_loc(d['s_3'], num_anchors)
            d['flat_cls_head3'] = tf.reshape(d['cls_head3'], (tf.shape(d['cls_head3'])[0], -1, num_classes+1))
            d['flat_loc_head3'] = tf.reshape(d['loc_head3'], (tf.shape(d['loc_head3'])[0], -1, 4))

        with tf.variable_scope('head'):
            d['cls_head'] = tf.concat((d['flat_cls_head3'],
                                       d['flat_cls_head4'],
                                       d['flat_cls_head5'],
                                       d['flat_cls_head6'],
                                       d['flat_cls_head7']), axis=1)

            d['loc_head'] = tf.concat((d['flat_loc_head3'],
                                       d['flat_loc_head4'],
                                       d['flat_loc_head5'],
                                       d['flat_loc_head6'],
                                       d['flat_loc_head7']), axis=1)

            d['logits'] = tf.concat((d['loc_head'], d['cls_head']), axis=2)

        return d

    def _build_loss(self, **kwargs):
        print('hi')

    def predict(self, sess, dataset, verbose=False, **kwargs):
        print('pre')
