import os
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2 as resnet_v2
from nets.mobilenet import mobilenet_v2
from datasets.utils import anchors_for_shape
from models.layers import build_head_cls, build_head_loc, conv_layer, resize_to_target
from models.utils import smooth_l1_loss, focal_loss, bbox_transform_inv
from models.nn import DetectNet

slim = tf.contrib.slim

class RetinaNet(DetectNet):
    """RetinaNet Class"""

    def __init__(self, input_shape, num_classes, anchors=None, **kwargs):
        self.anchors = anchors_for_shape(input_shape[:2]) if anchors is None else anchors
        self.y = tf.placeholder(tf.float32, [None, self.anchors.shape[0], 5 + num_classes])
        super(RetinaNet, self).__init__(input_shape, num_classes, **kwargs)
        # self.pred_y = self._build_pred_y(self)
        self.pred_y = self.pred

    def _build_model(self, **kwargs):
        d = dict()
        num_classes = self.num_classes
        frontend = kwargs.pop('frontend', 'resnet_v2_50')
        num_anchors = kwargs.pop('num_anchors', 9)

        if 'resnet_v2' in frontend:
            d['feature_map'] = self.X - [[[123.68, 116.779, 103.939]]]
            frontend_dir = os.path.join('pretrained_models', '{}.ckpt'.format(frontend))
            with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                logits, end_points = resnet_v2.resnet_v2_50(d['feature_map'], is_training=self.is_train)
                d['init_fn'] = slim.assign_from_checkpoint_fn(model_path=frontend_dir,
                                                          var_list=slim.get_model_variables(frontend))
            convs = [end_points[frontend + '/block{}'.format(x)] for x in [4, 2, 1]]
        elif 'mobilenet_v2' in frontend:
            d['feature_map'] = (2.0 / 255.0) * self.X  - 1.0
            frontend_dir = os.path.join('pretrained_models', 'mobilenet_v2_1.4_224', '{}.ckpt'.format(frontend))
            with slim.arg_scope(mobilenet_v2.training_scope()):
                _, end_points = mobilenet_v2.mobilenet_base(d['feature_map'], is_training=self.is_train)

            convs = [end_points[x] for x in ['layer_19', 'layer_14', 'layer_7']]
        else:
            #TODO build convNet
            raise NotImplementedError("Build own convNet!")

        with tf.variable_scope('layer5'):
            d['s_5'] = conv_layer(convs[0], 256, (1, 1), (1, 1))
            d['cls_head5'] = build_head_cls(d['s_5'], num_anchors, num_classes + 1)
            d['loc_head5'] = build_head_loc(d['s_5'], num_anchors)
            d['flat_cls_head5'] = tf.reshape(d['cls_head5'], (tf.shape(d['cls_head5'])[0], -1, num_classes + 1))
            d['flat_loc_head5'] = tf.reshape(d['loc_head5'], (tf.shape(d['loc_head5'])[0], -1, 4))

        with tf.variable_scope('layer6'):
            d['s_6'] = conv_layer(d['s_5'], 256, (3, 3), (2, 2))
            d['cls_head6'] = build_head_cls(d['s_6'], num_anchors, num_classes + 1)
            d['loc_head6'] = build_head_loc(d['s_6'], num_anchors)
            d['flat_cls_head6'] = tf.reshape(d['cls_head6'], (tf.shape(d['cls_head6'])[0], -1, num_classes + 1))
            d['flat_loc_head6'] = tf.reshape(d['loc_head6'], (tf.shape(d['loc_head6'])[0], -1, 4))

        with tf.variable_scope('layer7'):
            d['s_7'] = conv_layer(tf.nn.relu(d['s_6']), 256, (3, 3), (2, 2))
            d['cls_head7'] = build_head_cls(d['s_7'], num_anchors, num_classes + 1)
            d['loc_head7'] = build_head_loc(d['s_7'], num_anchors)
            d['flat_cls_head7'] = tf.reshape(d['cls_head7'], (tf.shape(d['cls_head7'])[0], -1, num_classes + 1))
            d['flat_loc_head7'] = tf.reshape(d['loc_head7'], (tf.shape(d['loc_head7'])[0], -1, 4))

        with tf.variable_scope('layer4'):
            d['up4'] = resize_to_target(d['s_5'], convs[1])
            d['s_4'] = conv_layer(convs[1], 256, (1, 1), (1, 1)) + d['up4']
            d['cls_head4'] = build_head_cls(d['s_4'], num_anchors, num_classes + 1)
            d['loc_head4'] = build_head_loc(d['s_4'], num_anchors)
            d['flat_cls_head4'] = tf.reshape(d['cls_head4'], (tf.shape(d['cls_head4'])[0], -1, num_classes + 1))
            d['flat_loc_head4'] = tf.reshape(d['loc_head4'], (tf.shape(d['loc_head4'])[0], -1, 4))

        with tf.variable_scope('layer3'):
            d['up3'] = resize_to_target(d['s_4'], convs[2])
            d['s_3'] = conv_layer(convs[2], 256, (1, 1), (1, 1)) + d['up3']
            d['cls_head3'] = build_head_cls(d['s_3'], num_anchors, num_classes + 1)
            d['loc_head3'] = build_head_loc(d['s_3'], num_anchors)
            d['flat_cls_head3'] = tf.reshape(d['cls_head3'], (tf.shape(d['cls_head3'])[0], -1, num_classes + 1))
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
            d['pred'] = tf.concat((d['loc_head'], tf.nn.softmax(d['cls_head'], axis=-1)), axis=2)

        return d

    def _build_loss(self, **kwargs):
        r_alpha = kwargs.pop('r_alpha', 1)
        with tf.variable_scope('losses'):
            conf_loss = focal_loss(self.logits, self.y)
            regress_loss = smooth_l1_loss(self.logits, self.y)
            total_loss = conf_loss + r_alpha * regress_loss

        # for debug
        self.conf_loss = conf_loss
        self.regress_loss = regress_loss
        return total_loss

    # def _build_pred_y(self, **kwargs):
    #     regressions  = self.pred[:, :, :4]
    #     regressions = tf.py_func(bbox_transform_inv, [self.anchors, regressions], tf.float32)
    #     pred_y = tf.concat((regressions, self.pred[:, :, 4:]), axis=2)
    #     return pred_y
