import tensorflow as tf
from models.layers import conv_layer, max_pool, batchNormalization

class YOLO(DetectNet):
    """YOLO class"""

    def __init__(self, input_shape, num_classes, anchors, **kwargs):

        self.grid_size = grid_size = [x // 32 for x in input_shape[:2]]
        self.num_anchors = len(anchors)
        self.anchors = anchors
        self.y = tf.placeholder(tf.float32, [None] +
                                [self.grid_size[0], self.grid_size[1], self.num_anchors, 5 + num_classes])
        super(YOLO, self).__init__(input_shape, num_classes, **kwargs)

    def _build_model(self, **kwargs):
        """
        Build model.
        :param kwargs: dict, extra arguments for building YOLO.
                -image_mean: np.ndarray, mean image for each input channel, shape: (C,).
        :return d: dict, containing outputs on each layer.
        """

        d = dict()
        x_mean = kwargs.pop('image_mean', 0.0)

        # input
        X_input = self.X - x_mean
        is_train = self.is_train

        #conv1 - batch_norm1 - leaky_relu1 - pool1
        with tf.variable_scope('layer1'):
            d['conv1'] = conv_layer(X_input, 32, (3, 3), 1,
                                    padding='SAME', use_bias=False)
            d['batch_norm1'] = batchNormalization(d['conv1'], is_train)
            d['leaky_relu1'] = tf.nn.leaky_relu(d['batch_norm1'], alpha=0.1)
            d['pool1'] = max_pool(d['leaky_relu1'], 2, 2, padding='SAME')
        # (416, 416, 3) --> (208, 208, 32)
        print('layer1.shape', d['pool1'].get_shape().as_list())

        #conv2 - batch_norm2 - leaky_relu2 - pool2
        with tf.variable_scope('layer2'):
            d['conv2'] = conv_layer(d['pool1'], 64, (3, 3), 1,
                                    padding='SAME', use_bias=False)
            d['batch_norm2'] = batchNormalization(d['conv2'], is_train)
            d['leaky_relu2'] = tf.nn.leaky_relu(d['batch_norm2'], alpha=0.1)
            d['pool2'] = max_pool(d['leaky_relu2'], 2, 2, padding='SAME')
        # (208, 208, 32) --> (104, 104, 64)
        print('layer2.shape', d['pool2'].get_shape().as_list())

        #conv3 - batch_norm3 - leaky_relu3
        with tf.variable_scope('layer3'):
            d['conv3'] = conv_layer(d['pool2'], 128, (3, 3), 1,
                                    padding='SAME', use_bias=False)
            d['batch_norm3'] = batchNormalization(d['conv3'], is_train)
            d['leaky_relu3'] = tf.nn.leaky_relu(d['batch_norm3'], alpha=0.1)
        # (104, 104, 64) --> (104, 104, 128)
        print('layer3.shape', d['leaky_relu3'].get_shape().as_list())

        #conv4 - batch_norm4 - leaky_relu4
        with tf.variable_scope('layer4'):
            d['conv4'] = conv_layer(d['leaky_relu3'], 64, (1, 1), 1,
                                    padding='SAME', use_bias=False)
            d['batch_norm4'] = batchNormalization(d['conv4'], is_train)
            d['leaky_relu4'] = tf.nn.leaky_relu(d['batch_norm4'], alpha=0.1)
        # (104, 104, 128) --> (104, 104, 64)
        print('layer4.shape', d['leaky_relu4'].get_shape().as_list())

        #conv5 - batch_norm5 - leaky_relu5 - pool5
        with tf.variable_scope('layer5'):
            d['conv5'] = conv_layer(d['leaky_relu4'], 128, (3, 3), 1,
                                    padding='SAME', use_bias=False)
            d['batch_norm5'] = batchNormalization(d['conv5'], is_train)
            d['leaky_relu5'] = tf.nn.leaky_relu(d['batch_norm5'], alpha=0.1)
            d['pool5'] = max_pool(d['leaky_relu5'], 2, 2, padding='SAME')
        # (104, 104, 64) --> (52, 52, 128)
        print('layer5.shape', d['pool5'].get_shape().as_list())

        #conv6 - batch_norm6 - leaky_relu6
        with tf.variable_scope('layer6'):
            d['conv6'] = conv_layer(d['pool5'], 256, (3, 3), 1,
                                    padding='SAME', use_bias=False)
            d['batch_norm6'] = batchNormalization(d['conv6'], is_train)
            d['leaky_relu6'] = tf.nn.leaky_relu(d['batch_norm6'], alpha=0.1)
        # (52, 52, 128) --> (52, 52, 256)
        print('layer6.shape', d['leaky_relu6'].get_shape().as_list())

        #conv7 - batch_norm7 - leaky_relu7
        with tf.variable_scope('layer7'):
            d['conv7'] = conv_layer(d['leaky_relu6'], 128, (1, 1), 1,
                                    padding='SAME', use_bias=False)
            d['batch_norm7'] = batchNormalization(d['conv7'], is_train)
            d['leaky_relu7'] = tf.nn.leaky_relu(d['batch_norm7'], alpha=0.1)
        # (52, 52, 256) --> (52, 52, 128)
        print('layer7.shape', d['leaky_relu7'].get_shape().as_list())

        #conv8 - batch_norm8 - leaky_relu8 - pool8
        with tf.variable_scope('layer8'):
            d['conv8'] = conv_layer(d['leaky_relu7'], 256, (3, 3), 1,
                                    padding='SAME', use_bias=False)
            d['batch_norm8'] = batchNormalization(d['conv8'], is_train)
            d['leaky_relu8'] = tf.nn.leaky_relu(d['batch_norm8'], alpha=0.1)
            d['pool8'] = max_pool(d['leaky_relu8'], 2, 2, padding='SAME')
        # (52, 52, 128) --> (26, 26, 256)
        print('layer8.shape', d['pool8'].get_shape().as_list())

        #conv9 - batch_norm9 - leaky_relu9
        with tf.variable_scope('layer9'):
            d['conv9'] = conv_layer(d['pool8'], 512, (3, 3), 1,
                                    padding='SAME', use_bias=False)
            d['batch_norm9'] = batchNormalization(d['conv9'], is_train)
            d['leaky_relu9'] = tf.nn.leaky_relu(d['batch_norm9'], alpha=0.1)
        # (26, 26, 256) --> (26, 26, 512)
        print('layer9.shape', d['leaky_relu9'].get_shape().as_list())

        #conv10 - batch_norm10 - leaky_relu10
        with tf.variable_scope('layer10'):
            d['conv10'] = conv_layer(d['leaky_relu9'], 256, (1, 1), 1,
                                     padding='SAME', use_bias=False)
            d['batch_norm10'] = batchNormalization(d['conv10'], is_train)
            d['leaky_relu10'] = tf.nn.leaky_relu(d['batch_norm10'], alpha=0.1)
        # (26, 26, 512) --> (26, 26, 256)
        print('layer10.shape', d['leaky_relu10'].get_shape().as_list())

        #conv11 - batch_norm11 - leaky_relu11
        with tf.variable_scope('layer11'):
            d['conv11'] = conv_layer(d['leaky_relu10'], 512, (3, 3), 1,
                                     padding='SAME', use_bias=False)
            d['batch_norm11'] = batchNormalization(d['conv11'], is_train)
            d['leaky_relu11'] = tf.nn.leaky_relu(d['batch_norm11'], alpha=0.1)
        # (26, 26, 256) --> (26, 26, 512)
        print('layer11.shape', d['leaky_relu11'].get_shape().as_list())

        #conv12 - batch_norm12 - leaky_relu12
        with tf.variable_scope('layer12'):
            d['conv12'] = conv_layer(d['leaky_relu11'], 256, (1, 1), 1
                                     padding='SAME', use_bias=False)
            d['batch_norm12'] = batchNormalization(d['conv12'], is_train)
            d['leaky_relu12'] = tf.nn.leaky_relu(d['batch_norm12'], alpha=0.1)
        # (26, 26, 512) --> (26, 26, 256)
        print('layer12.shape', d['leaky_relu12'].get_shape().as_list())

        #conv13 - batch_norm13 - leaky_relu13 - pool13
        with tf.variable_scope('layer13'):
            d['conv13'] = conv_layer(d['leaky_relu12'], 512, (3, 3), 1,
                                     padding='SAME', use_bias=False)
            d['batch_norm13'] = batchNormalization(d['conv13'], is_train)
            d['leaky_relu13'] = tf.nn.leaky_relu(d['batch_norm13'], alpha=0.1)
            d['pool13'] = max_pool(d['leaky_relu13'], 2, 2, padding='SAME')
        # (26, 26, 256) --> (13, 13, 512)
        print('layer13.shape', d['pool13'].get_shape().as_list())

        #conv14 - batch_norm14 - leaky_relu14
        with tf.variable_scope('layer14'):
            d['conv14'] = conv_layer(d['pool13'], 1024, (3, 3), 1,
                                     padding='SAME', use_bias=False)
            d['batch_norm14'] = batchNormalization(d['conv14'], is_train)
            d['leaky_relu14'] = tf.nn.leaky_relu(d['batch_norm14'], alpha=0.1)
        # (13, 13, 512) --> (13, 13, 1024)
        print('layer14.shape', d['leaky_relu14'].get_shape().as_list())

        #conv15 - batch_norm15 - leaky_relu15
        with tf.variable_scope('layer15'):
            d['conv15'] = conv_layer(d['leaky_relu14'], 512, (1, 1), 1,
                                     padding='SAME', use_bias=False)
            d['batch_norm15'] = batchNormalization(d['conv15'], is_train)
            d['leaky_relu15'] = tf.nn.leaky_relu(d['batch_norm15'], alpha=0.1)
        # (13, 13, 1024) --> (13, 13, 512)
        print('layer15.shape', d['leaky_relu15'].get_shape().as_list())

        #conv16 - batch_norm16 - leaky_relu16
        with tf.variable_scope('layer16'):
            d['conv16'] = conv_layer(d['leaky_relu15'], 1024, (3, 3), 1,
                                     padding='SAME', use_bias=False)
            d['batch_norm16'] = batchNormalization(d['conv16'], is_train)
            d['leaky_relu16'] = tf.nn.leaky_relu(d['batch_norm16'], alpha=0.1)
        # (13, 13, 512) --> (13, 13, 1024)
        print('layer16.shape', d['leaky_relu16'].get_shape().as_list())

        #conv17 - batch_norm16 - leaky_relu17
        with tf.variable_scope('layer17'):
            d['conv17'] = conv_layer(d['leaky_relu16'], 512, (1, 1), 1,
                                     padding='SAME', use_bias=False)
            d['batch_norm17'] = batchNormalization(d['conv17'], is_train)
            d['leaky_relu17'] = tf.nn.leaky_relu(d['batch_norm17'], alpha=0.1)
        # (13, 13, 1024) --> (13, 13, 512)
        print('layer17.shape', d['leaky_relu17'].get_shape().as_list())

        #conv18 - batch_norm18 - leaky_relu18
        with tf.variable_scope('layer18'):
            d['conv18'] = conv_layer(d['leaky_relu17'], 1024, (3, 3), 1,
                                     padding='SAME', use_bias=False)
            d['batch_norm18'] = batchNormalization(d['conv18'], is_train)
            d['leaky_relu18'] = tf.nn.leaky_relu(d['batch_norm18'], alpha=0.1)
        # (13, 13, 512) --> (13, 13, 1024)
        print('layer18.shape', d['leaky_relu18'].get_shape().as_list())

        #conv19 - batch_norm19 - leaky_relu19
        with tf.variable_scope('layer19'):
            d['conv19'] = conv_layer(d['leaky_relu18'], 1024, (3, 3), 1,
                                     padding='SAME', use_bias=False)
            d['batch_norm19'] = batchNormalization(d['conv19'], is_train)
            d['leaky_relu19'] = tf.nn.leaky_relu(d['batch_norm19'], alpha=0.1)
        # (13, 13, 1024) --> (13, 13, 1024)
        print('layer19.shape', d['leaky_relu19'].get_shape().as_list())

        #conv20 - batch_norm20 - leaky_relu20
        with tf.variable_scope('layer20'):
            d['conv20'] = conv_layer(d['leaky_relu19'], 1024, (3, 3), 1,
                                     padding='SAME', use_bias=False)
            d['batch_norm20'] = batchNormalization(d['conv20'], is_train)
            d['leaky_relu20'] = tf.nn.leaky_relu(d['batch_norm20'], alpha=0.1)
        # (13, 13, 1024) --> (13, 13, 1024)
        print('layer20.shape', d['leaky_relu20'].get_shape().as_list())

        # concatenate layer20 and layer 13 using space to depth
        with tf.variable_scope('layer21'):
            d['skip_connection'] = conv_layer(d['leaky_relu13'], 64, (1, 1), 1,
                                              padding='SAME', use_bias=False)
            d['skip_batch'] = batchNormalization(
                d['skip_connection'], is_train)
            d['skip_leaky_relu'] = tf.nn.leaky_relu(d['skip_batch'], alpha=0.1)
            d['skip_space_to_depth_x2'] = tf.space_to_depth(
                d['skip_leaky_relu'], block_size=2)
            d['concat21'] = tf.concat(
                [d['skip_space_to_depth_x2'], d['leaky_relu20']], axis=-1)
        # (13, 13, 1024) --> (13, 13, 256+1024)
        print('layer21.shape', d['concat21'].get_shape().as_list())

        #conv22 - batch_norm22 - leaky_relu22
        with tf.variable_scope('layer22'):
            d['conv22'] = conv_layer(d['concat21'], 1024, (3, 3), 1,
                                     padding='SAME', use_bias=False)
            d['batch_norm22'] = batchNormalization(d['conv22'], is_train)
            d['leaky_relu22'] = tf.nn.leaky_relu(d['batch_norm22'], alpha=0.1)
        # (13, 13, 1280) --> (13, 13, 1024)
        print('layer22.shape', d['leaky_relu22'].get_shape().as_list())

        output_channel = self.num_anchors * (5 + self.num_classes)
        d['logit'] = conv_layer(d['leaky_relu22'], output_channel, (1, 1), 1
                                padding='SAME', use_bias=True)
        d['pred'] = tf.reshape(
            d['logit'], (-1, self.grid_size[0], self.grid_size[1], self.num_anchors, 5 + self.num_classes))
        print('pred.shape', d['pred'].get_shape().as_list())
        # (13, 13, 1024) --> (13, 13, num_anchors , (5 + num_classes))

        return d

    def _build_loss(self, **kwargs):
        """
        Build loss function for the model training.
        :param kwargs: dict, extra arguments
                - loss_weights: list, [xy, wh, resp_confidence, no_resp_confidence, class_probs]
        :return tf.Tensor.
        """

        loss_weights = kwargs.pop('loss_weights', [5, 5, 5, 0.5, 1.0])
        # DEBUG
        # loss_weights = kwargs.pop('loss_weights', [1.0, 1.0, 1.0, 1.0, 1.0])
        grid_h, grid_w = self.grid_size
        num_classes = self.num_classes
        anchors = self.anchors
        grid_wh = np.reshape([grid_w, grid_h], [
                             1, 1, 1, 1, 2]).astype(np.float32)
        cxcy = np.transpose([np.tile(np.arange(grid_w), grid_h),
                             np.repeat(np.arange(grid_h), grid_w)])
        cxcy = np.reshape(cxcy, (1, grid_h, grid_w, 1, 2))

        txty, twth = self.pred[..., 0:2], self.pred[..., 2:4]
        confidence = tf.sigmoid(self.pred[..., 4:5])
        class_probs = tf.nn.softmax(
            self.pred[..., 5:], axis=-1) if num_classes > 1 else tf.sigmoid(self.pred[..., 5:])
        bxby = tf.sigmoid(txty) + cxcy
        pwph = np.reshape(anchors, (1, 1, 1, self.num_anchors, 2)) / 32
        bwbh = tf.exp(twth) * pwph

        # calculating for prediction
        nxny, nwnh = bxby / grid_wh, bwbh / grid_wh
        nx1ny1, nx2ny2 = nxny - 0.5 * nwnh, nxny + 0.5 * nwnh
        self.pred_y = tf.concat(
            (nx1ny1, nx2ny2, confidence, class_probs), axis=-1)

        # calculating IoU for metric
        num_objects = tf.reduce_sum(self.y[..., 4:5], axis=[1, 2, 3, 4])
        max_nx1ny1 = tf.maximum(self.y[..., 0:2], nx1ny1)
        min_nx2ny2 = tf.minimum(self.y[..., 2:4], nx2ny2)
        intersect_wh = tf.maximum(min_nx2ny2 - max_nx1ny1, 0.0)
        intersect_area = tf.reduce_prod(intersect_wh, axis=-1)
        intersect_area = tf.where(
            tf.equal(intersect_area, 0.0), tf.zeros_like(intersect_area), intersect_area)
        gt_box_area = tf.reduce_prod(
            self.y[..., 2:4] - self.y[..., 0:2], axis=-1)
        box_area = tf.reduce_prod(nx2ny2 - nx1ny1, axis=-1)
        iou = tf.truediv(
            intersect_area, (gt_box_area + box_area - intersect_area))
        sum_iou = tf.reduce_sum(iou, axis=[1, 2, 3])
        self.iou = tf.truediv(sum_iou, num_objects)

        gt_bxby = 0.5 * (self.y[..., 0:2] + self.y[..., 2:4]) * grid_wh
        gt_bwbh = (self.y[..., 2:4] - self.y[..., 0:2]) * grid_wh

        resp_mask = self.y[..., 4:5]
        no_resp_mask = 1.0 - resp_mask
        gt_confidence = resp_mask * tf.expand_dims(iou, axis=-1)
        gt_class_probs = self.y[..., 5:]

        loss_bxby = loss_weights[0] * resp_mask * \
            tf.square(gt_bxby - bxby)
        loss_bwbh = loss_weights[1] * resp_mask * \
            tf.square(tf.sqrt(gt_bwbh) - tf.sqrt(bwbh))
        loss_resp_conf = loss_weights[2] * resp_mask * \
            tf.square(gt_confidence - confidence)
        loss_no_resp_conf = loss_weights[3] * no_resp_mask * \
            tf.square(gt_confidence - confidence)
        loss_class_probs = loss_weights[4] * resp_mask * \
            tf.square(gt_class_probs - class_probs)

        merged_loss = tf.concat((
                                loss_bxby,
                                loss_bwbh,
                                loss_resp_conf,
                                loss_no_resp_conf,
                                loss_class_probs
                                ),
                                axis=-1)
        #self.merged_loss = merged_loss
        total_loss = tf.reduce_sum(merged_loss, axis=-1)
        total_loss = tf.reduce_mean(total_loss)
        return total_loss
