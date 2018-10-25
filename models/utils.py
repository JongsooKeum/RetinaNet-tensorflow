import tensorflow as tf
import numpy as np

def focal_loss(y_pred, y_true, alpha=0.25, gamma=2.0):
    y_true, y_pred = [x[:, :, 4:] for x in [y_true, y_pred]]

    total_state = tf.reduce_max(y_true, axis=-1, keepdims=True)
    total_state = tf.cast(tf.math.equal(total_state, 1), dtype=tf.float32)
    pos_state = tf.reduce_max(y_true[..., 1:], axis=-1, keepdims=True)
    pos_state = tf.cast(tf.math.equal(pos_state, 1), dtype=tf.float32)
    divisor = tf.reduce_sum(pos_state)
    divisor = tf.clip_by_value(divisor, 1, divisor)

    labels = tf.multiply(total_state, y_true)
    class_logits = tf.multiply(total_state, y_pred)
    class_probs = tf.nn.sigmoid(class_logits)
    focal_weight = alpha * tf.pow((1-class_probs), gamma)
    mask_focal_weight = tf.multiply(labels, focal_weight)
    mask_focal_weight = tf.reduce_max(mask_focal_weight, axis=-1, keepdims=True)
    focal_loss = mask_focal_weight * \
                tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=class_logits)
    focal_loss = tf.reduce_sum((focal_loss / divisor))
    return focal_loss

def smooth_l1_loss(y_pred, y_true, sigma=3.0):
    sigma2 = sigma * sigma
    anchor_state = tf.reduce_max(y_true[:, :, 5:], axis=-1, keepdims=True)
    y_true, y_pred = [x[:, :, :4] for x in [y_true, y_pred]]

    regression = y_pred
    regression_target = y_true[:, :, :4]
    pos_state = tf.cast(tf.math.equal(anchor_state, 1), dtype=tf.float32)
    divisor = tf.reduce_sum(pos_state)
    divisor = tf.clip_by_value(divisor, 1, divisor)

    abs_loss = tf.abs(tf.multiply(pos_state, (regression-regression_target)))

    smooth_l1_sign = tf.cast(tf.less(abs_loss, 1.0/sigma2), dtype=tf.float32)
    smooth_l1_option1 = tf.multiply(tf.pow(abs_loss, 2), 0.5*sigma2)
    smooth_l1_option2 = abs_loss - (0.5/sigma2)
    smooth_l1_results = tf.multiply(smooth_l1_option1, smooth_l1_sign) + \
                        tf.multiply(smooth_l1_option2, (1 - smooth_l1_sign))
    smooth_l1_results = tf.reduce_sum((smooth_l1_results / divisor))
    return smooth_l1_results

def generate_anchors(base_size, ratios=None, scales=None):
    if ratios is None:
        ratios = np.array([0.5, 1, 2], dtype=np.float32)
    elif isinstance(ratios, list):
        ratios = np.array(ratios, dtype=np.float32)
    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 **
                           (2.0 / 3.0)], dtype=np.float32)
    elif isinstance(scales, list):
        scales = np.array(scales, dtype=np.float32)

    num_anchors = len(ratios) * len(scales)

    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
    areas = anchors[:, 2] * anchors[:, 3]

    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def shifts(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).T

    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return np.array(all_anchors, dtype=np.float32)


def bbox_transform_inv(boxes, deltas, mean=None, std=None):
    if mean is None:
        mean = np.array([0, 0, 0, 0], dtype=np.float32)
    if std is None:
        std = np.array([0.1, 0.1, 0.2, 0.2], dtype=np.float32)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, :, 0] * std[0] + mean[0]
    dy = deltas[:, :, 1] * std[1] + mean[1]
    dw = deltas[:, :, 2] * std[2] + mean[2]
    dh = deltas[:, :, 3] * std[3] + mean[3]

    pred_ctr_x = ctr_x + dx * widths
    pred_ctr_y = ctr_y + dy * heights
    pred_w = np.exp(dw) * widths
    pred_h = np.exp(dh) * heights

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)

    pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
    pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
    pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
    pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

    pred_boxes = np.stack([pred_boxes_x1, pred_boxes_y1,
                           pred_boxes_x2, pred_boxes_y2], axis=2)

    return pred_boxes
