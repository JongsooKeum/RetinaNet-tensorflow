import numpy as np
import cv2


def padding(img, bboxes, pad_size):
    orig_size = img.shape[:2]
    assert orig_size[0] <= pad_size[0] and orig_size[1] <= pad_size[1]
    assert np.all(np.array([orig_size, pad_size]) % 2 == 0)
    img = _padding_img(img, pad_size)
    if bboxes is not None:
        bboxes = _padding_bboxes(bboxes, orig_size, pad_size)
    return img, bboxes


def _padding_img(img, pad_size):
    orig_size = img.shape[:2]
    img = np.copy(img)
    pad_w = (pad_size[1] - orig_size[1]) // 2
    pad_h = (pad_size[0] - orig_size[0]) // 2
    img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                 'constant', constant_values=0)
    return img


def _padding_bboxes(bboxes, im_size, pad_size):
    bboxes = np.copy(bboxes)
    pad_w = (pad_size[1] - im_size[1]) // 2
    pad_h = (pad_size[0] - im_size[0]) // 2
    bboxes[:, 0] += pad_w
    bboxes[:, 1] += pad_h
    bboxes[:, 2] += pad_w
    bboxes[:, 3] += pad_h
    return bboxes


def anchor_targets_bbox(
        image_shape,
        annotations,
        num_classes,
        anchors=None,
        mask_shape=None,
        negative_overlap=0.4,
        positive_overlap=0.5,
        **kwargs
):
    if anchors is None:
        anchors = anchors_for_shape(image_shape, **kwargs)
    labels = np.ones((anchors.shape[0], num_classes), dtype=np.float32) * -1

    if annotations.shape[0]:
        overlaps = compute_overlap(anchors, annotations[:, :4])
        argmax_overlaps_inds = np.argmax(overlaps, axis=1)
        max_overlaps = overlaps[
            np.arange(overlaps.shape[0]), argmax_overlaps_inds]
        labels[max_overlaps < negative_overlap, :] = 0
        labels[max_overlaps < negative_overlap, 0] = 1
        annotations = annotations[argmax_overlaps_inds]
        positive_indices = max_overlaps >= positive_overlap
        labels[positive_indices, :] = 0
        labels[positive_indices, annotations[
            positive_indices, 4].astype(int)] = 1
    else:
        labels[:] = 0
        annotations = np.zeros_like(anchors)

    mask_shape = image_shape if mask_shape is None else mask_shape
    anchors_centers = np.vstack(
        [(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3] / 2)]).T
    indices = np.logical_or(anchors_centers[:, 0] >= mask_shape[
                            1], anchors[:, 1] >= mask_shape[0])
    labels[indices, :] = -1

    return labels, annotations


def anchors_for_shape(
        image_shape,
        pyramid_levels=None,
        ratios=None,
        scales=None,
        strides=None,
        sizes=None
):
    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]
        # pyramid_levels = [5]
    if strides is None:
        strides = [2 ** x for x in pyramid_levels]
    if sizes is None:
        sizes = [2 ** (x + 2) for x in pyramid_levels]
    if ratios is None:
        ratios = np.array([0.5, 1, 2])
    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    image_shape = np.array(image_shape[:2])
    for i in range(pyramid_levels[0] - 1):
        image_shape = (image_shape + 1) // 2

    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        image_shape = (image_shape + 1) // 2
        anchors = generate_anchors(
            base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shifts(image_shape, strides[idx], anchors)
        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors.astype(np.float32)


def shifts(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def generate_anchors(base_size=16, ratios=None, scales=None):

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    areas = anchors[:, 2] * anchors[:, 3]

    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def compute_overlap(a, b):

    area = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - \
        np.maximum(np.expand_dims(a[:, 0], axis=1), b[:, 0]) + 1
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - \
        np.maximum(np.expand_dims(a[:, 1], axis=1), b[:, 1]) + 1

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0] + 1) *
                        (a[:, 3] - a[:, 1] + 1), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def bbox_transform(anchors, gt_boxes, mean=None, std=None):
    if mean is None:
        mean = np.array([0, 0, 0, 0])
    elif isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    else:
        raise ValueError

    if std is None:
        std = np.array([0.1, 0.1, 0.2, 0.2])
    elif isinstance(std, (list, tuple)):
        std = np.array(std)
    else:
        raise ValueError

    anchor_widths = anchors[:, 2] - anchors[:, 0] + 1.0
    anchor_heights = anchors[:, 3] - anchors[:, 1] + 1.0
    anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
    anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - anchor_ctr_x) / anchor_widths
    targets_dy = (gt_ctr_y - anchor_ctr_y) / anchor_heights
    targets_dw = np.log(gt_widths / anchor_widths)
    targets_dh = np.log(gt_heights / anchor_heights)

    targets = np.stack((targets_dx, targets_dy, targets_dw, targets_dh))
    targets = targets.T

    targets = (targets - mean) / std

    return targets


def bbox_transform_inv(boxes, deltas, mean=None, std=None):
    if mean is None:
        mean = np.array([0, 0, 0, 0], dtype=np.float32)
    if std is None:
        std = np.array([0.1, 0.1, 0.2, 0.2], dtype=np.float32)

    widths = boxes[:, :, 2] - boxes[:, :, 0]
    heights = boxes[:, :, 3] - boxes[:, :, 1]
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

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


def augment(image, bboxes, augment=[]):

    img = np.copy(image)
    bboxes = np.copy(bboxes)
    ih, iw = img.shape[:2]
    if 'hflip' in augment and np.random.random() < 0.5:
        x1 = np.copy(bboxes[:, 0])
        x2 = np.copy(bboxes[:, 2])

        img = cv2.flip(img, 1)
        bboxes[:, 0] = iw - x2
        bboxes[:, 2] = iw - x1

    if 'vflip' in augment and np.random.random() < 0.5:
        y1 = np.copy(bboxes[:, 1])
        y2 = np.copy(bboxes[:, 3])

        img = cv2.flip(img, 0)
        bboxes[:, 1] = ih - y2
        bboxes[:, 3] = ih - y1

    if 'rotate' in augment:
        angle = np.random.choice([0, 90, 180, 270], 1)[0]
        x1 = np.copy(bboxes[:, 0])
        y1 = np.copy(bboxes[:, 1])
        x2 = np.copy(bboxes[:, 2])
        y2 = np.copy(bboxes[:, 3])

        if angle == 270:
            img = np.transpose(img, (1, 0, 2))
            img = cv2.flip(img, 0)
            bboxes[:, 0] = y1
            bboxes[:, 2] = y2
            bboxes[:, 1] = iw - x2
            bboxes[:, 3] = iw - x1
        elif angle == 180:
            img = cv2.flip(img, -1)
            bboxes[:, 0] = iw - x2
            bboxes[:, 2] = iw - x1
            bboxes[:, 1] = ih - y2
            bboxes[:, 3] = ih - y1
        elif angle == 90:
            img = np.transpose(img, (1, 0, 2))
            img = cv2.flip(img, 1)
            bboxes[:, 0] = ih - y2
            bboxes[:, 2] = ih - y1
            bboxes[:, 1] = x1
            bboxes[:, 3] = x2
        elif angle == 0:
            pass

    return img, bboxes
