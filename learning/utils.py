import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(exp_idx, step_losses, step_scores, eval_scores=None,
                        mode='max', img_dir='.'):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].plot(np.arange(1, len(step_losses)+1), step_losses, marker='')
    axes[0].set_ylabel('loss')
    axes[0].set_xlabel('Number of iterations')
    axes[1].plot(np.arange(1, len(step_scores)+1), step_scores, color='b', marker='')
    if eval_scores is not None:
        axes[1].plot(np.arange(1, len(eval_scores)+1), eval_scores, color='r', marker='')
    if mode == 'max':
        axes[1].set_ylim(0.5, 1.0)
    else:    # mode == 'min'
        axes[1].set_ylim(0.0, 0.5)
    axes[1].set_ylabel('Error rate')
    axes[1].set_xlabel('Number of epochs')

    # Save plot as image file
    plot_img_filename = 'learning_curve-result{}.svg'.format(exp_idx)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    fig.savefig(os.path.join(img_dir, plot_img_filename))

    # Save details as pkl file
    pkl_filename = 'learning_curve-result{}.pkl'.format(exp_idx)
    with open(os.path.join(img_dir, pkl_filename), 'wb') as fo:
        pkl.dump([step_losses, step_scores, eval_scores], fo)
    plt.close()

def get_boxes(boxes, anchors, top_k_num=100, iou_thres=0.5, conf_thres=0.5, gt=True):
    pred_y = boxes
    regressions = pred_y[:, :, :4]
    regressions = bbox_transform_inv(anchors, regressions)
    confs = pred_y[:, :, 5:] if gt else pred_y[:, :, 4:]

    if top_k_num:
        tmp_confs = []
        tmp_regressions = []
        for conf, regression in zip(confs, regressions):
            scores = np.max(conf[:, 1:], axis=-1)
            inds = top_k(scores, top_k_num)
            tmp_confs.append(conf[inds])
            tmp_regressions.append(regression[inds])
        confs = np.array(tmp_confs, dtype=np.float32)
        regressions = np.array(tmp_regressions, dtype=np.float32)
    if iou_thres:
        tmp_confs = []
        tmp_regressions = []
        for conf, regression in zip(confs, regressions):
            scores = np.max(conf[:, 1:], axis=-
                            1).reshape(regression.shape[0], 1)
            bboxes = np.append(regression, scores, axis=-1)
            conf_inds = np.where(scores < conf_thres)[0]
            nms_inds = cpu_nms(bboxes, iou_thres)
            inds = list(set(nms_inds) - set(conf_inds))
            tmp_conf = conf[inds]
            zero_pad_num = conf.shape[0] - len(inds)
            zero_conf = np.zeros(
                [zero_pad_num, conf.shape[1]], dtype=np.float32)
            tmp_confs.append(np.append(tmp_conf, zero_conf, axis=0))
            tmp_regress = regression[inds]
            zero_regress = np.zeros(
                [zero_pad_num, regression.shape[1]], dtype=np.float32)
            tmp_regressions.append(
                np.append(tmp_regress, zero_regress, axis=0))
        confs = np.array(tmp_confs, dtype=np.float32)
        regressions = np.array(tmp_regressions, dtype=np.float32)

    pred_y = np.append(regressions, confs, axis=-1)
    return pred_y

def cal_recall(gt_bboxes, bboxes, iou_thres=0.5):
    p = 0
    tp = 0
    for idx, (gt, bbox) in enumerate(zip(gt_bboxes, bboxes)):
        gt = gt[np.nonzero(np.any(gt > 0, axis=1))]
        bbox = bbox[np.nonzero(np.any(bbox > 0, axis=1))]
        p += len(gt)
        if bbox.size == 0:
            continue
        iou = _cal_overlap(gt, bbox)
        predicted_class = np.argmax(bbox[...,5:], axis=-1)
        for g, area in zip(gt, iou):
            gt_c = np.argmax(g[5:])
            idx = np.argmax(area)
            if np.max(area) > iou_thres and predicted_class[idx] == gt_c:
                tp += 1
    return tp / p

def _cal_overlap(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - \
        np.maximum(np.expand_dims(a[:, 0], axis=1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - \
        np.maximum(np.expand_dims(a[:, 1], axis=1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)
    intersection = iw * ih

    ua = np.expand_dims((a[:, 2] - a[:, 0]) *
                        (a[:, 3] - a[:, 1]), axis=1) + area - intersection

    ua = np.maximum(ua, np.finfo(float).eps)

    return intersection / ua

def cpu_nms(boxes, iou_thres=0.5):
    x1 = boxes[..., 0]
    y1 = boxes[..., 1]
    x2 = boxes[..., 2]
    y2 = boxes[..., 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = boxes[..., 4]

    keep = []
    order = scores.argsort()[::-1]

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]

    return keep

def top_k(scores, max_num):
    order = scores.argsort()[::-1]
    inds = order[:max_num]
    return inds
