import numpy as np
import cv2
from utils.colors import COLORS

def draw_pred_boxes(image, pred_boxes, class_map, text=True, score=False):
    im_h, im_w = image.shape[:2]
    output = image.copy()

    for box in pred_boxes:
        overlay = output.copy()
        class_idx = np.argmax(box[5:])
        color = COLORS[class_idx]
        line_width, alpha = (2, 0.8)
        x_min, x_max = [int(x) for x in [box[0], box[2]]]
        y_min, y_max = [int(x) for x in [box[1], box[3]]]
        cv2.rectangle(overlay, (x_min, y_min),
                      (x_max, y_max), color, line_width)
        output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)

    return output