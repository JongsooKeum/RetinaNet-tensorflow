import tensorflow as tf

def focal_loss(y_pred, y_true, alpha=0.25, gamma=2.0):
    total_state = tf.reduce_max(y_true, axis=-1)
    total_state = tf.cast(tf.math.equal(total_state, 1), dtype=tf.float32)
    pos_state = tf.reduce_max(y_true[..., 1:], axis=-1)
    pos_state = tf.cast(tf.math.equal(pos_state, 1), dtype=tf.float32)
    divisor = tf.reduce_sum(pos_state)
    divisor = tf.clip_by_value(divisor, 1, divisor)

    labels = tf.multiply(total_state, y_true)
    class_logits = tf.multiply(total_state, y_pred)
    class_probs = tf.nn.sigmoid(class_logits)
    focal_weight = alpha * tf.pow((1-class_probs), gamma)
    mask_focal_weight = tf.multiply(labels, focal_weight)
    mask_focal_weight = tf.reduce_max(mask_focal_weight, axis=-1)
    focal_loss = mask_focal_weight * \
                tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=class_logits)
    focal_loss = tf.reduce_sum((focal_loss / divisor))
    return focal_loss

def smmoth_l1_loss(y_pred, y_true, sigma=3.0):
    sigma2 = sigma * sigma

    regression = y_pred
    regression_target = y_true[:, :4]
    anchor_state = y_true[:, 4]
    pos_state = tf.math.equal(anchor_state, 1)
    divisor = tf.reduce_sum(pos_state)
    divisor = tf.clip_by_value(divisor, 1, divisor)

    abs_loss = tf.abs(tf.multiply(pos_state, (regression-regression_target)))

    smooth_l1_sign = tf.less(abs_loss, 1.0/sigma2)
    smooth_l1_option1 = tf.multiply(tf.pow(abs_loss, 2), 0.5*sigma2)
    smooth_l1_option2 = abs_loss - (0.5/sigma2)
    smooth_l1_results = tf.multiply(smooth_l1_option1, smooth_l1_sign) + \
                        tf.multiply(smooth_l1_option2, (1 - smooth_l1_sign))
    smooth_l1_results = tf.reduce_sum((smooth_l1_results / divisor))
    return smooth_l1_resultsfg

