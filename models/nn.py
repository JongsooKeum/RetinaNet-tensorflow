import time
import numpy as np
import tensorflow as tf
from abc import abstractmethod, ABCMeta

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
        self.pred = self.d['pred']
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

    def predict(self, sess, dataset, verbose=False, **kwargs):
        """
        Make predictions for the given dataset.
        :param sess: tf.Session.
        :param dataset: DataSet.
        :param verbose: bool, whether to print details during prediction.
        :param kwargs: dict, extra arguments for prediction.
                -batch_size: int, batch size for each iteration.
        :return _y_pred: np.ndarray, shape: shape of self.pred
        """

        batch_size = kwargs.pop('batch_size', 16)

        num_classes = self.num_classes
        pred_size = dataset.num_examples
        num_steps = pred_size // batch_size
        flag = int(bool(pred_size % batch_size))
        if verbose:
            print('Running prediction loop...')

        # Start prediction loop
        _y_pred = []
        start_time = time.time()
        for i in range(num_steps + flag):
            if i == num_steps and flag:
                _batch_size = pred_size - num_steps * batch_size
            else:
                _batch_size = batch_size
            X, _ = dataset.next_batch(_batch_size, shuffle=False)

            # Compute predictions
            y_pred = sess.run(self.pred_y, feed_dict={
                              self.X: X, self.is_train: False})

            _y_pred.append(y_pred)

        if verbose:
            print('Total prediction time(sec): {}'.format(
                time.time() - start_time))

        _y_pred = np.concatenate(_y_pred, axis=0)
        return _y_pred
