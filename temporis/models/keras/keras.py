import tensorflow as tf

from temporis.iterators.iterators import WindowedDatasetIterator
import numpy as np

def tf_regression_dataset(iterator: WindowedDatasetIterator):
    n_features = iterator.n_features

    def generator_function():
        for X, y, sw in iterator:
            yield X, y, sw

    a = tf.data.Dataset.from_generator(
        generator_function,
        output_signature=(
            tf.TensorSpec(
                shape=(iterator.window_size, n_features), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(iterator.output_shape, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(1), dtype=tf.float32),
        ),
    )

    return a


def tf_seq_to_seq_dataset(iterator: WindowedDatasetIterator):
    n_features = iterator.n_features

    def generator_function():
        for X, y, sw in iterator:
            yield X, y, sw

    a = tf.data.Dataset.from_generator(
        generator_function,
        output_signature=(
            tf.TensorSpec(
                shape=(iterator.window_size, n_features), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(iterator.window_size, iterator.output_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(1), dtype=tf.float32),
        ),
    )

    return a


def tf_autoencoder_dataset(iterator: WindowedDatasetIterator):
    n_features = iterator.n_features

    def gen_train():
        for X, y, sw in iterator:
            yield X, X, sw

    a = tf.data.Dataset.from_generator(
        gen_train,
        (tf.float32, tf.float32, tf.float32),
        (
            tf.TensorShape([iterator.window_size, n_features]),
            tf.TensorShape([iterator.window_size, n_features]),
            tf.TensorShape([1]),
        ),
    )

    return a
