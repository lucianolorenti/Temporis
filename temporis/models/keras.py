import tensorflow as tf 

def keras_regression_batcher(batcher):
    n_features = batcher.n_features

    def generator_function():
        for X, y, w in batcher:
            yield X, y, w

    a = tf.data.Dataset.from_generator(
        generator_function,
        output_signature=(
            tf.TensorSpec(
                shape=(None, batcher.window_size, n_features), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(None, batcher.output_shape, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        ),
    )

    if batcher.prefetch_size is not None:
        a = a.prefetch(batcher.batch_size * 2)
    return a


def keras_autoencoder_batcher(batcher):
    n_features = batcher.n_features

    def gen_train():
        for X, w in batcher:
            yield X, X, w

    a = tf.data.Dataset.from_generator(
        gen_train,
        (tf.float32, tf.float32, tf.float32),
        (
            tf.TensorShape([None, batcher.window_size, n_features]),
            tf.TensorShape([None, batcher.window_size, n_features]),
            tf.TensorShape([None, 1]),
        ),
    )

    if batcher.prefetch_size is not None:
        a = a.prefetch(batcher.batch_size * 2)

    return a
