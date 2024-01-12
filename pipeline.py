from config import (BATCH_SIZE, HEIGHT, WIDTH, NUM_CHANNELS)
import tensorflow as tf


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=NUM_CHANNELS)
    image = tf.image.resize(image, [HEIGHT, WIDTH])
    image /= 255.0

    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def input_pipeline(files, categories):
    path_ds = tf.data.Dataset.from_tensor_slices(files)
    image_ds = path_ds.map(load_and_preprocess_image,
                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(categories, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    ds = image_label_ds.shuffle(buffer_size=1000 * BATCH_SIZE)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    # `prefetch` lets the dataset fetch batches, in the background while the model is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds
