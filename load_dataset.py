import pathlib
import tensorflow as tf
import numpy as np

from preprocessing import *

def prepare_dataset_builtin(x_train, y_train, image_size, dataset_name):
    num_images = x_train.shape[0]
    indices = np.arange(num_images)
    np.random.shuffle(indices)

    x_train = x_train[indices]
    y_train = y_train[indices]

    scale = (np.random.sample((num_images,)) * 1.5 + 0.5).tolist()
    rotate_prob = np.random.sample((num_images,)).tolist()
    rotate = (np.pi * (np.random.sample((num_images,)) - 0.5) / 9).tolist()
    flip = np.random.sample((num_images,)).tolist()
    crop = np.random.sample((num_images,)).tolist()

    main_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    factors_dataset = tf.data.Dataset.from_tensor_slices(
        (scale, rotate_prob, rotate, flip, crop, np.tile(image_size[:2], (num_images, 1)).tolist()))

    total_dataset = tf.data.Dataset.zip((main_dataset, factors_dataset))

    if dataset_name == 'CIFAR-100':
        total_dataset = total_dataset.map(preprocessing_cifar100_train, tf.data.experimental.AUTOTUNE)

    return total_dataset

def load_dataset(dataset_name):
    if dataset_name == 'CIFAR-100':
        (x_train, y_train), _ = tf.keras.datasets.cifar100.load_data()
        total_num = x_train.shape[0]
        
        return x_train, y_train, total_num
    else:
        pass

def load_dataset_val(dataset_name, batch_size):
    if dataset_name == 'CIFAR-100':
        _, (x_val, y_val) = tf.keras.datasets.cifar100.load_data()
        total_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        total_dataset = total_dataset.map(preprocessing_cifar100_val, tf.data.experimental.AUTOTUNE)

    return total_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
