import os

import tensorflow as tf
import tensorflow_addons as tfa

RGB_mean = [123.68, 116.779, 103.939]

def image_RGB_subtraction(image):
    channels = tf.split(image, 3, 2)

    for i in range(3):
        channels[i] -= RGB_mean[i]

    return tf.concat([channels[2], channels[1], channels[0]], 2)

def image_scaling(image, scale):
    new_size = tf.cast(scale * tf.cast(tf.shape(image)[:2], tf.float32), tf.int32)
    
    return tf.image.resize(image, new_size)

def image_rotate(image, rotate_prob, angle):
    if rotate_prob >= 0.5:
        return tfa.image.rotate(image, angle)
    else:
        return image

def image_flip(image, flip_prob):
    if flip_prob >= 0.5:
        return tf.image.flip_left_right(image)
    else:
        return image

def image_crop(image, coordination, size):
    input_size = tf.shape(image)[:2]

    if input_size[0] < size[0] or input_size[1] < size[1]:
        return tf.image.resize_with_pad(image, size[0], size[1], 'bilinear')
    else:
        crop_location = tf.cast(tf.math.multiply(coordination, tf.cast(input_size - size, tf.float32)), tf.int32)

        return tf.image.crop_to_bounding_box(image, crop_location[0], crop_location[1], size[0], size[1])

def augmentation(image, factors):
    image = tf.image.resize(image, [224, 224])
    image = image_RGB_subtraction(tf.cast(image, tf.float32))
    image = image_scaling(image, factors[0])
    image = image_rotate(image, factors[1], factors[2])
    image = image_flip(image, factors[3])
    image = image_crop(image, factors[4], factors[5])

    return image

def preprocessing_cifar100_train(data, factors):
    image = data[0]
    label = data[1]

    image = augmentation(image, factors)

    return image, label

def preprocessing_cifar100_val(image, label):
    image = tf.image.resize(image, [224, 224])
    image = image_RGB_subtraction(tf.cast(image, tf.float32))

    return image, label
