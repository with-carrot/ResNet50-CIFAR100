import os
import argparse
import pathlib

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import numpy as np

from preprocessing import *
from load_model import load_model_val

def get_arguments():
    parser = argparse.ArgumentParser(description='Model arguments')

    parser.add_argument('--path', type=str, default='./')
    parser.add_argument('--dataset', type=str, default='CIFAR-100')

    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--compute', type=str, default='FP16')
    parser.add_argument('--starting-epoch', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)

    return parser.parse_args()

def main():
    args = get_arguments()

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[args.gpu], True)

    if args.compute == 'FP16':
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    if args.dataset == 'CIFAR-100':
        from load_dataset import load_dataset_val
        validation_dataset = load_dataset_val(args.dataset, args.batch_size)
    
    acc_obj = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def validate_one_step(x, y):
        outputs = Model(x, training=False)
        acc_obj.update_state(y, outputs)

    def validate(dataset):
        for x, y in dataset:
            validate_one_step(x, y)

        accuracy = acc_obj.result()
        acc_obj.reset_states()

        tf.print('Accuracy: ', accuracy)

    with tf.device('/GPU:' + str(args.gpu)):
        Model = load_model_val(args.path)
        Model.summary()
        
        validate(validation_dataset)

    del Model

if __name__ == '__main__':
    main()
