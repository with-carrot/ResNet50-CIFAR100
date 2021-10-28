import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import shutil
import argparse

import numpy as np
import tensorflow as tf

from preprocessing import *
from lr_scheduler import *
from load_dataset import *

def get_arguments():
    parser = argparse.ArgumentParser(description='Model arguments')

    parser.add_argument('--workspace', type=str, default='./')
    parser.add_argument('--folder', type=str, default='')

    parser.add_argument('--model', type=str, default='ResNet50')
    parser.add_argument('--dataset', type=str, default='CIFAR-100')
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--learning-rate', type=float, default=1e-2)
    parser.add_argument('--learning-rate-d', type=float, default=1e-4)
    parser.add_argument('--learning-rate-policy', type=str, default='Stair')
    parser.add_argument('--learning-rate-policy-d', type=str, default='Stair')
    parser.add_argument('--l2-decay', type=float, default=1e-4)

    parser.add_argument('--total-epoch', type=int, default=200)
    parser.add_argument('--option-epoch', type=int, default=5)
    parser.add_argument('--stair-rate', type=float, default=.1)

    parser.add_argument('--type', type=str, default='Single')
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--temperature-d', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=0.1)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--compute', type=str, default='FP16')
    
    parser.add_argument('--record-steps', type=int, default=10)
    parser.add_argument('--record-epochs', type=int, default=5)
    parser.add_argument('--continue-epoch', type=int, default=0)    

    return parser.parse_args()

def main():
    args = get_arguments()

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[args.gpu], True)

    if args.compute == 'FP16':
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    result_path = os.path.join(args.workspace, args.folder)    
    model_folder = os.path.join(result_path, args.model)

    if args.continue_epoch == 0:
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if os.path.exists(model_folder):
            shutil.rmtree(model_folder)

        os.makedirs(model_folder)

    if args.dataset == 'CIFAR-100':
        x_origin, y_origin, total_num = load_dataset(args.dataset)
        image_size = [224, 224, 3]
        num_classes = 100

    steps_per_epoch = int(np.ceil(total_num / args.batch_size))

    lr = args.learning_rate
    lr_d = args.learning_rate_d
    step = args.continue_epoch * steps_per_epoch + 1

    with tf.device('/GPU:' + str(args.gpu)):
        from load_model import load_model
        Discriminator, Model = load_model(args, image_size, num_classes)
        
        if args.optimizer == 'SGD':
            Opt = tf.keras.optimizers.SGD(lr, 0.9)
        elif args.optimizer == 'Adam':
            Opt = tf.keras.optimizers.Adam(lr, 0.9)

        if not args.type == 'Single':
            Opt_d = tf.keras.optimizers.Adam(lr_d, 0.5)

        if args.compute == 'FP16':
            Opt = tf.keras.mixed_precision.LossScaleOptimizer(Opt)

            if not args.type == 'Single':
                Opt_d = tf.keras.mixed_precision.LossScaleOptimizer(Opt_d)

    model_searched = False

    if args.type == 'Single':
        discriminator_searched = True
    else:
        discriminator_searched = False

    if args.continue_epoch > 0:
        for file in os.listdir(model_folder):
            if file.startswith('model-epoch-' + str(args.continue_epoch)) and model_searched == False:
                pass

            if not args.type == 'Single':
                pass

            if model_searched and discriminator_searched:
                break
    else:
        pass

    Model.summary()

    if args.type == 'Single-Distillation':
        Discriminator.summary()

    loss_mc_object = tf.keras.losses.SparseCategoricalCrossentropy()

    @tf.function
    def train_discriminator(x, y):
        with tf.GradientTape() as tape:
            mask = tf.cast(tf.less(y, num_classes), tf.float32)
            y_correction = tf.multiply(y, tf.cast(mask, tf.int32))

            outputs = Model(x, training=False)

            if args.model == 'PSPNet_ResNet50':
                pred = outputs[1]
                pred_d = outputs[2]
            else:
                pred = outputs[0]
                pred_d = outputs[1]

            y_d = tf.one_hot(tf.squeeze(y_correction), num_classes) * mask

            d_g = Discriminator(pred, training=True)
            d_real = Discriminator(y_d, training=True)
            d_fake = Discriminator(pred_d, training=True)

            # loss_g = 0.5 * tf.reduce_min((d_g*mask-1)**2)
            # loss_d = 0.5 * (tf.reduce_max((d_fake*mask)**2)+tf.reduce_min((d_real*mask-1)**2))

            loss_g = 0.5 * d_loss_select((d_g*mask-1)**2, args.G)
            loss_d = 0.5 * (d_loss_select((d_fake*mask)**2, args.Df) + d_loss_select((d_real*mask-1)**2, args.Dr))

            if args.compute == 'FP16':
                scaled_loss_d = Opt_d.get_scaled_loss(loss_d)

        if not tf.math.is_nan(loss_g):
            if args.compute == 'FP16':
                scaled_gradients_d = tape.gradient(scaled_loss_d, Discriminator.trainable_variables)
                gradients_d = Opt_d.get_unscaled_gradients(scaled_gradients_d)
            else:
                gradients_d = tape.gradient(loss_d, Discriminator.trainable_variables)

            Opt_d.apply_gradients(zip(gradients_d, Discriminator.trainable_variables))
        else:
            loss_g = 0.0
            loss_d = 0.0

        return [loss_g, loss_d]

    @tf.function
    def train_generator(x, y, loss_g):
        with tf.GradientTape(persistent=True) as tape:
            l2_list = [v for v in Model.trainable_variables if 'kernel' in v.name]
            loss = args.l2_decay * tf.add_n(tf.nn.l2_loss(v) for v in l2_list)
            # loss = 0

            mask = tf.cast(tf.less(y, num_classes), tf.float32)
            y_correction = tf.multiply(y, tf.cast(mask, tf.int32))

            outputs = Model(x, training=True)

            if args.model == 'PSPNet_ResNet50':
                pred = outputs[1]
                loss += 0.4 * loss_mc_object(y_correction, outputs[0], sample_weight=tf.squeeze(mask))
            else:
                pred = outputs[0]

            loss += loss_mc_object(y_correction, pred, sample_weight=tf.squeeze(mask)) + args.alpha*loss_g

            if args.compute == 'FP16':
                scaled_loss = Opt.get_scaled_loss(loss)

        if args.model == 'PSPNet_ResNet50':
            fc_list = ['pyramid_conv_1', 'pyramid_conv_2', 'pyramid_conv_3', 'pyramid_conv_6', 'logits']
            fc_trainable = [v for v in Model.trainable_variables if v.name.split('/')[0] in fc_list]
            conv_trainable = [v for v in Model.trainable_variables if v.name.split('/')[0] not in fc_list]
        elif args.model == 'FCN_VGG16':
            w_trainable = [v for v in Model.trainable_variables if 'kernel' in v.name]
            b_trainable = [v for v in Model.trainable_variables if 'bias' in v.name]
        elif args.model == 'ResNet50':
            trainable = [v for v in Model.trainable_variables]

        if args.compute == 'FP16':
            if args.model == 'PSPNet_ResNet50':
                scaled_gradients_main = tape.gradient(scaled_loss, conv_trainable)
                scaled_gradients_aux = tape.gradient(scaled_loss, fc_trainable)

                gradients_aux = Opt_fc.get_unscaled_gradients(scaled_gradients_aux)
            elif args.model == 'FCN_VGG16':
                scaled_gradients_main = tape.gradient(scaled_loss, w_trainable)
                scaled_gradients_aux = tape.gradient(scaled_loss, b_trainable)

                gradients_aux = Opt_bias.get_unscaled_gradients(scaled_gradients_aux)
            elif args.model == 'ResNet50':
                scaled_gradients_main = tape.gradient(scaled_loss, trainable)

            gradients_main = Opt.get_unscaled_gradients(scaled_gradients_main)
        else:
            if args.model == 'PSPNet_ResNet50':
                gradients_main = tape.gradient(loss, conv_trainable)
                gradients_aux = tape.gradient(loss, fc_trainable)
            elif args.model == 'FCN_VGG16':
                gradients_main = tape.gradient(loss, w_trainable)
                gradients_aux = tape.gradient(loss, b_trainable)
            elif args.model == 'ResNet50':
                gradients_main = tape.gradient(loss, trainable)

        if args.model == 'PSPNet_ResNet50':
            Opt.apply_gradients(zip(gradients_main, conv_trainable))
            Opt_fc.apply_gradients(zip(gradients_aux, fc_trainable))
        elif args.model == 'FCN_VGG16':
            Opt.apply_gradients(zip(gradients_main, w_trainable))
            Opt_bias.apply_gradients(zip(gradients_aux, b_trainable))
        elif args.model == 'ResNet50':
            Opt.apply_gradients(zip(gradients_main, trainable))

        del tape

        return loss

    def train_one_step_d(x, y, type):
        losses = train_discriminator(x, y)
        loss_main = train_generator(x, y, losses[0])

        return losses, loss_main

    @tf.function
    def train_one_step(x, y):
        with tf.GradientTape(persistent=True) as tape:
            l2_list = [v for v in Model.trainable_variables if '_conv' in v.name]
            loss = args.l2_decay * tf.add_n(tf.nn.l2_loss(v) for v in l2_list)

            outputs = Model(x, training=True)

            if args.compute == 'FP16':
                loss += loss_mc_object(y, tf.cast(outputs, tf.float32))
                scaled_loss = Opt.get_scaled_loss(loss)
            else:
                loss += loss_mc_object(y, outputs)

            tape.stop_recording()

        trainable = [v for v in Model.trainable_variables]

        if args.compute == 'FP16':
            scaled_gradients = tape.gradient(scaled_loss, trainable)
            gradients = Opt.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(loss, trainable)

        del tape

        Opt.apply_gradients(zip(gradients, trainable))

        return loss

    def train(dataset, epoch, init_step):
        print('** Training **')

        for x, y in dataset:
            Opt.learning_rate = lr_update_by_policy(lr, epoch, init_step, steps_per_epoch, args.total_epoch,
                                                    args.option_epoch, args.stair_rate, args.learning_rate_policy)

            if 'Distillation' in args.type:
                Opt_d.learning_rate = lr_update_by_policy(lr_d, epoch, init_step, steps_per_epoch, args.total_epoch,
                                                          args.option_epoch, args.stair_rate, args.learning_rate_policy_d)

                losses_d, loss = train_one_step_d(x, y, args.type)
            else:
                loss = train_one_step(x, y)

            if init_step % args.record_steps == 0 or init_step == steps_per_epoch * (epoch+1):
                tf.print()
                tf.print('Step ', init_step)

                if 'Distillation' in args.type:
                    tf.print('L_(Seg+G): ', loss, ' @lr=', Opt.learning_rate, ', alpha=', args.alpha)
                    tf.print('L_G & L_D: ', losses_d, ' @lr=', Opt_d.learning_rate)
                    # tf.print('Map: ', map)
                else:
                    tf.print('Loss: ', loss, ' @lr=', Opt.learning_rate)

            init_step += 1

        return init_step

    def save_model(epoch):
        Model.save(os.path.join(model_folder, 'model-epoch-' + str(epoch)))

        if args.type == 'Single-Distillation':
            Discriminator.save_weights(os.path.join(model_folder, 'discriminator-epoch-' + str(epoch) + '.h5'))

        print('\nModel has been saved!')

    for epoch in range(args.continue_epoch, args.total_epoch):
        print('\n*** Epoch ' + str(epoch+1) + ' ***\n')

        dataset = prepare_dataset_builtin(x_origin, y_origin, image_size, args.dataset).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        step = train(dataset, epoch, step)

        if (epoch+1) % 5 == 0 or epoch >= args.total_epoch - 10:
            save_model(epoch+1)

if __name__ == '__main__':
    main()
