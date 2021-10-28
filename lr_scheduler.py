import tensorflow as tf
import numpy as np

class StairLrScheduler(tf.keras.callbacks.Callback):
    def __init__(self, init_lr, decay_epochs, decay_rate):
        super(StairLrScheduler, self).__init__()

        self.init_lr = init_lr
        self.current_lr = init_lr
        self.decay_epochs = decay_epochs
        self.decay_rate = decay_rate

    def on_epoch_begin(self, epoch, logs=None):
        self.current_lr = self.init_lr * np.power(self.decay_rate, int(epoch / self.decay_epochs))
        tf.keras.backend.set_value(self.model.optimizer.lr, self.current_lr)

        print('Start learning with lr=' + str(self.current_lr))

def cosine_with_warmup_scheduler(init_lr, batch_in_epoch, batch_per_epoch, total_batch, current_epoch, warmup_epoch):
    current_batch = batch_in_epoch + (current_epoch - 1) * batch_per_epoch

    if current_epoch <= warmup_epoch:
        cosine_lr = init_lr * current_batch / (warmup_epoch * batch_per_epoch)
    else:
        current_batch = current_batch - warmup_epoch * batch_per_epoch
        cosine_lr = 0.5 * (1 + np.cos(current_batch * np.pi / total_batch)) * init_lr

    return cosine_lr

class CosineWithWarmupLrScheduler(tf.keras.callbacks.Callback):
    def __init__(self, init_lr, batch_per_epoch, total_epoch, warmup_epoch):
        super(CosineWithWarmupLrScheduler, self).__init__()

        self.init_lr = init_lr
        self.current_lr = 0
        self.array_lr = []

        self.current_epoch = 1
        self.warmup_epoch = warmup_epoch

        self.batch_per_epoch = batch_per_epoch
        self.total_batch = (total_epoch - warmup_epoch) * batch_per_epoch + 1

    def on_batch_begin(self, batch, logs=None):
        self.current_lr = cosine_with_warmup_scheduler(self.init_lr, batch + 1, self.batch_per_epoch, self.total_batch,
                                                       self.current_epoch, self.warmup_epoch)

        tf.keras.backend.set_value(self.model.optimizer.lr, self.current_lr)
        self.array_lr.append(self.current_lr)

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        self.array_lr = []

        if epoch > 1:
            self.current_lr = cosine_with_warmup_scheduler(self.init_lr, 0, self.batch_per_epoch, self.total_batch,
                                                           self.current_epoch, self.warmup_epoch)

        tf.keras.backend.set_value(self.model.optimizer.lr, self.current_lr)

        print('Start learning with lr=' + str(self.current_lr))

def Mutual_Stair_Lr(epoch, base_lr, stair_rate, stair_epoch):
    return base_lr * np.power(stair_rate, int(epoch / stair_epoch))

def Mutual_Cosine_Lr(steps, base_lr, base_steps, warmup_epoch, total_epoch):
    warmup_steps = base_steps * warmup_epoch

    if steps <= warmup_steps:
        return base_lr * steps / warmup_steps
    else:
        net_steps = base_steps * (total_epoch - warmup_epoch) + 1
        current_steps = steps - warmup_steps

        return 0.5 * (1 + np.cos(current_steps * np.pi / net_steps)) * base_lr

def lr_update_by_policy(lr, current_epoch, current_step, steps_per_epoch, total_epoch, option_epoch, stair_rate,
                        policy=None):
    if policy == 'Stair':
        return Mutual_Stair_Lr(current_epoch, lr, stair_rate, option_epoch)
    elif policy == 'Cosine':
        return Mutual_Cosine_Lr(current_step, lr, steps_per_epoch, option_epoch, total_epoch)
    else:
        return lr