import tensorflow as tf

def load_model(args, image_size, num_classes, training=True):
    Model_D = None

    if training:
        temperature = args.temperature
        temperature_d = args.temperature_d
        type = args.type
    else:
        temperature = 1
        temperature_d = 1
        type = 'Single'

    inputs = tf.keras.Input(shape=image_size)

    if args.model == 'ResNet50':
        resnet = tf.keras.applications.resnet50.ResNet50(input_tensor=inputs,\
            include_top=False, pooling='avg')
        resnet.trainable = True
        out = tf.keras.layers.Dense(100, activation='softmax')(resnet.output)

        Model = tf.keras.Model(inputs=resnet.input, outputs=out)

    return [Model_D, Model]

def load_model_val(path):
    return tf.keras.models.load_model(path, compile=False)
