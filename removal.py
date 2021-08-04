from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import Model
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def Generator():
    inputs = Input(shape=[256, 256, 3])
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv5)
    drop5 = Dropout(0.5)(conv5)


    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv9)
    conv9 = Conv2D(2*6, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv9)
    conv10 = Conv2D(1*3, 1, activation = 'tanh')(conv9)

    return Model(inputs=inputs, outputs=conv10)


class WatermarkRemoval:
    def __init__(self):
        generator_optimizer = tf.keras.optimizers.Adam(2e-4)
        self.generator = Generator()

        checkpoint_dir = './models/removal/'
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                        generator=self.generator)

        # restoring the latest checkpoint in checkpoint_dir
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def generate_image(self, inp, h, w):
        prediction = self.generator(inp, training=False)
        prediction = prediction[0]
        prediction = tf.image.resize(prediction, [h, w],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        prediction = (prediction+1) * 127.5
        return prediction.numpy()


def preprocess(input_image):
    input_image = tf.image.resize(input_image, [256, 256],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    input_image = (input_image / 127.5) - 1
    input_image = tf.expand_dims(input_image, 0)

    return input_image
