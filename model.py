import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Add, Activation, MaxPool2D, GlobalAveragePooling2D, Dense, BatchNormalization, Dropout, Flatten, Permute
import kapre
from kapre.composed import get_melspectrogram_layer


def build_model(params):
    '''
      build_model(params)
        Creates the keras model and returns it.
        Called by do_learning()
    '''
    AUDIO_DURATION = params['AUDIO_DURATION']
    SAMPLING_RATE = params['SAMPLING_RATE']
    N_FFT = params['N_FFT']
    HOP_LENGTH = params['HOP_LENGTH']
    N_MELS = params['N_MELS']
    N_CLASSES = params['N_CLASSES']

    input_shape = (AUDIO_DURATION*SAMPLING_RATE, 1)

    inputs = Input(shape=input_shape, name="input")
    x = get_melspectrogram_layer(n_fft=1764, hop_length=882, return_decibel=True, n_mels=N_MELS)(inputs)

    x = Permute((2,1,3))(x)
    x = Conv2D(16, (7,7), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(16, (7,7), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPool2D(pool_size=(5,5))(x)
    x = Dropout(.3)(x)

    x = Conv2D(32, (7,7), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPool2D(pool_size=(4,20))(x)
    x = Dropout(.3)(x)

    x = Flatten()(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(.3)(x)

    outputs = Dense(N_CLASSES, activation="softmax")(x)


    return tf.keras.models.Model(inputs=inputs, outputs=outputs)
