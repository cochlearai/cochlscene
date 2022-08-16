# # # # # # # # # # # # # # # # # # # # # # #
# Baseline code for CochlScene dataset      #
# (APSIPA 2022)                             #
# Author: Jeongsoo Park (jspark@cochl.ai)   #
# Copyright (C) 2022 Cochl, Inc.            #
# All rights reserved.                      #
# # # # # # # # # # # # # # # # # # # # # # #

import tensorflow as tf
import numpy as np

from params import *
from prepare_data import *
from model import *
from eval import *


def do_learning(params):
    '''
        do_learning(params)
            Manages entire process of training
            from the data parsing, model generation, 
            and the actual training.
            Called by main() function.
            Returns training history.
    '''
    DATASET_DIR = params['DATASET_DIR']
    BATCH_SIZE = params['BATCH_SIZE']
    AUDIO_DURATION = params['AUDIO_DURATION']
    SAMPLING_RATE = params['SAMPLING_RATE']
    MODEL_FILENAME = params['MODEL_FILENAME']
    EPOCHS = params['EPOCHS']

    # # # Training data
    train_wav_list, train_label_list = parse_data_list(params, "Train")
    train_ds = generate_dataset(params, train_wav_list, train_label_list)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)


    # # # Validation data
    val_wav_list, val_label_list = parse_data_list(params, "Val")
    valid_ds = generate_dataset(params, val_wav_list, val_label_list)
    valid_ds = valid_ds.batch(BATCH_SIZE)
    valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

    # # # Model
    model = build_model(params)
    # Confirm structure
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"]
    )


    # # # Train
    # Callback
    earlystopping_callback = tf.keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        MODEL_FILENAME, monitor="val_accuracy", save_best_only=True
    )
    callbacks = [earlystopping_callback, checkpoint_callback]

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=valid_ds,
        callbacks=callbacks
    )

    return history


def save_train_history(params, history):
    '''
        save_train_history(params, history)
            Saves the accuracy and loss curves 
            contained in the training history.
    '''
    RESULT_DIR = params['RESULT_DIR']

    # # # Save results
    try:
        os.makedirs(RESULT_DIR)
    except:
        pass
    # Plot and save
    plt.figure(1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy over the epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    plt.savefig(os.path.join(RESULT_DIR, "Accuracy.png"))
    plt.close()

    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss over the epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    plt.savefig(os.path.join(RESULT_DIR, "Loss.png"))
    plt.close()
    return 


def main():
    '''
        main()
            Manages the entire process of the 
            baseline model training/evaluation process
    '''
    params = get_params()
    train_history = do_learning(params)
    save_train_history(params, train_history)
    do_testing(params)


if __name__ == '__main__':
    main()
