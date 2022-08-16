import os
import tensorflow as tf
import csv


def parse_data_list(params, split):
    '''
        Reads the list of data paths and labels from the tsv file,
        and returns them in the form of python lists.
    '''
    DATASET_DIR = params['DATASET_DIR']
    wav_list = []
    label_list = []

    with open(params['DATASET_INFO'], "r") as f:
        csv_reader = csv.reader(f, delimiter="\t")
        for row in csv_reader:
            if row[2] == split:
                wav_list.append(os.path.join(DATASET_DIR, row[0]) )
                label_list.append(int(row[1]))
    label_list = tf.keras.utils.to_categorical(label_list, 13)
    return wav_list, label_list


def generate_dataset(params, audio_paths, labels):
    '''
        generate_dataset(params, audio_paths, labels)
            Convert the data paths and labels to a tensorflow dataset.
    '''
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(params, x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def path_to_audio(params, path):
    '''
        path_to_audio(params, path)
            Get audio file path, read it, and return the audio.
    '''
    AUDIO_DURATION = params['AUDIO_DURATION']
    SAMPLING_RATE = params['SAMPLING_RATE']

    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, AUDIO_DURATION*SAMPLING_RATE)
    return audio
