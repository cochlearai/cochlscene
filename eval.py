import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import kapre
from kapre import STFT, Magnitude, MagnitudeToDecibel,ApplyFilterbank
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from prepare_data import *

def do_testing(params):
    '''
        do_testing(params)
            Manages the performance evaluation.
            Performs the inferernce and shows 
            the performance evaluation results.
            - Training set: Precision, Recall, F-score
            - Validation set: Precision, Recall, F-score
            - Test set: Precision, Recall, F-score, Confusion matrix
    '''
    MODEL_FILENAME = params['MODEL_FILENAME']
    BATCH_SIZE = params['BATCH_SIZE']
    DATASET_DIR = params['DATASET_DIR']
    RESULT_DIR = params['RESULT_DIR']
    labels = os.listdir(os.path.join(DATASET_DIR, "Train"))
    labels.sort()

    # Load model and predict
    model = tf.keras.models.load_model(MODEL_FILENAME, 
                                       custom_objects={'STFT':STFT, 
                                           'Magnitude':Magnitude, 
                                           'ApplyFilterbank':ApplyFilterbank, 
                                           'MagnitudeToDecibel':MagnitudeToDecibel}
                                       )

    # # # Training set performance check
    train_wav_list, train_label_list = parse_data_list(params, "Train")
    train_ds = generate_dataset(params, train_wav_list, train_label_list)
    train_ds = train_ds.batch(BATCH_SIZE)
    y = model.predict(train_ds)
    y = tf.math.argmax(y, -1)
    train_label_list = tf.math.argmax(train_label_list, -1)
    # Precision, Recall, F-score
    print("# # # # # Training set # # # # #")
    p, r, fscore, support = precision_recall_fscore_support(train_label_list, y)
    print("Label          \tPrecision\tRecall\tF-score")
    for i in range(len(labels)):
        print("{}\t{:.3f}    \t{:.3f} \t{:.3f}".format(labels[i].ljust(15), p[i], r[i], fscore[i]))


    # # # Validation set performance check
    val_wav_list, val_label_list = parse_data_list(params, "Val")
    valid_ds = generate_dataset(params, val_wav_list, val_label_list)
    valid_ds = valid_ds.batch(BATCH_SIZE)
    y = model.predict(valid_ds)
    y = tf.math.argmax(y, -1)
    val_label_list = tf.math.argmax(val_label_list, -1)
    # Precision, Recall, F-score
    print("# # # # # Validation set # # # # #")
    p, r, fscore, support = precision_recall_fscore_support(val_label_list, y)
    print("Label          \tPrecision\tRecall\tF-score")
    for i in range(len(labels)):
        print("{}\t{:.3f}    \t{:.3f} \t{:.3f}".format(labels[i].ljust(15), p[i], r[i], fscore[i]))


    test_wav_list, test_label_list = parse_data_list(params, "Test")
    test_ds = generate_dataset(params, test_wav_list, test_label_list)
    test_ds = test_ds.batch(BATCH_SIZE)
    y = model.predict(test_ds)
    y = tf.math.argmax(y, -1)
    test_label_list = tf.math.argmax(test_label_list, -1)

    # Confusion matrix
    save_confusionmatrix(test_label_list, y, labels, RESULT_DIR)

    # Precision, Recall, F-score
    print("# # # # # Test set # # # # #")
    p, r, fscore, support = precision_recall_fscore_support(test_label_list, y)
    print("Label          \tPrecision\tRecall\tF-score")
    for i in range(len(labels)):
        print("{}\t{:.3f}    \t{:.3f} \t{:.3f}".format(labels[i].ljust(15), p[i], r[i], fscore[i]))
    return 


def save_confusionmatrix(y_true, y_esti, labels, path):
    '''
        save_confusionmatrix(y_true, y_esti, labels, path)
            Calculates confusion matrix,
            draws a figure, and saves it.
    '''
    cm = confusion_matrix(y_true, y_esti)

    plt.figure(figsize=(10,11))
    ax = sns.heatmap(cm, annot=True, fmt='d' ,cmap='Blues')

    # ax.set_title('Confusion Matrix\n\n');
    ax.set_xlabel('Predicted classes')
    ax.set_ylabel('Actual classes');

    # Labels
    ax.xaxis.set_ticklabels(labels, rotation=45, ha='right', minor=False)
    ax.yaxis.set_ticklabels(labels, rotation=45, ha='right', minor=False)

    plt.show()
    plt.savefig(os.path.join(path, "Confusion_matrix_original.eps"))
    plt.close()
    return 
