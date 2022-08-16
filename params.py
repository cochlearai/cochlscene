def get_params():
    '''
        get_params()
            Returns a python dict variable
            that contains all the necessary information.
    '''
    params = dict()

    params['DATASET_DIR'] = "./cochlscene_data"
    params['DATASET_INFO'] = "Data_info.tsv"
    
    # Audio data parameters
    params['AUDIO_DURATION'] = 10
    params['SAMPLING_RATE'] = 44100

    # Mel-spectrogram parameters
    params['N_FFT'] = 4096
    params['HOP_LENGTH'] = 4096
    params['N_MELS'] = 128

    # Model parameters
    params['BATCH_SIZE'] = 128
    params['EPOCHS'] = 100
    params['N_CLASSES'] = 13
    params['MODEL_FILENAME'] = "CochlScene_model.h5"
    
    params['RESULT_DIR'] = "./results"
    return params
