# cochlscene
Baseline code for cochlscene dataset

## How to use?

1. Download this repository
```
$ git clone https://github.com/cochlearai/cochlscene
```

2. Install requirements
```
$ cd cochlscene
$ pip install -r requirements.txt
```

3. Change the dataset path (params['DATASET_DIR']) in `params.py`

4. Run main.py
```
$ python main.py
```

5. The trained model is saved as `CochlScene_model.h5` and the confusion matrix can be found in `results` folder.
