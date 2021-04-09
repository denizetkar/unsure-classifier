# UnsureClassifier

A gradient boosted decision tree (GBDT) is used for classifying **tabular data**. It is **multiclass capable** out of the box. It uses thresholds to judge the classifier confidence for a new sample `x` to be predicted. If the most likely class `c` is not confident enough, then the classifier makes no prediction and **is unsure** (**-1** label for unsure).

A threshold for each class is calculated with a hyperparameter optimization framework named `optuna`. The optimization uses a `C x C` **misclassification weights matrix** where entry `[i,j]` is the cost for predicting a sample of class `j` as class `i`, and `C` is the number of classes.
* Requires a misclassification weights matrix in `.csv` format.
* The diagonals must be `0` and all entries must be non-negative!

## Usage

    usage: main.py [-h] {train,eval,pred} ...

    positional arguments:
    {train,eval,pred}

    optional arguments:
    -h, --help         show this help message and exit

***
### In training mode:
    usage: main.py train [-h] [--model-path MODEL_PATH]
                        [--best-param-path BEST_PARAM_PATH]
                        [--unsure-coef UNSURE_COEF] [--k-fold K_FOLD]
                        dataset_path miscls_weight_path

    positional arguments:
    dataset_path          Path to the dataset for training
    miscls_weight_path    Path for misclassification weights

    optional arguments:
    -h, --help            show this help message and exit
    --model-path MODEL_PATH
                            Path for the trained model
    --best-param-path BEST_PARAM_PATH
                            Path for the training parameters
    --unsure-coef UNSURE_COEF
                            Weighting coefficient for minimizing unsure
                            classification
    --k-fold K_FOLD, -k K_FOLD
                            Number of folds to use for Cross Validation

***
### In evaluation mode:
    usage: main.py eval [-h] dataset_path model_path class_cnt

    positional arguments:
    dataset_path  Path to the dataset for evaluation
    model_path    Path for loading the model
    class_cnt     Number of classes

    optional arguments:
    -h, --help    show this help message and exit

***
### In prediction mode:
    usage: main.py pred [-h] dataset_path model_path class_cnt

    positional arguments:
    dataset_path  Path to the dataset for prediction
    model_path    Path for loading the model
    class_cnt     Number of classes

    optional arguments:
    -h, --help    show this help message and exit
