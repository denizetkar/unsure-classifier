# UnsureClassifier

A gradient boosted decision tree (GBDT) is used for classifying **tabular data**. It is **multiclass capable** out of the box. It modifies the classification problem by adding 1 more label for unsure. Then, augments the dataset with self generated **unsure datapoints** with a simulation.
* The ratio of unsure points are determined by `--unsure-ratio` parameter.
* Unsure points are initialized from a low discrepancy **sobol sequence**.
* Unsure points are generated as far from each other and as far from other real data points as the simulation allows.
* **The higher the class coefficient, the higher the safety criticalness** and the weaker the particles of that class repel the unsure particles. This ensures less confidence around the input space surrounding safety critical class datapoints (or at least that is the idea).
* Requires a **class coefficient vector** in `.csv` format.
* The coefficients must be non-negative!

The multiclass validation score used during the model training also takes into account of class coefficients.
* An `Fbeta` score with `beta=0.5` (favoring precision) is calculated for each class and weighted by the class coefficients.

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
                        [--unsure-ratio UNSURE_RATIO] [--k-fold K_FOLD]
                        dataset_path cls_coef_path

    positional arguments:
    dataset_path          Path to the dataset for training
    cls_coef_path         Path for misclassification coefficients

    optional arguments:
    -h, --help            show this help message and exit
    --model-path MODEL_PATH
                            Path for the trained model
    --best-param-path BEST_PARAM_PATH
                            Path for the training parameters
    --unsure-ratio UNSURE_RATIO
                            Ratio of unsure samples to the number of real (sure)
                            samples
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
