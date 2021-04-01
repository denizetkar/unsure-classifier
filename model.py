import os
import pickle

import lightgbm as lgb
import numpy as np
import optuna.integration.lightgbm as lgb_opt
from scipy.stats import kurtosis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import utils


# TODO: implement unsure classification!
class UnsureClassifier:
    """A classifier containing the model and the corresponding training procedure.
    Unsure classifier takes the model predictions and outputs a prediction only if
    the underlying model is confident enough. Otherwise, the output is set to unsure.

    Attributes:
      params: LightGBM parameters for constructing and training the model.
      dataset_path: Path of the dataset to be used.
      model_path: Path to the model for saving and loading.
      best_params_path: Binary file containing dictionary of LightGBM parameters.
        If it does not exist, then it is created with hyperparameter optimization.
      model: The LightGBM booster model used for classification.
    """

    def __init__(
        self,
        params=None,
        dataset_path=None,
        model_path=None,
        best_param_path=None,
        model=None,
    ):
        if params is None:
            params = {
                "objective": "binary",
                # "metric": "f1",
                "verbosity": -1,
                "boosting_type": "gbdt",
            }
        self.params = params
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.best_param_path = best_param_path
        self.model = model

    def _load_args(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key) and val is not None:
                setattr(self, key, val)

    def get_best_params(self, params=None, dataset_path=None, best_param_path=None):
        self._load_args(
            params=params, dataset_path=dataset_path, best_param_path=best_param_path
        )
        data, target = utils.get_dataset(self.dataset_path)
        train_x, val_x, train_y, val_y = train_test_split(data, target, test_size=0.1)
        dtrain = lgb.Dataset(train_x, label=train_y)
        dval = lgb.Dataset(val_x, label=val_y)

        if os.path.isfile(self.best_param_path):
            with open(self.best_param_path, "rb") as f:
                best_params = pickle.load(file=f)
        else:
            tuner = lgb_opt.LightGBMTuner(
                self.params,
                dtrain,
                feval=utils.lgb_f1_score,
                valid_sets=[dval],
                valid_names=["val"],
                verbose_eval=False,
                early_stopping_rounds=100,
            )
            tuner.run()
            best_params = tuner.best_params

            with open(self.best_param_path, "wb") as f:
                pickle.dump(best_params, file=f)

        return best_params

    def train(
        self, dataset_path=None, model_path=None, best_param_path=None, k_fold=10
    ):
        self._load_args(
            dataset_path=dataset_path,
            model_path=model_path,
            best_param_path=best_param_path,
        )

        if self.best_param_path is None:
            dataset_dir, dataset_file = os.path.split(self.dataset_path)
            self.best_param_path = os.path.join(
                dataset_dir, f"{dataset_file}.best_params"
            )
        best_params = self.get_best_params()
        best_params.update(self.params)

        data, target = utils.get_dataset(self.dataset_path)
        skf = StratifiedKFold(n_splits=k_fold, shuffle=True)
        cv_scores = []
        for train_idx, test_idx in skf.split(data, target):
            train_x, val_x, train_y, val_y = train_test_split(
                data[train_idx], target[train_idx], test_size=0.1
            )
            dtrain = lgb.Dataset(train_x, label=train_y)
            dval = lgb.Dataset(val_x, label=val_y)
            model = lgb.train(
                best_params,
                dtrain,
                num_boost_round=1000,
                feval=utils.lgb_acc_score,
                valid_sets=[dval],
                valid_names=["val"],
                verbose_eval=False,
                early_stopping_rounds=100,
            )
            cv_scores.append(
                accuracy_score(
                    target[test_idx],
                    np.rint(
                        model.predict(
                            data[test_idx], num_iteration=model.best_iteration
                        )
                    ),
                )
            )
        self.model = model
        if self.model_path:
            model.save_model(self.model_path, num_iteration=model.best_iteration)

        cv_scores = np.array(cv_scores)
        mean = np.mean(cv_scores)
        ex_kur = kurtosis(cv_scores, bias=False)
        # https://www.wikiwand.com/en/Unbiased_estimation_of_standard_deviation#/Other_distributions
        std = np.sqrt(
            np.sum((cv_scores - mean) ** 2) / (len(cv_scores) - 1.5 - ex_kur / 4)
        )
        # https://www.wikiwand.com/en/Chebyshev%27s_inequality
        return mean - 5 * std

    def evaluate(self, dataset_path=None, model_path=None):
        self._load_args(dataset_path=dataset_path, model_path=model_path)
        data, target = utils.get_dataset(self.dataset_path)
        if self.model is None:
            self.model = lgb.Booster(model_file=self.model_path, silent=True)

        prediction = np.rint(
            self.model.predict(data, num_iteration=self.model.best_iteration)
        )
        return accuracy_score(target, prediction)

    def predict(self, dataset_path=None, model_path=None):
        self._load_args(dataset_path=dataset_path, model_path=model_path)
        if self.model is None:
            self.model = lgb.Booster(model_file=self.model_path, silent=True)
        return np.rint(
            self.model.predict(
                utils.get_dataset(self.dataset_path),
                num_iteration=self.model.best_iteration,
            )
        ).tolist()
