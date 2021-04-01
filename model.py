import os
import pickle

import lightgbm as lgb
import numpy as np
import optuna.integration.lightgbm as lgb_opt
from scipy.stats import kurtosis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import utils


def get_best_params(params, data, target, best_param_path):
    train_x, val_x, train_y, val_y = train_test_split(data, target, test_size=0.1)
    dtrain = lgb.Dataset(train_x, label=train_y)
    dval = lgb.Dataset(val_x, label=val_y)

    if os.path.isfile(best_param_path):
        with open(best_param_path, "rb") as f:
            best_params = pickle.load(file=f)
    else:
        tuner = lgb_opt.LightGBMTuner(
            params,
            dtrain,
            feval=utils.lgb_f1_score,
            valid_sets=[dval],
            valid_names=["val"],
            verbose_eval=False,
            early_stopping_rounds=100,
        )
        tuner.run()
        best_params = tuner.best_params

        with open(best_param_path, "wb") as f:
            pickle.dump(best_params, file=f)

    return best_params


def train(dataset_path, model_path=None, best_param_path=None, k_fold=10):
    data, target = utils.get_dataset(dataset_path)

    params = {
        "objective": "binary",
        # "metric": "f1",
        "verbosity": -1,
        "boosting_type": "gbdt",
    }
    if best_param_path is None:
        dataset_dir, dataset_file = os.path.split(dataset_path)
        best_param_path = os.path.join(dataset_dir, f"{dataset_file}.best_params")
    best_params = get_best_params(params, data, target, best_param_path)
    best_params.update(params)

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
                    model.predict(data[test_idx], num_iteration=model.best_iteration)
                ),
            )
        )
    cv_scores = np.array(cv_scores)
    mean = np.mean(cv_scores)
    ex_kur = kurtosis(cv_scores, bias=False)
    # https://www.wikiwand.com/en/Unbiased_estimation_of_standard_deviation#/Other_distributions
    std = np.sqrt(np.sum((cv_scores - mean) ** 2) / (len(cv_scores) - 1.5 - ex_kur / 4))

    if model_path:
        model.save_model(model_path, num_iteration=model.best_iteration)

    # https://www.wikiwand.com/en/Chebyshev%27s_inequality
    return mean - 5 * std


def evaluate(dataset_path, model_path):
    data, target = utils.get_dataset(dataset_path)
    model = lgb.Booster(model_file=model_path, silent=True)

    prediction = np.rint(model.predict(data, num_iteration=model.best_iteration))
    return accuracy_score(target, prediction)


def predict(dataset_path, model_path):
    model = lgb.Booster(model_file=model_path, silent=True)
    return np.rint(
        model.predict(
            utils.get_dataset(dataset_path), num_iteration=model.best_iteration
        )
    ).tolist()
