import os
import pickle
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import optuna.integration.lightgbm as lgb_opt
from scipy.stats import kurtosis
from sklearn.model_selection import StratifiedKFold, train_test_split

import utils


class UnsureClassifier:
    """A classifier containing the model and the corresponding training procedure.
    Unsure classifier takes the model predictions and outputs a prediction only if
    the underlying model is confident enough. Otherwise, the output is set to unsure.

    Attributes:
        params:
            LightGBM parameters for constructing and training the model.
            dataset_path: Path of the dataset to be used.
        dataset_path:
            Path to the dataset for training, evaluation and prediction.
        model_path:
            Path to the model for saving and loading.
        best_params_path:
            Binary file containing dictionary of LightGBM parameters.
            If it does not exist, then it is created with hyperparameter optimization.
        miscls_weight_path:
            Path to the .csv file with misclassification cost matrix.
        unsure_coef:
            Weighting coefficient used in the minimization of unsure classification.
            Must be >= 0!
        model:
            (p, t) where "p" if a LightGBM booster and "t" is a numpy array with
            shape (n,) containing classification confidence thresholds for n classes.
    """

    def __init__(
        self,
        params: Dict[str, Any] = None,
        dataset_path: str = None,
        model_path: str = None,
        best_param_path: str = None,
        miscls_weight_path: str = None,
        unsure_coef: float = None,
        model: Tuple[lgb.Booster, np.ndarray] = (None, None),
    ):
        if params is None:
            params = {
                "objective": "binary",
                # "metric": "f1",
                "verbosity": -1,
                "boosting_type": "gbdt",
            }
        if unsure_coef is None:
            unsure_coef = 5.0
        self.params = params
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.best_param_path = best_param_path
        self.miscls_weight_path = miscls_weight_path
        self.unsure_coef = unsure_coef
        self.predictor, self.thresholds = model

    def _load_args(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key) and val is not None:
                setattr(self, key, val)

    def pred_cls_probs(self, x: np.ndarray) -> np.ndarray:
        one_probs = self.predictor.predict(
            x, num_iteration=self.predictor.best_iteration
        )
        probs = np.stack([1 - one_probs, one_probs], axis=1)

        return probs

    def predict_numpy(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        cls_probs = self.pred_cls_probs(x)
        pred: np.ndarray = np.argmax(cls_probs, axis=1)
        pred_probs = np.take_along_axis(
            cls_probs, pred.reshape(-1, 1), axis=1
        ).flatten()
        is_confident: np.ndarray = pred_probs >= self.thresholds[pred]
        pred[np.logical_not(is_confident)] = -1
        return pred, is_confident.size - np.sum(is_confident)

    def _get_best_params(self) -> Dict[str, Any]:
        data, target = utils.get_dataset(self.dataset_path)
        train_x, val_x, train_y, val_y = train_test_split(data, target, test_size=0.1)
        dtrain = lgb.Dataset(train_x, label=train_y)
        dval = lgb.Dataset(val_x, label=val_y)

        if os.path.isfile(self.best_param_path):
            with open(self.best_param_path, "rb") as f:
                best_params: Dict[str, Any] = pickle.load(file=f)
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

    def _get_thresholds(self, best_params: Dict[str, Any]):
        miscls_weights = utils.get_miscls_weights(self.miscls_weight_path)
        class_cnt = miscls_weights.shape[0]
        data, target = utils.get_dataset(self.dataset_path)
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        idx_iter = skf.split(data, target)

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_params, miscls_weights, class_cnt, data, target, skf, idx_iter
            try:
                train_idx, test_idx = next(idx_iter)
            except StopIteration:
                idx_iter = skf.split(data, target)
                train_idx, test_idx = next(idx_iter)
            train_x, val_x, train_y, val_y = train_test_split(
                data[train_idx], target[train_idx], test_size=0.1
            )
            dtrain = lgb.Dataset(train_x, label=train_y)
            dval = lgb.Dataset(val_x, label=val_y)
            self.predictor = lgb.train(
                best_params,
                dtrain,
                num_boost_round=1000,
                feval=utils.lgb_acc_score,
                valid_sets=[dval],
                valid_names=["val"],
                verbose_eval=False,
                early_stopping_rounds=100,
            )
            self.thresholds = np.array(
                [
                    trial.suggest_uniform(f"thresh{i}", 1 / class_cnt, 1.0)
                    for i in range(class_cnt)
                ]
            )
            test_y: np.ndarray = target[test_idx]
            test_pred, unsure_cnt = self.predict_numpy(data[test_idx])

            test_size = test_y.shape[0]
            miscls_cost = utils.get_miscls_cost(
                test_y, test_pred, miscls_weights / test_size
            )

            return miscls_cost + self.unsure_coef * (unsure_cnt / test_size)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=150)
        self.thresholds = np.array(
            [study.best_params[f"thresh{i}"] for i in range(class_cnt)]
        )

    def _train_with_hyperparams(self, best_params: Dict[str, Any], k_fold: int):
        data, target = utils.get_dataset(self.dataset_path)
        skf = StratifiedKFold(n_splits=k_fold, shuffle=True)
        cv_scores = []
        for train_idx, test_idx in skf.split(data, target):
            train_x, val_x, train_y, val_y = train_test_split(
                data[train_idx], target[train_idx], test_size=0.1
            )
            dtrain = lgb.Dataset(train_x, label=train_y)
            dval = lgb.Dataset(val_x, label=val_y)
            self.predictor = lgb.train(
                best_params,
                dtrain,
                num_boost_round=1000,
                feval=utils.lgb_acc_score,
                valid_sets=[dval],
                valid_names=["val"],
                verbose_eval=False,
                early_stopping_rounds=100,
            )
            test_y = target[test_idx]
            test_pred, unsure_cnt = self.predict_numpy(data[test_idx])
            cv_scores.append(utils.eval_scores(test_y, test_pred, unsure_cnt))

        return cv_scores

    def train(
        self,
        dataset_path: str = None,
        model_path: str = None,
        best_param_path: str = None,
        k_fold: int = 10,
    ) -> Tuple[float, float]:
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
        best_params = self._get_best_params()
        best_params.update(self.params)

        self._get_thresholds(best_params)
        cv_scores = self._train_with_hyperparams(best_params, k_fold)
        if self.model_path:
            with open(self.model_path, "wb") as f:
                pickle.dump((self.predictor, self.thresholds), f)

        cv_scores = np.array(cv_scores)
        mean = np.mean(cv_scores, axis=0)
        ex_kur = kurtosis(cv_scores, bias=False, axis=0)
        # https://www.wikiwand.com/en/Unbiased_estimation_of_standard_deviation#/Other_distributions
        std = np.sqrt(
            np.sum((cv_scores - mean) ** 2, axis=0)
            / (len(cv_scores) - 1.5 - ex_kur / 4)
        )
        # https://www.wikiwand.com/en/Chebyshev%27s_inequality
        return tuple(mean - 5 * std)

    def evaluate(
        self, dataset_path: str = None, model_path: str = None
    ) -> Tuple[float, float]:
        self._load_args(dataset_path=dataset_path, model_path=model_path)
        data, target = utils.get_dataset(self.dataset_path)
        if self.predictor is None or self.thresholds is None:
            with open(self.model_path, "rb") as f:
                self.predictor, self.thresholds = pickle.load(f)

        pred, unsure_cnt = self.predict_numpy(data)
        return utils.eval_scores(target, pred, unsure_cnt)

    def predict(
        self, dataset_path: str = None, model_path: str = None
    ) -> Tuple[List[int], float]:
        self._load_args(dataset_path=dataset_path, model_path=model_path)
        if self.predictor is None or self.thresholds is None:
            with open(self.model_path, "rb") as f:
                self.predictor, self.thresholds = pickle.load(f)

        pred, unsure_cnt = self.predict_numpy(utils.get_dataset(self.dataset_path))
        return pred.tolist(), unsure_cnt
