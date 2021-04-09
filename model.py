import os
import pickle
from typing import Any, Dict, Tuple, Union

import lightgbm as lgb
import numpy as np
import optuna.integration.lightgbm as lgb_opt
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import utils


class UnsureClassifier:
    """A classifier containing the model and the corresponding training procedure.
    Unsure classifier takes the model predictions and outputs a prediction only if
    the underlying model is confident enough. Otherwise, the output is set to unsure.

    Attributes:
        cls_coefs:
            Coefficients of safety criticalness for each class.
        class_cnt:
            Number of classes excluding the unsure class.
        params:
            LightGBM parameters for constructing and training the model.
        dataset_path:
            Path to the dataset for training, evaluation and prediction.
        model_path:
            Path to the model for saving and loading.
        best_params_path:
            Binary file containing dictionary of LightGBM parameters.
            If it does not exist, then it is created with hyperparameter optimization.
        cls_coef_path:
            Path to the .csv file with class coefficients that are proportional to
            how safety critical it is to predict that class.
        unsure_ratio:
            Ratio of unsure samples to the number of real (sure) samples.
        model:
            The LightGBM booster model used for classification.
    """

    def __init__(
        self,
        dataset_path: str = None,
        model_path: str = None,
        best_param_path: str = None,
        cls_coef_path: str = None,
        class_cnt: int = None,
        unsure_ratio: float = None,
        model: lgb.Booster = None,
    ):
        if cls_coef_path is not None:
            self.cls_coefs = utils.get_cls_coefs(cls_coef_path)
            self.class_cnt = self.cls_coefs.size
        else:
            self.class_cnt = class_cnt
        if unsure_ratio is None:
            unsure_ratio = 1 / self.class_cnt
        self.params = {
            "objective": "multiclass",
            "metric": "multi_logloss",
            "num_class": self.class_cnt + 1,
            # "metric": "f1",
            "verbosity": -1,
            "boosting_type": "gbdt",
        }
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.best_param_path = best_param_path
        self.unsure_ratio = unsure_ratio
        self.model = model

    def _load_args(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key) and val is not None:
                setattr(self, key, val)

    def pred_cls_probs(self, x: np.ndarray) -> np.ndarray:
        probs = self.model.predict(x, num_iteration=self.model.best_iteration)
        return probs

    def predict_numpy(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        cls_probs = self.pred_cls_probs(x)
        pred: np.ndarray = np.argmax(cls_probs, axis=1)
        return pred, int(np.sum(pred == self.class_cnt))

    def lgb_fbeta_score(
        self, y_hat: Union[list, np.ndarray], data: lgb.Dataset
    ) -> Tuple[str, float, bool]:
        y_true = data.get_label()
        y_hat = np.argmax(y_hat.reshape(self.class_cnt + 1, -1), axis=0)
        score = np.dot(
            [
                fbeta_score(y_true, y_hat, labels=[i], beta=0.5, average="weighted")
                for i in range(self.class_cnt)
            ],
            self.cls_coefs / np.sum(self.cls_coefs),
        )
        return "fbeta", score, True

    def _get_best_params(self) -> Dict[str, Any]:
        data, target = utils.get_dataset(
            self.dataset_path,
            load_unsures=True,
            cls_coefs=self.cls_coefs,
            unsure_ratio=self.unsure_ratio,
        )
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
                feval=self.lgb_fbeta_score,
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

    def _train_with_hyperparams(self, best_params: Dict[str, Any], k_fold: int):
        data, target = utils.get_dataset(
            self.dataset_path,
            load_unsures=True,
            cls_coefs=self.cls_coefs,
            unsure_ratio=self.unsure_ratio,
        )
        skf = StratifiedKFold(n_splits=k_fold, shuffle=True)
        cv_scores = []
        for train_idx, test_idx in skf.split(data, target):
            train_x, val_x, train_y, val_y = train_test_split(
                data[train_idx], target[train_idx], test_size=0.1
            )
            dtrain = lgb.Dataset(train_x, label=train_y)
            dval = lgb.Dataset(val_x, label=val_y)
            self.model = lgb.train(
                best_params,
                dtrain,
                num_boost_round=1000,
                feval=self.lgb_fbeta_score,
                valid_sets=[dval],
                valid_names=["val"],
                verbose_eval=False,
                early_stopping_rounds=100,
            )
            test_y = target[test_idx]
            test_pred, unsure_cnt = self.predict_numpy(data[test_idx])
            cv_scores.append(
                utils.eval_score_counts(test_y, test_pred, unsure_cnt, self.class_cnt)
            )

        return cv_scores

    def train(
        self,
        dataset_path: str = None,
        model_path: str = None,
        best_param_path: str = None,
        k_fold: int = 10,
    ) -> np.ndarray:
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

        cv_score_counts = self._train_with_hyperparams(best_params, k_fold)
        if self.model_path:
            self.model.save_model(
                self.model_path, num_iteration=self.model.best_iteration
            )

        cv_score_counts = np.stack(cv_score_counts, axis=0)
        return utils.get_lower_bounds_2(cv_score_counts)

    def evaluate(self, dataset_path: str = None, model_path: str = None) -> np.ndarray:
        self._load_args(dataset_path=dataset_path, model_path=model_path)
        data, target = utils.get_dataset(self.dataset_path)
        if self.model is None:
            self.model = lgb.Booster(model_file=self.model_path, silent=True)

        pred, unsure_cnt = self.predict_numpy(data)
        score_counts = utils.eval_score_counts(
            target,
            pred,
            unsure_cnt,
            self.class_cnt,
        )
        return score_counts[:, 0] / score_counts[:, 1]

    def predict(
        self, dataset_path: str = None, model_path: str = None
    ) -> Tuple[np.ndarray, float]:
        self._load_args(dataset_path=dataset_path, model_path=model_path)
        if self.model is None:
            self.model = lgb.Booster(model_file=self.model_path, silent=True)

        pred, unsure_cnt = self.predict_numpy(utils.get_excel_table(self.dataset_path))
        return pred, unsure_cnt
