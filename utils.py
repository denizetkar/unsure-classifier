import os
from typing import Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score


def assert_file_path(file_path: str):
    """Asserts the existence of the file path.

    Args:
      file_path: Path of the file to check.
    """
    assert os.path.isfile(file_path), f"'{file_path}' file path is not valid"


def assert_newfile_path(newfile_path: str):
    """Asserts the validity of the path for a new file.
    1. It must be in a valid directory,
    2. The basename (file name) must exist.

    Args:
      newfile_path: Path to check for a new file.
    """
    assert os.path.isdir(
        os.path.dirname(newfile_path)
    ), "new file path is not in a valid directory"
    assert os.path.basename(newfile_path) != "", "new file path does not lead to a file"


def get_dataset(
    dataset_path: str, col_labelled: bool = True, row_labelled: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Assumes the dataset path points to an excel file and reads it.

    Args:
      dataset_path: Path of the excel file dataset.
      col_labelled: Boolean indicating if columns are labelled with a header.
      row_labelled: Boolean indicating if rows are labelled with a column.

    Returns:
      (X, y) numpy arrays where "X" excludes the last column and "y" is the last column.
    """
    dataset = pd.read_excel(
        dataset_path,
        header=0 if col_labelled else None,
        index_col=0 if row_labelled else None,
    )
    return dataset.iloc[:, :-1].to_numpy(), dataset.iloc[:, -1].to_numpy()


def get_miscls_weights(miscls_weight_path: str) -> np.ndarray:
    """Assumes the path points to a .csv file and reads it.

    Args:
      miscls_weight_path: Path of the misclassification weights.

    Returns:
      A numpy array of shape (n, n) where "n" is the number of classes.
    """
    miscls_weights: np.ndarray = pd.read_csv(miscls_weight_path, header=None).to_numpy()
    assert np.all(
        miscls_weights.diagonal() == 0
    ), "misclassification weight matrix must be diagonally 0"
    assert np.all(miscls_weights >= 0), "misclassification matrix must be non-negative"
    return miscls_weights


def get_miscls_cost(
    target: np.ndarray, pred: np.ndarray, miscls_weights: np.ndarray
) -> float:
    class_cnt = miscls_weights.shape[0]
    sample_size = target.shape[0]
    target_one_hot: np.ndarray = np.zeros((sample_size, class_cnt))
    np.put_along_axis(target_one_hot, target.reshape(-1, 1), 1, axis=1)
    pred_one_hot: np.ndarray = np.zeros((sample_size, class_cnt))
    np.put_along_axis(pred_one_hot, pred.reshape(-1, 1), 1, axis=1)
    pred_one_hot[pred == -1] = 0

    miscls_cost = np.matmul(
        np.matmul(
            pred_one_hot.reshape(sample_size, 1, class_cnt),
            np.broadcast_to(miscls_weights, (sample_size, *miscls_weights.shape)),
        ),
        target_one_hot.reshape(sample_size, class_cnt, 1),
    )
    # np.broadcast_to(miscls_weights, (data_size, *miscls_weights.shape))
    return np.sum(miscls_cost)


def eval_scores(
    target: np.ndarray, pred: np.ndarray, unsure_cnt: int
) -> Tuple[float, float, float]:
    conf_preds = pred != -1
    return (
        accuracy_score(target[conf_preds], pred[conf_preds]),
        precision_score(target[conf_preds], pred[conf_preds]),
        1.0 - unsure_cnt / pred.size,
    )


def lgb_f1_score(
    y_hat: Union[list, np.ndarray], data: lgb.Dataset
) -> Tuple[str, float, bool]:
    y_true = data.get_label()
    y_hat = np.rint(y_hat)  # scikits f1 doesn't like probabilities
    return "f1", f1_score(y_true, y_hat), True


def lgb_acc_score(
    y_hat: Union[list, np.ndarray], data: lgb.Dataset
) -> Tuple[str, float, bool]:
    y_true = data.get_label()
    y_hat = np.rint(y_hat)  # scikits acc doesn't like probabilities
    return "acc", accuracy_score(y_true, y_hat), True
