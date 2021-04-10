import os
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.metrics import confusion_matrix
from statsmodels.stats.proportion import proportion_confint

np.seterr(divide="ignore", invalid="ignore")


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


def get_excel_table(
    dataset_path: str, col_labelled: bool = True, row_labelled: bool = True
) -> np.ndarray:
    """Assumes the dataset path points to an excel file and reads it as it is.

    Args:
      dataset_path: Path of the excel file dataset.
      col_labelled: Boolean indicating if columns are labelled with a header.
      row_labelled: Boolean indicating if rows are labelled with a column.

    Returns:
      A numpy array containing the table inside the dataset file.
    """
    dataset = pd.read_excel(
        dataset_path,
        header=0 if col_labelled else None,
        index_col=0 if row_labelled else None,
    )
    return dataset.to_numpy()


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
    dataset = get_excel_table(dataset_path, col_labelled, row_labelled)
    return dataset[:, :-1], dataset[:, -1].astype(int)


def get_miscls_weights(miscls_weight_path: str) -> np.ndarray:
    """Assumes the path points to a .csv file and reads it.

    Args:
      miscls_weight_path: Path of the misclassification weights.

    Returns:
      A numpy array of shape (c, c) where "c" is the number of classes.
    """
    miscls_weights: np.ndarray = pd.read_csv(miscls_weight_path, header=None).to_numpy()
    assert np.all(
        miscls_weights.diagonal() == 0
    ), "misclassification weight matrix must be diagonally 0"
    assert np.all(
        miscls_weights >= 0
    ), "misclassification weight matrix must be non-negative"
    assert (
        miscls_weights.shape[0] == miscls_weights.shape[1]
    ), "misclassification weight matrix must be square"
    return miscls_weights


def get_miscls_cost(
    target: np.ndarray, pred: np.ndarray, miscls_weights: np.ndarray
) -> float:
    """Calculates the misclassification cost given a vector of targets,
    predictions and the misclassification weights matrix.

    Args:
      target: A numpy array of shape (n,) for target labels.
      pred: A numpy array of shape (n,) for predicted labels. It contains
        "-1" for unsure label.
      miscls_weights: A numpy array of shape (c, c) where "c" is the
        number of classes.

    Returns:
      A float containing the non-negative misclassification cost.
    """
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
    return np.sum(miscls_cost)


def eval_score_counts(
    target: np.ndarray,
    pred: np.ndarray,
    unsure_cnt: int,
    class_cnt: int,
) -> np.ndarray:
    """Calculates various evaluation score counts from the target labels and
    predicted labels. The last score is always sureness ratio.

    Args:
      target: A numpy array of shape (n,) for target labels.
      pred: A numpy array of shape (n,) for predicted labels. It contains
        "-1" for unsure label.
      unsure_cnt: Number of unsure samples in the prediction.
      class_cnt: Number of classes excluding the unsure class.

    Returns:
      A numpy array of shape (d, 2) where 'd' is the number of different
      evaluation scores and the last dimension is for a pair of integers
      (s, n) where 's' is the number of relevant samples and 'n' is the
      total number of samples. So, the corresponding evaluation score is
      's/n'.
    """
    conf_preds = pred != -1
    conf_matrix: np.ndarray = confusion_matrix(
        target[conf_preds], pred[conf_preds], labels=[i for i in range(class_cnt)]
    )
    return np.array(
        (
            (int(conf_matrix.diagonal().sum()), int(conf_matrix.sum())),
            *(
                (int(conf_matrix[i, i]), int(conf_matrix[:, i].sum()))
                for i in range(class_cnt)
            ),
            (int(pred.size) - unsure_cnt, int(pred.size)),
        )
    )


def get_lower_bounds_1(cv_score_counts: np.ndarray) -> np.ndarray:
    """Calculates a 99% confidence lower bound for the cross validation
    scores from mean and standard deviation. Uses Chebyshev's inequality.
    https://www.wikiwand.com/en/Chebyshev%27s_inequality

    Args:
      cv_score_counts: A numpy array of shape (k, d, 2) where 'k' is the
      number of CV folds, 'd' is the number of different evaluation scores
      and the last dimension is for a pair of integers as returned by the
      "eval_score_counts()" method.

    Returns:
      A numpy array of shape (d,) with lower bound estimates of each score.
    """
    cv_scores = cv_score_counts[:, :, 0] / cv_score_counts[:, :, 1]
    mean = np.nanmean(cv_scores, axis=0)
    ex_kur = np.array(kurtosis(cv_scores, bias=False, axis=0, nan_policy="omit"))
    valid_scores = np.logical_not(np.isnan(cv_scores))
    # https://www.wikiwand.com/en/Unbiased_estimation_of_standard_deviation#/Other_distributions
    std = np.sqrt(
        np.nansum((cv_scores - mean) ** 2, axis=0)
        / (np.sum(valid_scores, axis=0) - 1.5 - ex_kur / 4)
    )
    return mean - 5 * std


def get_lower_bounds_2(cv_score_counts: np.ndarray) -> np.ndarray:
    """Calculates a 99% confidence lower bound for the cross validation
    scores from Binomial proportion confidence interval.
    https://www.wikiwand.com/en/Binomial_proportion_confidence_interval#/Clopper%E2%80%93Pearson_interval

    Args:
      cv_score_counts: A numpy array of shape (k, d, 2) where 'k' is the
      number of CV folds, 'd' is the number of different evaluation scores
      and the last dimension is for a pair of integers as returned by the
      "eval_score_counts()" method.

    Returns:
      A numpy array of shape (d,) with lower bound estimates of each score.
    """
    cv_score_counts = np.sum(cv_score_counts, axis=0)

    # https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html
    lb, _ = proportion_confint(
        cv_score_counts[:, 0], cv_score_counts[:, 1], alpha=0.02, method="beta"
    )
    return lb
