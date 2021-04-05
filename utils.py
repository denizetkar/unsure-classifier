import os
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.metrics import confusion_matrix
from statsmodels.stats.proportion import proportion_confint

import unsure_sim

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


def calc_unsure_dataset(
    dataset: np.ndarray, cls_coefs: np.ndarray, unsure_ratio: float
) -> np.ndarray:
    sim = unsure_sim.UnsureSimulator(dataset, cls_coefs, unsure_ratio)
    return sim.simulate()


def get_dataset(
    dataset_path: str,
    col_labelled: bool = True,
    row_labelled: bool = True,
    load_unsures: bool = False,
    cls_coefs: np.ndarray = None,
    unsure_ratio: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assumes the dataset path points to an excel file and reads it.

    Args:
      dataset_path: Path of the excel file dataset.
      col_labelled: Boolean indicating if columns are labelled with a header.
      row_labelled: Boolean indicating if rows are labelled with a column.
      load_unsures: Boolean indicating if the unsure samples are to be also loaded.
      cls_coefs: Class coefficients for when "load_unsures" is True.
      unsure_ratio: Ratio of unsure to real samples for when "load_unsures" is True.

    Returns:
      (X, y) numpy arrays where "X" excludes the last column and "y" is the last column.
    """
    dataset = get_excel_table(dataset_path, col_labelled, row_labelled)
    if not load_unsures:
        return dataset[:, :-1], dataset[:, -1].astype(int)

    unsures_path = os.path.join(
        os.path.dirname(dataset_path),
        "{}.unsure.csv".format(os.path.basename(dataset_path)),
    )
    if os.path.isfile(unsures_path):
        unsure_dataset = pd.read_csv(unsures_path, header=None).to_numpy()
    else:
        unsure_dataset = calc_unsure_dataset(dataset, cls_coefs, unsure_ratio)
        np.savetxt(unsures_path, unsure_dataset, delimiter=",")
    class_cnt = cls_coefs.size
    unsure_labels = np.full((unsure_dataset.shape[0], 1), class_cnt)
    dataset = np.block([[dataset], [unsure_dataset, unsure_labels]])

    return dataset[:, :-1], dataset[:, -1].astype(int)


def get_cls_coefs(cls_coef_path: str) -> np.ndarray:
    """Assumes the path points to a .csv file and reads it.

    Args:
      cls_coef_path: Path of the class coefficients.

    Returns:
      A numpy array of shape (n,) where "n" is the number of classes.
    """
    cls_coefs: np.ndarray = pd.read_csv(cls_coef_path, header=None).to_numpy()
    assert np.all(cls_coefs > 0), "class coefficient vector must be positive"
    return cls_coefs.flatten()


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
    conf_preds = pred != class_cnt
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
