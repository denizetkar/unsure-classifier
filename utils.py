import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score

import unsure_sim


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
        return dataset[:, :-1], dataset[:, -1]

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
    unsure_labels = np.full((unsure_dataset.shape[0], 1), class_cnt - 1)
    dataset = np.block([[dataset], [unsure_dataset, unsure_labels]])

    return dataset[:, :-1], dataset[:, -1]


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


def eval_scores(
    target: np.ndarray,
    pred: np.ndarray,
    unsure_cnt: int,
    class_cnt: int,
) -> Tuple[float, ...]:
    """Calculates various evaluation scores from the target labels and
    predicted labels. The last score is always sureness ratio.

    Args:
      target: A numpy array of shape (n,) for target labels.
      pred: A numpy array of shape (n,) for predicted labels.
      unsure_cnt: Number of unsure samples in the prediction.
      class_cnt: Number of classes excluding the unsure class.

    Returns:
      A numpy array of shape (d,) with lower bound estimates of each score.
    """
    conf_preds = pred != class_cnt
    return (
        accuracy_score(target[conf_preds], pred[conf_preds]),
        *(
            precision_score(
                target[conf_preds], pred[conf_preds], labels=[i], average="weighted"
            )
            for i in range(class_cnt)
        ),
        1.0 - unsure_cnt / pred.size,
    )
