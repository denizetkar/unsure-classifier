import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def assert_file_path(file_path):
    """Asserts the existence of the file path.

    Args:
      file_path: Path of the file to check.
    """
    if not os.path.isfile(file_path):
        print(f"'{file_path}' file path is not valid!", file=sys.stderr)
        exit(1)


def assert_newfile_path(newfile_path):
    """Asserts the validity of the path for a new file.
    1. It must be in a valid directory,
    2. The basename (file name) must exist.

    Args:
      newfile_path: Path to check for a new file.
    """
    if not os.path.isdir(os.path.dirname(newfile_path)):
        print("new file path is not in a valid directory!", file=sys.stderr)
        exit(1)
    if os.path.basename(newfile_path) == "":
        print("new file path does not lead to a file!", file=sys.stderr)
        exit(1)


def get_dataset(dataset_path, col_labelled=True, row_labelled=True):
    """Assumes the dataset path points to an excel file and reads it.

    Args:
      dataset_path: Path of the excel file dataset.
      col_labelled: Boolean indicating if columns are labelled with a header.
      row_labelled: Boolean indicating if rows are labelled with a column.

    Returns:
      (X, y) numpy arrays where "X" excludes the last column and "y" is the last column.
    """
    assert_file_path(dataset_path)
    dataset = pd.read_excel(
        dataset_path,
        header=0 if col_labelled else None,
        index_col=0 if row_labelled else None,
    )
    return dataset.iloc[:, :-1].to_numpy(), dataset.iloc[:, -1].to_numpy()


def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.rint(y_hat)  # scikits f1 doesn't like probabilities
    return "f1", f1_score(y_true, y_hat), True


def lgb_acc_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.rint(y_hat)  # scikits acc doesn't like probabilities
    return "acc", accuracy_score(y_true, y_hat), True
