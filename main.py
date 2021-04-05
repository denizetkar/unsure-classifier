import argparse

import model
import utils


def train_mode(args: argparse.Namespace):
    """Main execution function for when "train" subcommand is used.

    Args:
      args: A Namespace object containing command line arguments.
    """
    utils.assert_file_path(args.dataset_path)
    utils.assert_file_path(args.cls_coef_path)
    if args.model_path:
        utils.assert_newfile_path(args.model_path)
    if args.best_param_path:
        utils.assert_newfile_path(args.best_param_path)
    classifier = model.UnsureClassifier(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        best_param_path=args.best_param_path,
        cls_coef_path=args.cls_coef_path,
        unsure_ratio=args.unsure_coef,
    )
    lower_bounds = classifier.train(k_fold=args.k_fold)
    print(lower_bounds)


def eval_mode(args: argparse.Namespace):
    """Main execution function for when "eval" subcommand is used.

    Args:
      args: A Namespace object containing command line arguments.
    """
    utils.assert_file_path(args.dataset_path)
    utils.assert_file_path(args.model_path)
    assert args.class_cnt > 1, "class count is not bigger than 1"
    classifier = model.UnsureClassifier(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        class_cnt=args.class_cnt,
    )
    eval_scores = classifier.evaluate()
    print(eval_scores)


def pred_mode(args: argparse.Namespace):
    """Main execution function for when "pred" subcommand is used.

    Args:
      args: A Namespace object containing command line arguments.
    """
    utils.assert_file_path(args.dataset_path)
    utils.assert_file_path(args.model_path)
    assert args.class_cnt > 1, "class count is not bigger than 1"
    classifier = model.UnsureClassifier(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        class_cnt=args.class_cnt,
    )
    pred, unsure_cnt = classifier.predict()
    print((pred, unsure_cnt))


parser = argparse.ArgumentParser()
subparser = parser.add_subparsers()
train_parser = subparser.add_parser("train")
train_parser.add_argument(
    "dataset_path",
    help="Path to the dataset for training",
)
train_parser.add_argument(
    "cls_coef_path", help="Path for misclassification coefficients"
)
train_parser.add_argument("--model-path", help="Path for the trained model")
train_parser.add_argument("--best-param-path", help="Path for the training parameters")
train_parser.add_argument(
    "--unsure-coef",
    type=float,
    help="Weighting coefficient for minimizing unsure classification",
)
train_parser.add_argument(
    "--k-fold",
    "-k",
    type=int,
    help="Number of folds to use for Cross Validation",
    default=20,
)
train_parser.set_defaults(func=train_mode)

eval_parser = subparser.add_parser("eval")
eval_parser.add_argument("dataset_path", help="Path to the dataset for evaluation")
eval_parser.add_argument("model_path", help="Path for loading the model")
eval_parser.add_argument("class_cnt", type=int, help="Number of classes")
eval_parser.set_defaults(func=eval_mode)

pred_parser = subparser.add_parser("pred")
pred_parser.add_argument("dataset_path", help="Path to the dataset for prediction")
pred_parser.add_argument("model_path", help="Path for loading the model")
pred_parser.add_argument("class_cnt", type=int, help="Number of classes")
pred_parser.set_defaults(func=pred_mode)


if __name__ == "__main__":
    args = parser.parse_args()
    args.func(args)
