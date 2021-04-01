import argparse

import model
import utils


def train_mode(args):
    """Main execution function for when "train" subcommand is used.

    Args:
      args: A Namespace object containing command line arguments.
    """
    utils.assert_file_path(args.dataset_path)
    if args.model_path:
        utils.assert_newfile_path(args.model_path)
    classifier = model.UnsureClassifier(
        dataset_path=args.dataset_path, model_path=args.model_path
    )
    lower_bound = classifier.train(k_fold=args.k_fold)
    print(lower_bound)


def eval_mode(args):
    """Main execution function for when "eval" subcommand is used.

    Args:
      args: A Namespace object containing command line arguments.
    """
    utils.assert_file_path(args.dataset_path)
    utils.assert_file_path(args.model_path)
    classifier = model.UnsureClassifier(
        dataset_path=args.dataset_path, model_path=args.model_path
    )
    eval_score = classifier.evaluate()
    print(eval_score)


def pred_mode(args):
    """Main execution function for when "pred" subcommand is used.

    Args:
      args: A Namespace object containing command line arguments.
    """
    utils.assert_file_path(args.dataset_path)
    utils.assert_file_path(args.model_path)
    classifier = model.UnsureClassifier(
        dataset_path=args.dataset_path, model_path=args.model_path
    )
    pred = classifier.predict()
    print(pred)


parser = argparse.ArgumentParser()
subparser = parser.add_subparsers()
train_parser = subparser.add_parser("train")
train_parser.add_argument(
    "dataset_path",
    help="Path to the dataset for training",
)
train_parser.add_argument("--model-path", help="Path for saving the trained model")
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
eval_parser.set_defaults(func=eval_mode)

pred_parser = subparser.add_parser("pred")
pred_parser.add_argument("dataset_path", help="Path to the dataset for prediction")
pred_parser.add_argument("model_path", help="Path for loading the model")
pred_parser.set_defaults(func=pred_mode)


if __name__ == "__main__":
    args = parser.parse_args()
    args.func(args)
