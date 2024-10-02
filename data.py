import argparse

from utils import run_prepare_data


def parser():
    parser = argparse.ArgumentParser(
        description="Prepare the medical data for training"
    )
    parser.add_argument(
        "-dt",
        "--dataset_type",
        type=str,
        default="Tr",
        help="The dataset to prepare",
        choices=["Tr", "Val", "Ts"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parser()
    run_prepare_data(args)
