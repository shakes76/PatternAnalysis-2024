"""Example usage of triamese network for skin lesion classification"""

import argparse

from trainer import SiameseController, HyperParams


def main():
    """Runs program"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load-model")
    args = parser.parse_args()

    trainer = SiameseController(HyperParams(), "predict_model")
    if args.load_model is not None:
        trainer.load_model(args.load_model)

    svm_train_dataset = 


if __name__ == "__main__":
    main()
