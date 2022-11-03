import os
from expemb import (
    TxTrainer,
    SemVecTxTrainer,
    TxModelArguments,
    TrainingArguments,
)
from simple_parsing import ArgumentParser


def main():
    # Command line arguments
    arg_parser = ArgumentParser("ExpEmb-TX training script")
    arg_parser.add_arguments(TxModelArguments, dest = "model")
    arg_parser.add_arguments(TrainingArguments, dest = "train")
    args = arg_parser.parse_args()
    model_args, train_args = args.model, args.train

    if train_args.semvec:
        trainer = SemVecTxTrainer(model_args = model_args, train_args = train_args)
    else:
        trainer = TxTrainer(model_args = model_args, train_args = train_args)

    trainer.run_training()


if __name__ == "__main__":
    os.environ["WANDB_SILENT"] = "true"
    main()
