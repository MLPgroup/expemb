import os
import torch
import pytorch_lightning as pl
from datetime import datetime
from torch.utils.data import DataLoader
from expemb import (
    ExpressionDataset,
    FileNameDataset,
    ExpEmbTx,
    SemVecExpEmbTx,
    TestingArguments,
    TestingResults,
)
from simple_parsing import ArgumentParser


def main():
    # Command line arguments
    arg_parser = ArgumentParser("ExpEmb-TX testing script.")
    arg_parser.add_arguments(TestingArguments, dest = "options")
    args = arg_parser.parse_args()
    args = args.options

    assert os.path.exists(args.save_dir), f"{args.save_dir} does not exist"
    assert os.path.exists(args.test_file), f"{args.test_file} does not exist"

    modelpath = os.path.join(args.save_dir, f"saved_models/{args.ckpt_name}.ckpt")
    assert os.path.exists(modelpath), f"{modelpath} does not exist"

    # Load model
    tokenizer = torch.load(modelpath)["tokenizer"]
    model_cls = SemVecExpEmbTx if args.semvec else ExpEmbTx
    if args.sympy_timeout is not None:
        model = model_cls.load_from_checkpoint(modelpath, tokenizer = tokenizer, beam_sizes = args.beam_sizes, sympy_timeout = args.sympy_timeout)
    else:
        model = model_cls.load_from_checkpoint(modelpath, tokenizer = tokenizer, beam_sizes = args.beam_sizes)

    # Dataloader
    if args.semvec:
        dataset = FileNameDataset(full_datafile = args.full_file, test_datafile = args.test_file)
    else:
        dataset = ExpressionDataset(
            args.test_file,
            tokenizer = tokenizer,
            max_seq_len = args.max_seq_len,
            n_examples = args.max_test_examples,
        )
    dataloader = DataLoader(dataset, batch_size = args.batch_size, collate_fn = dataset.collate_fn)

    # Logger
    logger = pl.loggers.CSVLogger(save_dir = args.save_dir, name = "test_logs")

    # Test
    trainer = pl.Trainer(accelerator = "auto", precision = args.precision, logger = logger)
    test_results = trainer.test(model = model, dataloaders = dataloader)
    print(f"test_results: {test_results}")

    # Save results
    results = TestingResults(args = args, accuracy = test_results)
    results_file = os.path.join(args.save_dir, f"{args.result_file_prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S%f')}.yaml")
    results.save(results_file)


if __name__ == "__main__":
    main()
