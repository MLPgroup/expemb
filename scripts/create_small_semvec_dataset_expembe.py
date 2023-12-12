import os
import gzip
import random
from typing import List, Optional
from dataclasses import dataclass
from simple_parsing import ArgumentParser
from tqdm import tqdm


class DatasetGenerator:
    def __init__(self, srcdir: str, outdir: str, dataset: str, n_examples: int):
        self.srcdir = srcdir
        self.outdir = outdir
        self.dataset = dataset
        self.n_examples = n_examples

        assert os.path.exists(self.srcdir) and os.path.exists(self.outdir)


    def n_expemba_examples(self):
        expemba_datapath = os.path.join(self.srcdir, f"{dataset}_autoenc.train.gz")
        with gzip.open(expemba_datapath, "rt") as f:
            n_examples = sum(1 for line in f)
        return n_examples


    def sample_from_expembe(self, n_examples: int):
        expembe_datapath = os.path.join(self.srcdir, f"{dataset}.train.gz")
        with gzip.open(expembe_datapath, "rt") as f:
            examples = f.readlines()
        return list(set(random.sample(examples, n_examples)))


    def write_to_file(self, examples: list):
        outfile = os.path.join(self.outdir, f"{dataset}.train.gz")
        assert not os.path.exists(outfile), f"{outfile} already exists!"

        with gzip.open(outfile, "wt") as f:
            for example in examples:
                f.write(example)


    def run(self):
        if self.n_examples is None:
            n_examples = self.n_expemba_examples()
        else:
            n_examples = self.n_examples

        examples = self.sample_from_expembe(n_examples)
        self.write_to_file(examples)


@dataclass
class DatasetGenerationAgruments:
    srcdir: str                      # Source dataset directory.
    outdir: str                      # Output directory.
    dataset: List[str]               # List of datasets to process.
    seed: int = Optional[None]       # Seed.
    n_examples: int = Optional[None] # Number of examples to sample to create the new dataset.


if __name__ == "__main__":
    arg_parser = ArgumentParser("Generate SemVec ExpEmb-E with the size equal to the corresponding ExpEmb-A dataset.")
    arg_parser.add_arguments(DatasetGenerationAgruments, dest = "args")

    args = arg_parser.parse_args()
    args = args.args

    if args.seed is not None:
        random.seed(args.seed)

    for dataset in tqdm(args.dataset):
        generator = DatasetGenerator(args.srcdir, args.outdir, dataset, args.n_examples)
        generator.run()
