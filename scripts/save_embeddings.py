import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import dataclass
from simple_parsing import ArgumentParser
from expemb import ExpEmbTx, ExpressionDataset


@dataclass
class Args:
    modelpath: str          # Save checkpoint location.
    filepath: str           # Data file location.
    outfilepath: str        # Output file location.
    n_examples: int = -1    # Maximum number of examples to read from the input file.
    batch_size: int = 256   # Batch size.



class EmbeddingGenerator:
    def __init__(self, modelpath: str, filepath: str, outfilepath: str, n_examples: int, batch_size: int):
        assert os.path.exists(modelpath) and os.path.exists(filepath) and not os.path.exists(outfilepath)

        self.modelpath = modelpath
        self.filepath = filepath
        self.outfilepath = outfilepath
        self.n_examples = n_examples
        self.batch_size = batch_size
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        tokenizer = torch.load(modelpath)["tokenizer"]
        self.model = ExpEmbTx.load_from_checkpoint(modelpath, tokenizer = tokenizer).to(self.device)
        dataset = ExpressionDataset(filepath = filepath, tokenizer = tokenizer, n_examples = n_examples, max_seq_len = 510)
        self.dataloader = DataLoader(dataset, batch_size = batch_size, collate_fn = dataset.collate_fn)


    @torch.no_grad()
    def run(self):
        emb_list = []
        exp_list = []
        for src, exps in tqdm(self.dataloader):
            src = src.to(self.device)
            src_mask, src_padding_mask = self.model.create_src_mask(src)
            memory = self.model.encode(src = src, src_mask = src_mask, src_padding_mask = src_padding_mask)

            memory[src_padding_mask.T] = float("-inf")
            memory[0, :, :] = float("-inf")
            memory[-1, :, :] = float("-inf")
            embeddings, _ = memory.max(dim = 0)

            for idx in range(memory.size(1)):
                emb = embeddings[idx].detach().cpu().numpy()
                exp = exps[idx]
                emb_list.append(emb)
                exp_list.append(exp)

        torch.save({
            "modelpath": self.modelpath,
            "filepath": self.filepath,
            "n_examples": self.n_examples,
            "emb_list": emb_list,
            "exp_list": exp_list,
        }, self.outfilepath)


if __name__ == "__main__":
    arg_parser = ArgumentParser("Script to generate and save embeddings on disk.")
    arg_parser.add_arguments(Args, dest = "options")
    args = arg_parser.parse_args()

    generator = EmbeddingGenerator(
        modelpath = args.options.modelpath,
        filepath = args.options.filepath,
        outfilepath = args.options.outfilepath,
        n_examples = args.options.n_examples,
        batch_size = args.options.batch_size,
    )
    generator.run()
