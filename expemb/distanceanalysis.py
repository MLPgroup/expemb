import os
import csv
import json
import gzip
import torch
import numpy as np
import pytorch_lightning as pl
from typing import List
from datetime import datetime
from scipy.spatial.distance import cdist
from tqdm import tqdm
from expemb import ExpEmbTx, TrainingArguments


class DistanceAnalysis:
    def __init__(
        self,
        model_cls: str,
        save_dir: str,
        ckpt_name: str,
        train_file: str,
        test_file: str,
        batch_size: int,
    ):
        self.model_cls = model_cls
        self.save_dir = save_dir
        self.ckpt_name = ckpt_name
        self.train_file = train_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.k = 5

        train_args = TrainingArguments.load(os.path.join(self.save_dir, "train_args.yaml"))
        self.max_seq_len = train_args.max_seq_len

        self.test_examples = self.load_test_examples()
        self.model = self.load_model()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()


    def load_model(self):
        model_path = os.path.join(self.save_dir, f"saved_models/{self.ckpt_name}.ckpt")
        assert os.path.exists(model_path), f"{model_path} does not exist."

        tokenizer = torch.load(model_path)["tokenizer"]
        if self.model_cls == "ExpEmbTx":
            return ExpEmbTx.load_from_checkpoint(model_path, tokenizer = tokenizer)
        else:
            raise NotImplementedError(f"{self.model_cls} is not supported.")


    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]


    def load_test_examples(self):
        test_examples = []
        with gzip.open(self.test_file, "rt") as f:
            csvreader = csv.reader(f)
            for row in csvreader:
                test_examples.append(row[0])

        return test_examples


    @torch.no_grad()
    def get_expression_embeddings(self):
        exp_list = set()
        # Expressions from training file
        with gzip.open(self.train_file, "rt") as f:
            for line in f:
                line = line.strip()
                for exp in line.split("\t"):
                    if len(exp.split(" ")) <= self.max_seq_len:
                        exp_list.add(exp)

        # Expressions from test examples
        for example in self.test_examples:
            exp_list.add(example)

        exp_list = list(exp_list)

        emb_list = []
        for exp_batch in tqdm(self.chunks(exp_list, self.batch_size)):
            emb_batch = self.model.get_embedding(exp_batch)
            for idx in range(len(exp_batch)):
                emb = emb_batch[idx].cpu().numpy()
                emb_list.append(emb)

        return exp_list, np.array(emb_list)


    def find_nearest_neighbors(self, x: str, exp_list: List[str], emb_list: np.ndarray):
        embx = self.get_embedding(x)

        # Compute distance matrix
        distmat = cdist(XA = embx[None, :], XB = emb_list, metric = "cosine")[0]

        # Exclude x
        idxx = exp_list.index(x)
        distmat[idxx] = float("inf")

        # k nearest elements
        nearest_idx = np.argpartition(distmat, self.k)[:self.k]

        result = {exp_list[idx] : distmat[idx] for idx in nearest_idx}
        result = dict(sorted(result.items(), key=lambda item: item[1]))

        return result


    @torch.no_grad()
    def get_embedding(self, exp):
        emb = self.model.get_embedding([exp])[0]
        return emb.cpu().numpy()


    def run(self):
        exp_list, emb_list = self.get_expression_embeddings()
        results_file = os.path.join(self.save_dir, f"dist_analysis_results_{datetime.now().strftime('%Y%m%d-%H%M%S%f')}.json")

        results = {}
        for example in self.test_examples:
            assert example in exp_list, f"{example['x2']} is not present in the expression list."

            nns = self.find_nearest_neighbors(
                x = example,
                exp_list = exp_list,
                emb_list = emb_list,
            )
            results[example] = nns

        with open(results_file, "w") as f:
            json.dump(results, f, indent = 4)

        print(f"Written results to {results_file}")
