import os
import csv
import gzip
import torch
import numpy as np
import pytorch_lightning as pl
from typing import List
from datetime import datetime
from scipy.spatial.distance import cdist
from tqdm import tqdm
from expemb import ExpEmbTx


class EmbeddingMathematics:
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
        self.k = 1

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
        with open(self.test_file, "r") as f:
            csvreader = csv.DictReader(f, delimiter = ",")
            for row in csvreader:
                test_examples.append({
                    "x1": row["x1"],
                    "y1": row["y1"],
                    "x2": row["x2"],
                    "y2": row["y2"],
                })

        return test_examples


    @torch.no_grad()
    def get_expression_embeddings(self):
        exp_list = set()
        # Expressions from training file
        with gzip.open(self.train_file, "rt") as f:
            for line in f:
                line = line.strip()
                for exp in line.split("\t"):
                    if len(exp.split(" ")) <= self.model.max_seq_len:
                        exp_list.add(exp)

        # Expressions from test examples
        for example in self.test_examples:
            exp_list.add(example["x1"])
            exp_list.add(example["x2"])
            exp_list.add(example["y1"])
            exp_list.add(example["y2"])

        exp_list = list(exp_list)

        emb_list = []
        for exp_batch in tqdm(self.chunks(exp_list, self.batch_size)):
            emb_batch = self.model.get_embedding(exp_batch)
            for idx in range(len(exp_batch)):
                emb = emb_batch[idx].cpu().numpy()
                emb_list.append(emb)

        return exp_list, np.array(emb_list)


    def run_analogy(self, x1: str, y1: str, y2: str, x2: str, exp_list: List[str], emb_list: np.ndarray):
        embx1 = self.get_embedding(x1)
        emby1 = self.get_embedding(y1)
        emby2 = self.get_embedding(y2)
        embx2 = embx1 - emby1 + emby2

        # Compute distance matrix
        distmat = cdist(XA = embx2[None, :], XB = emb_list, metric = "cosine")[0]

        # Exclude x1, y1, and y2
        idxx1 = exp_list.index(x1)
        idxy1 = exp_list.index(y1)
        idxy2 = exp_list.index(y2)
        idxx2 = exp_list.index(x2)
        distmat[idxx1] = float("inf")
        distmat[idxy1] = float("inf")
        distmat[idxy2] = float("inf")

        # k nearest elements
        nearest_idx = np.argpartition(distmat, self.k)[:self.k]

        result = {exp_list[idx] : distmat[idx] for idx in nearest_idx}
        result = dict(sorted(result.items(), key=lambda item: item[1]))

        return result, distmat[idxx2]


    @torch.no_grad()
    def get_embedding(self, exp):
        emb = self.model.get_embedding([exp])[0]
        return emb.cpu().numpy()


    def run(self):
        exp_list, emb_list = self.get_expression_embeddings()
        results_file = os.path.join(self.save_dir, f"emb_math_results_{datetime.now().strftime('%Y%m%d-%H%M%S%f')}.csv")
        results_header = ["x1", "y1", "y2", "expected_x2", "predicted_x2", "cosine_dist_predicted", "cosine_dist_expected"]

        results = []
        for example in self.test_examples:
            assert example["x2"] in exp_list, f"{example['x2']} is not present in the expression list."

            result, exp_dist = self.run_analogy(
                x1 = example["x1"],
                y1 = example["y1"],
                y2 = example["y2"],
                x2 = example["x2"],
                exp_list = exp_list,
                emb_list = emb_list,
            )
            nearest = min(result, key = result.get)
            results.append({
                "x1": example["x1"],
                "y1": example["y1"],
                "y2": example["y2"],
                "expected_x2": example["x2"],
                "predicted_x2": nearest,
                "cosine_dist_predicted": result[nearest],
                "cosine_dist_expected": exp_dist,
            })

        with open(results_file, "w") as f:
            writer = csv.DictWriter(f, fieldnames = results_header)
            writer.writeheader()
            for result in results:
                writer.writerow(result)

        print(f"Written results to {results_file}")
