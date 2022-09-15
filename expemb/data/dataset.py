import os
import gzip
import torch
from torch.utils.data import Dataset
from ..tokenizer import Tokenizer


class ExpressionTupleDataset(Dataset):
    def __init__(self, filepath: str, tokenizer: Tokenizer, n_examples: int = -1, max_seq_len: int = -1):
        super(ExpressionTupleDataset, self).__init__()

        assert os.path.exists(filepath), f"{filepath} does not exist"
        self.filepath = filepath
        self.tokenizer = tokenizer

        print(f"Reading {n_examples} training examples from {self.filepath}")

        self.eq_tuples = []
        if filepath.endswith(".gz"):
            file = gzip.open(filepath, "rt")
        else:
            file = open(filepath, "r", encoding="utf-8")

        skipped = 0
        for idx, line in enumerate(file):
            if idx == n_examples:
                break

            line = line.strip()
            if "\t" in line:
                ip_eq = line.split("\t")[0]
                op_eq = line.split("\t")[1]
            else:
                ip_eq = line
                op_eq = line

            ip_len = len(ip_eq.split(" "))
            op_len = len(op_eq.split(" "))
            if max_seq_len == -1: 
                self.eq_tuples.append((ip_eq, op_eq))
            elif ip_len <= max_seq_len and op_len <= max_seq_len:
                self.eq_tuples.append((ip_eq, op_eq))
            else:
                skipped += 1

        print(f"Skipped {skipped} lines due to max sequence length restriction.")
        file.close()


    def __len__(self) -> int:
        return len(self.eq_tuples)


    def __getitem__(self, idx: int) -> tuple:
        eq_tuple = self.eq_tuples[idx]
        ip_tensor = self.tokenizer.encode(eq_tuple[0])
        op_tensor = self.tokenizer.encode(eq_tuple[1])
        return ip_tensor, op_tensor


    def collate_fn(self, sequences: tuple) -> tuple:
        ip_eqs, op_eqs = zip(*sequences)
        ip_eqs = self.batch_sequence(ip_eqs)
        op_eqs = self.batch_sequence(op_eqs)
        return ip_eqs, op_eqs


    def batch_sequence(self, sequences: tuple) -> tuple:
        lengths = [len(s) for s in sequences]
        sent = torch.LongTensor(max(lengths), len(lengths)).fill_(self.tokenizer.get_pad_index())

        for i, s in enumerate(sequences):
            sent[0:lengths[i], i] = s

        return sent
