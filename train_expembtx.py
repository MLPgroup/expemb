import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from expemb import (
    ExpressionTupleDataset,
    ExpEmbTx,
    EquivExpTokenizer,
)


def main():
    tokenizer = EquivExpTokenizer()
    train_dataset = ExpressionTupleDataset("./data/prim_fwd_5_ops.train.gz", tokenizer = tokenizer, max_seq_len = 512)
    train_dataloder = DataLoader(train_dataset, batch_size = 64, collate_fn = train_dataset.collate_fn)
    logger = pl.loggers.WandbLogger(name = "test", project = "expembtx")
    model = ExpEmbTx(
        vocab_size = tokenizer.n_comp,
        padding_idx = tokenizer.get_pad_index(),
    )
    trainer = pl.Trainer(
        accelerator = "auto",
        max_epochs = 20,
        track_grad_norm = 2,
        logger = logger,
    )
    trainer.fit(model, train_dataloder)


if __name__ == "__main__":
    main()