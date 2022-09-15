import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import Tensor

# Reference: https://pytorch.org/tutorials/beginner/translation_transformer.html

class PositionalEncoding(nn.Module):
    def __init__(self,
        emb_size: int,
        dropout: float,
        maxlen: int = 5000
    ):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)


    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size


    def forward(self, input: Tensor):
        return self.embedding(input.long()) * math.sqrt(self.emb_size)


class ExpEmbTx(pl.LightningModule):
    def __init__(self,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        emb_size: int = 512,
        n_heads: int = 8,
        vocab_size: int = 1000,
        dim_feedforward: int = 512,
        dropout: int = 0.1,
        max_seq_len: int = 1000,
        optim: str = "Adam",
        lr: float = 0.0001,
        padding_idx: int = 2,
    ):
        super(ExpEmbTx, self).__init__()
        self.model_type = "ExpEmb-TX"
        self.emb_size = emb_size
        self.optim = optim
        self.lr = lr
        self.n_heads = n_heads
        self.padding_idx = padding_idx

        self.token_embedding = TokenEmbedding(vocab_size=vocab_size, emb_size=emb_size)
        self.positional_encoding = PositionalEncoding(emb_size=emb_size, dropout=dropout, maxlen=max_seq_len)
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(emb_size, vocab_size)
        self.save_hyperparameters()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index = self.padding_idx)


    def forward(self, batch: Tensor):
        """
        Parameters
        ----------
        batch: tuple
            Tuple of src, src_len, tgt, tgt_len with the following sizes:
            src: src_seq_len x batch_size
            src_len: batch_size
            tgt: tgt_seq_len x batch_size
            tgt_len: batch_size

        Returns
        -------
        Tensor of shape tgt_seq_len x batch_size x vocab_size
        """
        src, tgt = batch
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt_input)

        src_emb = self.positional_encoding(self.token_embedding(src))
        tgt_emb = self.positional_encoding(self.token_embedding(tgt_input))
        output = self.transformer(
            src = src_emb,
            tgt = tgt_emb,
            src_mask = src_mask,
            tgt_mask = tgt_mask,
            src_key_padding_mask = src_padding_mask,
            tgt_key_padding_mask = tgt_padding_mask,
            memory_key_padding_mask = src_padding_mask
        )
        logits = self.generator(output)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        return logits, loss


    def create_mask(self, src, tgt):
        """
        Parameters
        ----------
        src: Tensor
            Tensor of size src_seq_len x batch_size
        tgt: Tensor
            Tensor of size tgt_seq_len x batch_size

        Returns
        -------
        Tuple of src_mask, tgt_mask, src_padding_mask, tgt_padding_mask where:
        src_mask: Tensor
            Tensor of src_seq_len x src_seq_len
        tgt_mask: Tensor
            Tensor of tgt_seq_len x tgt_seq_len
        src_padding_mask: Tensor
            Tensor of size batch_size x src_seq_len
        tgt_padding_mask: Tensor
            Tensor of size batch_size x tgt_seq_len
        """
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(self.device)
        src_mask = torch.zeros((src_seq_len, src_seq_len), dtype=torch.bool).to(self.device)
        src_padding_mask = (src == self.padding_idx).transpose(0, 1)
        tgt_padding_mask = (tgt == self.padding_idx).transpose(0, 1)

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


    def configure_optimizers(self):
        if self.optim == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        else:
            raise Exception(f"{self.optim} optimizer is not supported.")

        return optimizer


    def training_step(self, batch: Tensor, batch_idx: int):
        _, loss = self.forward(batch)
        self.log("train/loss", loss)
        return loss


    def validation_step(self, batch: Tensor, batch_idx: int):
        _, loss = self.forward(batch)
        self.log("val/loss", loss, on_epoch = True)
        return loss
