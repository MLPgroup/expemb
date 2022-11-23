import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sympy as sp
from typing import Optional, List
from torch import Tensor
from .txdecoder import *
from .hypothesis import BeamHypotheses
from ..timeout import timeout
from ..tokenizer import Tokenizer

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

        self._reset_parameters()


    def _reset_parameters(self):
        """
        Reference: https://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf
        """
        nn.init.normal_(self.embedding.weight, mean = 0, std = self.emb_size ** -0.5)


    def forward(self, input: Tensor):
        return self.embedding(input.long()) * math.sqrt(self.emb_size)


class ExpEmbTx(pl.LightningModule):
    def __init__(self,
        tokenizer: Optional[Tokenizer] = None,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        emb_size: int = 512,
        n_heads: int = 8,
        vocab_size: int = 1000,
        dim_feedforward: int = 512,
        dropout: int = 0.1,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = True,
        max_seq_len: int = 1000,
        max_out_len: int = 100,
        optim: str = "Adam",
        lr: float = 0.0001,
        weight_decay: float = 0.0,
        beam_sizes: list = [],
        sympy_timeout: int = 2,
        bool_dataset: bool = False,
        activation: str = "relu",
        label_smoothing: float = 0.0,
        autoencoder: bool = False,
    ):
        super(ExpEmbTx, self).__init__()
        self.model_type = "ExpEmb-TX"
        self.emb_size = emb_size
        self.optim = optim
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_heads = n_heads
        self.tokenizer = tokenizer
        self.padding_idx = tokenizer.get_pad_index()
        self.soe_idx = tokenizer.get_soe_index()
        self.eoe_idx = tokenizer.get_eoe_index()
        self.max_seq_len = max_seq_len
        self.max_out_len = max_out_len
        self.vocab_size = vocab_size
        self.sympy_timeout = sympy_timeout
        self.bool_dataset = bool_dataset
        self.autoencoder = autoencoder
        # Only used in test_step
        self.beam_sizes = beam_sizes

        self.token_embedding = TokenEmbedding(vocab_size = vocab_size, emb_size = emb_size)
        self.positional_encoding = PositionalEncoding(emb_size = emb_size, dropout = dropout, maxlen = max_seq_len)
        decoder_layer = CausalTransformerDecoderLayer(
            d_model = emb_size,
            nhead = n_heads,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            norm_first = norm_first,
            layer_norm_eps = layer_norm_eps,
            batch_first = batch_first,
            activation = self._get_activation_fn(activation),
        )
        decoder = CausalTransformerDecoder(
            decoder_layer = decoder_layer,
            num_layers = n_decoder_layers,
            norm = nn.LayerNorm(emb_size, eps = layer_norm_eps),
        )
        self.transformer = nn.Transformer(
            d_model = emb_size,
            nhead = n_heads,
            num_encoder_layers = n_encoder_layers,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            norm_first = norm_first,
            custom_decoder = decoder,
            activation = self._get_activation_fn(activation),
        )
        self.generator = nn.Linear(emb_size, vocab_size)
        self.save_hyperparameters(ignore = ["tokenizer"])
        self.loss_fn = nn.CrossEntropyLoss(ignore_index = self.padding_idx, label_smoothing = label_smoothing)

        # Initialize parameters
        # self._reset_parameters()


    def _reset_parameters(self):
        """
        Initializes the parameters based on the following paper:
        https://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf

        1. nn.Transformer uses Xavier initialization by default.
        """
        # Initialize generator
        nn.init.xavier_uniform_(self.generator.weight)


    def _get_activation_fn(self, activation: str):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        elif activation == "silu":
            return F.silu
        else:
            raise Exception(f"{activation} is not supported.")


    def forward(self, batch: Tensor):
        """
        Parameters
        ----------
        batch: tuple
            Tuple of src, src_len, tgt, tgt_len with the following sizes:
            src: src_seq_len x batch_size
            tgt: tgt_seq_len x batch_size

        Returns
        -------
        Tensor of shape tgt_seq_len x batch_size x vocab_size
        """
        src, tgt = batch
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]
        src_mask, src_padding_mask = self.create_src_mask(src)
        tgt_mask, tgt_padding_mask = self.create_tgt_mask(tgt_input)

        src_emb = self.positional_encoding(self.token_embedding(src))
        tgt_emb = self.positional_encoding(self.token_embedding(tgt_input))
        output = self.transformer(
            src = src_emb,
            tgt = tgt_emb,
            src_mask = src_mask,
            tgt_mask = tgt_mask,
            src_key_padding_mask = src_padding_mask,
            tgt_key_padding_mask = tgt_padding_mask,
            memory_key_padding_mask = src_padding_mask,
        )
        logits = self.generator(output)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        return logits, loss


    def get_embedding(self, exp_list: List[str], mode: str = "max"):
        """
        Parameters
        ----------
        exp_list: List[str]
            List of expressions in prefix notation
        mode: str
            max/mean/decoder

        Returns
        -------
        Tensor of size len(exp_list) x emb_size
        """
        encoded = []
        lens = []
        for exp in exp_list:
            tensor = self.tokenizer.encode(exp)
            encoded.append(tensor)
            lens.append(tensor.size(0))

        src = torch.empty(max(lens), len(lens), dtype = torch.long, device = self.device).fill_(self.padding_idx)
        emb_mask = torch.zeros(max(lens), len(lens), dtype = torch.bool, device = self.device)
        for idx, curlen in enumerate(lens):
            src[:curlen, idx] = encoded[idx]
            emb_mask[1:curlen - 1, idx] = True

        src_mask, src_padding_mask = self.create_src_mask(src)
        memory = self.encode(src, src_mask, src_padding_mask)
        if mode == "decoder":
            tgt = torch.empty(1, 1, dtype = torch.long).fill_(self.soe_idx).to(self.device)
            tgt_mask, tgt_padding_mask = self.create_tgt_mask(tgt)
            logits = self.decode(tgt, memory, tgt_mask, tgt_padding_mask, src_padding_mask)
        else:
            logits = None

        if mode == "mean":
            memory[~emb_mask] = 0.
            embedding = memory.sum(dim = 0, keepdim = False)
            size = emb_mask.sum(dim = 0, keepdim = True)
            embedding = embedding / size.T
        elif mode == "max":
            memory[~emb_mask] = float("-inf")
            embedding, _ = memory.max(dim = 0, keepdim = False)
        elif mode == "decoder":
            embedding = logits[0]
        else:
            raise Exception(f"{self.mode} is not supported.")

        return embedding


    def create_src_mask(self, src):
        """
        Parameters
        ----------
        src: Tensor
            Tensor of size src_seq_len x batch_size

        Returns
        -------
        Tuple of src_mask, src_padding_mask where:
        src_mask: Tensor
            Tensor of src_seq_len x src_seq_len
        src_padding_mask: Tensor
            Tensor of size batch_size x src_seq_len
        """
        src_seq_len = src.shape[0]
        src_mask = torch.zeros((src_seq_len, src_seq_len), dtype=torch.bool).to(self.device)
        src_padding_mask = (src == self.padding_idx).transpose(0, 1)
        return src_mask, src_padding_mask


    def create_tgt_mask(self, tgt):
        """
        Parameters
        ----------
        tgt: Tensor
            Tensor of size tgt_seq_len x batch_size

        Returns
        -------
        Tuple of tgt_mask, tgt_padding_mask where:
        tgt_mask: Tensor
            Tensor of tgt_seq_len x tgt_seq_len
        tgt_padding_mask: Tensor
            Tensor of size batch_size x tgt_seq_len
        """
        tgt_seq_len = tgt.shape[0]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(self.device)
        tgt_padding_mask = (tgt == self.padding_idx).transpose(0, 1)
        return tgt_mask, tgt_padding_mask


    def configure_optimizers(self):
        if self.optim == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        elif self.optim == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        else:
            raise Exception(f"{self.optim} optimizer is not supported.")

        return optimizer


    def training_step(self, batch: tuple, batch_idx: int):
        src, _ = batch
        _, loss = self.forward(batch)
        self.log("train/loss", loss, batch_size = src.size(1), sync_dist = True)
        return loss


    def validation_step(self, batch: tuple, batch_idx: int):
        src, _ = batch
        accuracy = self.compute_accuracy(batch)
        self.log("val/accuracy", accuracy, on_epoch = True, batch_size = src.size(1), sync_dist = True)
        return accuracy


    def test_step(self, batch: tuple, batch_idx: int):
        src, _ = batch
        accuracy_dict = {}
        for beam_size in self.beam_sizes:
            accuracy = self.compute_accuracy_beam(batch, beam_size)
            self.log(f"test/accuracy_{beam_size}", accuracy, on_epoch = True, batch_size = src.size(1), sync_dist = True)
            accuracy_dict[beam_size] = accuracy

        return accuracy


    def compute_accuracy(self, batch: Tensor):
        src, src_exps = batch
        batch_size = len(src_exps)
        predicted = self.generate(src)
        correct = 0
        for idx in range(batch_size):
            try:
                src_prefix = src_exps[idx]
                src_sp = self.prefix_to_sympy(src_prefix)
                tensor = predicted[:, idx]
                predicted_prefix = self.tokenizer.decode(tensor, True)
                predicted_prefix = " ".join(predicted_prefix.split(" ")[1:-1])
                predicted_sp = self.prefix_to_sympy(predicted_prefix)
                if self.autoencoder:
                    equivalent = src_prefix == predicted_prefix
                else:
                    equivalent = src_prefix != predicted_prefix and self.are_equivalent(src_sp, predicted_sp)
                if equivalent:
                    correct += 1
            except Exception as e:
                continue

        accuracy = correct / batch_size
        return accuracy


    def compute_accuracy_beam(self, batch: Tensor, beam_size: int):
        src, src_exps = batch
        batch_size = len(src_exps)
        predicted, _ = self.generate_beam(src, beam_size)
        correct = 0
        for idx in range(batch_size):
            try:
                src_prefix = src_exps[idx]
                src_spexp = self.prefix_to_sympy(src_prefix)
                decoded_list = predicted[:, idx]
                predicted_prefix_list = [self.tokenizer.decode(decoded_list[:, _], True) for _ in range(beam_size)]
                predicted_prefix_list = [" ".join(_.split(" ")[1:-1]) for _ in predicted_prefix_list]
                equivalent = False
                for predicted_prefix in predicted_prefix_list:
                    try:
                        predicted_spexp = self.prefix_to_sympy(predicted_prefix)
                        if self.autoencoder:
                            equivalent = equivalent or (src_prefix == predicted_prefix)
                        else:
                            equivalent = equivalent or (src_prefix != predicted_prefix and self.are_equivalent(src_spexp, predicted_spexp))

                        if equivalent:
                            break
                    except Exception as e:
                        continue
                if equivalent:
                    correct += 1
            except Exception as e:
                continue

        accuracy = correct / batch_size
        return accuracy


    def encode(self, src: Tensor, src_mask: Tensor, src_padding_mask: Tensor):
        return self.transformer.encoder(
            src = self.positional_encoding(self.token_embedding(src)),
            mask = src_mask,
            src_key_padding_mask = src_padding_mask,
        )


    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
        cache: Optional[Tensor] = None
    ):
        return self.transformer.decoder(
            tgt = self.positional_encoding(self.token_embedding(tgt)),
            memory = memory,
            tgt_mask = tgt_mask,
            tgt_key_padding_mask = tgt_padding_mask,
            memory_key_padding_mask = memory_key_padding_mask,
            cache = cache,
        )


    def generate(self, src: Tensor, use_cache: bool = True):
        """
        Parameters
        ----------
        src: Tensor
            Tensor of size src_seq_len x batch_size
        use_cache: bool
            If caching should be used for decoding

        Returns
        -------
        Tensor of size max_out_len x batch_size
        """
        batch_size = src.size(1)
        src_mask, src_padding_mask = self.create_src_mask(src)
        memory = self.encode(src, src_mask, src_padding_mask)
        tgt = torch.empty(self.max_out_len, batch_size, dtype = torch.long).fill_(self.padding_idx).to(self.device)
        tgt[0, :].fill_(self.soe_idx)
        done = tgt.new_zeros(batch_size, dtype = torch.bool)

        cache = {} if use_cache else None
        maxlen = 0
        for idx in range(self.max_out_len - 1):
            tgt_input = tgt[:idx + 1, :]
            tgt_mask, tgt_padding_mask = self.create_tgt_mask(tgt_input)
            logits = self.decode(
                tgt = tgt_input,
                memory = memory,
                tgt_mask = tgt_mask,
                tgt_padding_mask = tgt_padding_mask,
                memory_key_padding_mask = src_padding_mask,
                cache = cache,
            )
            logits = logits[-1, :, :]
            logits = self.generator(logits)
            output = logits.argmax(dim = -1)
            # Only update indices that are not done yet
            tgt[idx + 1, ~done] = output[~done]
            done[~done] = output[~done] == self.eoe_idx
            maxlen += 1
            if all(done):
                break

        return tgt[:maxlen + 1]


    def generate_beam(self, src: Tensor, beam_size: int, length_penalty: float = 1.0, early_stopping: bool = True):
        """
        Parameters
        ----------
        src: Tensor
            Tensor of size src_seq_len x batch_size
        beam_size: int
            Beam size

        Output
        ------
        Tuple of (decoded, scores) where:
        decoded: Tensor
            Tensor of size seq_len x batch_size x beam_size
        scores: Tensor
            Tensor of size batch_size x beam_size
        """
        src_seq_len, batch_size = src.size()
        src_mask, src_padding_mask = self.create_src_mask(src)
        assert src_mask.size() == (src_seq_len, src_seq_len)
        assert src_padding_mask.size() == (batch_size, src_seq_len)

        memory = self.encode(src, src_mask, src_padding_mask)
        assert memory.size() == (src_seq_len, batch_size, self.emb_size)
        src_seq_len, batch_size, emb_size = memory.size()

        memory = memory.unsqueeze(2).expand((src_seq_len, batch_size, beam_size, emb_size))
        assert memory.size() == (src_seq_len, batch_size, beam_size, emb_size)
        memory = memory.contiguous().view(src_seq_len, batch_size * beam_size, emb_size)
        memory_key_padding_mask = src_padding_mask.unsqueeze(1).expand((batch_size, beam_size, src_seq_len))
        assert memory_key_padding_mask.size() == (batch_size, beam_size, src_seq_len)
        memory_key_padding_mask = memory_key_padding_mask.contiguous().view(batch_size * beam_size, src_seq_len)

        # Generated decoder output
        generated = src.new(self.max_out_len, batch_size * beam_size).fill_(self.padding_idx)
        generated[0].fill_(self.soe_idx)

        # Generated hypotheses
        generated_hyps = [BeamHypotheses(beam_size, self.max_out_len, length_penalty, early_stopping) for _ in range(batch_size)]

        # Scores for each expression in the beam
        beam_scores = memory.new(batch_size, beam_size).fill_(0)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # Current position
        cur_len = 1

        # Cache states
        cache = {}

        # Done inputs
        done = [False for _ in range(batch_size)]

        while cur_len < self.max_out_len:
            tgt = generated[:cur_len]
            tgt_mask, tgt_padding_mask = self.create_tgt_mask(tgt)
            assert tgt.size() == (cur_len, batch_size * beam_size)
            assert tgt_mask.size() == (cur_len, cur_len)
            assert tgt_padding_mask.size() == (batch_size * beam_size, cur_len)

            logits = self.decode(tgt, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask, cache)  # cur_len x batch_size * beam_size x emb_size
            assert logits.size() == (cur_len, beam_size * batch_size, self.emb_size)
            logits = logits[-1, :, :] # batch_size * beam_size x emb_size
            logits = self.generator(logits)  # batch_size * beam_size x vocab_size
            scores = F.log_softmax(logits, dim = -1)
            assert scores.size() == (batch_size * beam_size, self.vocab_size)

            scores = scores + beam_scores[:, None].expand_as(scores)
            scores = scores.view(batch_size, beam_size * self.vocab_size)

            next_scores, next_words = torch.topk(scores, 2 * beam_size, dim = 1, largest = True, sorted = True)
            assert next_scores.size() == next_words.size() == (batch_size, 2 * beam_size)

            next_batch_beam = []

            for input_id in range(batch_size):
                done[input_id] = done[input_id] or generated_hyps[input_id].is_done(next_scores[input_id].max().item())
                if done[input_id]:
                    next_batch_beam.extend([(0, self.padding_idx, 0)] * beam_size)
                    continue

                next_input_beam = []
                for idx, val in zip(next_words[input_id], next_scores[input_id]):
                    beam_id = torch.div(idx, self.vocab_size, rounding_mode = "floor")
                    token_id = idx % self.vocab_size

                    if token_id == self.eoe_idx or cur_len + 1 == self.max_out_len:
                        generated_hyps[input_id].add(generated[:cur_len, input_id * beam_size + beam_id].clone().cpu(), val.item())
                    else:
                        next_input_beam.append((val, token_id, input_id * beam_size + beam_id))

                    if len(next_input_beam) == beam_size:
                        break

                assert len(next_input_beam) == 0 if cur_len + 1 == self.max_out_len else beam_size
                if len(next_input_beam) == 0:
                    next_input_beam = [(0, self.padding_idx, 0)] * beam_size
                next_batch_beam.extend(next_input_beam)
                assert len(next_batch_beam) == beam_size * (input_id + 1)

            assert len(next_batch_beam) == batch_size * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src.new([x[2] for x in next_batch_beam])

            generated = generated[:, beam_idx]
            generated[cur_len] = beam_tokens
            if "cache" in cache:
                # Cache size is n_decoder_layers x cur_len x beam_size * batch_size x vocab_size
                cache["cache"] = cache["cache"][:, :, beam_idx]

            cur_len += 1

            if all(done):
                break

        batch_hypotheses = src.new(cur_len, batch_size, beam_size).fill_(self.padding_idx)
        batch_scores = memory.new(batch_size, beam_size).fill_(0)
        for b_idx, hyp in enumerate(generated_hyps):
            sorted_hyps = sorted(hyp.hyp, key=lambda x: x[0])
            for h_idx, hyp in enumerate(sorted_hyps):
                batch_scores[b_idx, h_idx] = hyp[0]
                batch_hypotheses[:len(hyp[1]), b_idx, h_idx] = hyp[1]
                batch_hypotheses[len(hyp[1]), b_idx, h_idx] = self.eoe_idx

        return batch_hypotheses, batch_scores


    def prefix_to_sympy(self, prefix):
        assert self.sympy_timeout > 0

        @timeout(self.sympy_timeout)
        def _prefix_to_sympy(prefix):
            return self.tokenizer.prefix_to_sympy(prefix, evaluate=False)

        return _prefix_to_sympy(prefix)


    def are_equivalent(self, exp1, exp2):
        assert self.sympy_timeout > 0

        @timeout(self.sympy_timeout)
        def _are_equivalent_poly(exp1, exp2):
            return sp.simplify(exp1 - exp2) == 0

        def _are_equivalent_bool(exp1, exp2):
            return not sp.logic.boolalg.simplify_logic(exp1 ^ exp2)

        return (self.bool_dataset and _are_equivalent_bool(exp1, exp2)) or \
            (not self.bool_dataset and _are_equivalent_poly(exp1, exp2))


    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint["tokenizer"] = self.tokenizer
