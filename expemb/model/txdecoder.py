import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor


# This is a modified version of
# https://github.com/alexmt-scale/causal-transformer-decoder/blob/e79413205b0301893218aaeeadaf79b25d7667ad/causal_transformer_decoder/model.py

class CausalTransformerDecoder(nn.TransformerDecoder):
    """Implementation of a transformer decoder based on torch implementation but
    more efficient. The difference is that it doesn't need to recompute the
    embeddings of all the past decoded tokens but instead uses a cache to
    store them. This makes use of the fact that the attention of a decoder is
    causal, so new predicted tokens don't affect the old tokens' embedding bc
    the corresponding attention cells are masked.
    The complexity goes from seq_len^3 to seq_len^2.

    This only happens in eval mode.
    In training mode, teacher forcing makes these optimizations unnecessary. Hence the
    Decoder acts like a regular nn.TransformerDecoder (except that the attention tgt
    masks are handled for you).
    """
    def __init__(self, **kwargs):
        decoder_layer = kwargs.get("decoder_layer", None)
        assert isinstance(decoder_layer, CausalTransformerDecoderLayer), f"decoder_layer must be of type CausalTransformerDecoderLayer"
        super(CausalTransformerDecoder, self).__init__(**kwargs)


    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        cache: Optional[dict] = None,
    ) -> Tensor:
        """
        Args:
            tgt (Tensor): current_len_output x bsz x hidden_dim
            memory (Tensor): len_encoded_seq x bsz x hidden_dim
            cache (Optional[Tensor]):
                n_layers x (current_len_output - 1) x bsz x hidden_dim
                If current_len_output == 1, nothing is cached yet, so cache
                should be None. Same if the module is in training mode.
            others (Optional[Tensor]): see official documentations
        Returns:
            output (Tensor): current_len_output x bsz x hidden_dim
            cache (Optional[Tensor]): n_layers x current_len_output x bsz x hidden_dim
                Only returns it when module is in eval mode (no caching in training)
        """
        assert not self.training or cache is None, f"cache parameter should be None in the training mode"

        output = tgt

        if cache is None:
            for mod in self.layers:
                output = mod(
                    tgt = output,
                    memory = memory,
                    tgt_mask = tgt_mask,
                    memory_mask = memory_mask,
                    tgt_key_padding_mask = tgt_key_padding_mask,
                    memory_key_padding_mask = memory_key_padding_mask,
                )
        else:
            new_token_cache = []
            for i, mod in enumerate(self.layers):
                output = mod(
                    tgt = output,
                    memory = memory,
                    tgt_mask = tgt_mask,
                    memory_mask = memory_mask,
                    tgt_key_padding_mask = tgt_key_padding_mask,
                    memory_key_padding_mask = memory_key_padding_mask,
                    last_only = True
                )
                new_token_cache.append(output)
                if "cache" in cache:
                    output = torch.cat([cache["cache"][i], output], dim=0)

            if "cache" in cache:
                cache["cache"] = torch.cat([cache["cache"], torch.stack(new_token_cache, dim=0)], dim=1)
            else:
                cache["cache"] = torch.stack(new_token_cache, dim=0)

        if self.norm is not None:
            output = self.norm(output)

        return output


class CausalTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        last_only: bool = False,
    ) -> Tensor:
        """
        Args:
            see CausalTransformerDecoder
        Returns:
            Tensor:
                If training: embedding of the whole layer: seq_len x bsz x hidden_dim
                If eval mode: embedding of last token: 1 x bsz x hidden_dim
        """
        assert not self.training or not last_only, f"last_only parameter should be True only for the non-training mode"

        if not last_only:
            return super().forward(
                tgt = tgt,
                memory = memory,
                tgt_mask = tgt_mask,
                memory_mask = memory_mask,
                tgt_key_padding_mask = tgt_key_padding_mask,
                memory_key_padding_mask = memory_key_padding_mask,
            )

        # This part is the modified version of the official PyTorch implementation.
        # See https://github.com/pytorch/pytorch/blob/v1.12.1/torch/nn/modules/transformer.py

        x = tgt
        x_last_tok = x[-1:, :, :]

        if self.norm_first:
            x_last_tok = x_last_tok + self._sa_block_causal(self.norm1(x), key_padding_mask = tgt_key_padding_mask)
            x_last_tok = x_last_tok + self._mha_block(self.norm2(x_last_tok), mem = memory, attn_mask = memory_mask, key_padding_mask = memory_key_padding_mask)
            x_last_tok = x_last_tok + self._ff_block(self.norm3(x_last_tok))
        else:
            x_last_tok = self.norm1(x_last_tok + self._sa_block_causal(x, key_padding_mask = tgt_key_padding_mask))
            x_last_tok = self.norm2(x_last_tok + self._mha_block(x_last_tok, mem = memory, attn_mask = memory_mask, key_padding_mask = memory_key_padding_mask))
            x_last_tok = self.norm3(x_last_tok + self._ff_block(x_last_tok))

        return x_last_tok


    # self-attention block
    def _sa_block_causal(self, x: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        x_last_tok = x[-1:, :, :]
        x_last_tok = self.self_attn(x_last_tok, x, x, attn_mask = None, key_padding_mask = key_padding_mask, need_weights = False)[0]
        return self.dropout1(x_last_tok)
