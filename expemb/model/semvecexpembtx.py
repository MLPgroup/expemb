import torch
import torch.nn.functional as F
from torch import Tensor
from encoders.evaluation.knnstats import SemanticEquivalentDistanceEvaluation
from encoders.baseencoder import AbstractEncoder
from .expembtx import ExpEmbTx
from ..constants import SEMVEC_SCORE_MODES


class SemVecExpEmbTx(ExpEmbTx):
    def validation_step(self, batch: Tensor, batch_idx: int):
        _, val_filename = batch[0]
        score_dict = {}
        for mode in SEMVEC_SCORE_MODES:
            encoder = SemVecTxEncoder(transformer = self, device = self.device, mode = mode)
            evaluator = SemanticEquivalentDistanceEvaluation(encoder = encoder, encoder_filename = None)
            scores = evaluator.evaluate(data_filename = val_filename, num_nns = 5)
            assert len(scores) == 5
            score = scores[-1]
            self.log(f"val/score@5/{mode}", score, batch_size = 1, sync_dist = True)
            score_dict[mode] = score

        return score_dict


    def test_step(self, batch: Tensor, batch_idx: int):
        data_filename, test_filename = batch[0]
        all_scores = []
        for mode in SEMVEC_SCORE_MODES:
            encoder = SemVecTxEncoder(transformer = self, device = self.device, mode = mode)
            evaluator = SemanticEquivalentDistanceEvaluation(encoder = encoder, encoder_filename = None)
            scores = evaluator.evaluate_with_test(data_filename = data_filename, test_filename = test_filename, num_nns = 15)
            assert len(scores) == 15
            self.log(f"val/score@5/{mode}", scores[4], batch_size = 1, sync_dist = True)
            self.log(f"val/score@10/{mode}", scores[9], batch_size = 1, sync_dist = True)
            self.log(f"val/score@15/{mode}", scores[14], batch_size = 1, sync_dist = True)
            all_scores.extend(scores)

        return all_scores


class SemVecTxEncoder(AbstractEncoder):
    def __init__(self, transformer: SemVecExpEmbTx, device: torch.device, mode: str):
        self.transformer = transformer
        self.device = device
        self.mode = mode


    def get_encoding(self, data: tuple):
        prefix_eq = self.get_prefix_notation(data[1])
        embedding = self.transformer.get_embedding([prefix_eq], self.mode)
        embedding = embedding[0]
        return embedding.detach().cpu().numpy()


    def get_prefix_notation(self, tree):
        preorder = [node.name.lower() for node in tree]
        prefix = " ".join(preorder[1:])
        return prefix
