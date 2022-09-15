import torch
from torch import Tensor
from abc import ABC, abstractmethod
from .constants import *


class Tokenizer(ABC):
    @abstractmethod
    def get_index(self, comp: str) -> int:
        pass

    
    @abstractmethod
    def get_comp(self, idx: int) -> str:
        pass


    def get_pad_token(self) -> str:
        return "PAD"


    def get_pad_index(self) -> str:
        return self.get_index(self.get_pad_token())


    def encode(self, exp: str) -> Tensor:
        indexes = [self.get_index("SOE")] + [self.get_index(comp) for comp in exp.split(" ")] + [self.get_index("EOE")]
        return torch.LongTensor(indexes).view(-1)


    def decode(self, encoded: Tensor) -> str:
        assert encoded.ndim == 1, f"1-D tensors are allowed."
        exp_arr = []
        for elem in encoded:
            exp_arr.append(self.get_comp(elem.item()))

        return " ".join(exp_arr)


class EquivExpTokenizer(Tokenizer):
    def __init__(self):
        super(EquivExpTokenizer, self).__init__()
        self.special_words = SPECIAL_WORDS
        self.int_base = INT_BASE
        self.operators = OPERATORS
        self.coefficients = COEFFICIENTS
        self.variables = VARIABLES
        self.symbols = SYMBOLS
        self.constants = CONSTANTS
        self.elements = [str(i) for i in range(abs(INT_BASE))]

        self.components = self.special_words + self.constants + list(self.variables.keys()) + list(self.operators.keys()) + self.symbols + self.elements
        self.comp2index = {comp : idx for idx, comp in enumerate(self.components)}
        self.index2comp = {idx : comp for comp, idx in self.comp2index.items()}
        self.n_comp = len(self.comp2index)


    def get_index(self, comp: str) -> int:
        return self.comp2index[comp]


    def get_comp(self, idx: int) -> str:
        return self.index2comp[idx]


class SemVecTokenizer(Tokenizer):
    def __init__(self):
        super(SemVecTokenizer, self).__init__()
        self.special_words = SPECIAL_WORDS
        self.operators = SEMVEC_OPERATORS
        self.variables = SEMVEC_VARIABLES
        self.constants = SEMVEC_CONSTANTS
        self.components = self.special_words + self.constants + list(self.variables.keys()) + list(self.operators.keys())
        self.comp2index = {comp : idx for idx, comp in enumerate(self.components)}
        self.index2comp = {idx : comp for comp, idx in self.comp2index.items()}
        self.n_comp = len(self.comp2index)


    def get_index(self, comp: str) -> int:
        return self.comp2index[comp]


    def get_comp(self, idx: int) -> str:
        return self.index2comp[idx]
