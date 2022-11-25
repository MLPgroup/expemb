from dataclasses import dataclass
from typing import Optional, List
from simple_parsing import ArgumentParser, Serializable


@dataclass
class TxModelArguments(Serializable):
    """Model arguments"""
    emb_size: int = 512                          # Model dimension / embedding size.
    n_heads: int = 8                             # Number of attention heads.
    n_encoder_layers: int = 6                    # Number of encoder layers.
    n_decoder_layers: int = 6                    # Number of decoder layers.
    dim_feedforward: int = 2048                  # Dimension of the feedforward layers.
    dropout: float = 0.1                         # Dropout probability (0-1).
    norm_first: bool = True                      # If True, LayerNorms is performed before other attention and feedforward operations.
    activation: str = "relu"                     # The activation function for encoder/decoder intermediate layers (relu/gelu/geglu/swiglu).


@dataclass
class TrainingArguments(Serializable):
    """Training arguments"""
    train_file: str                              # Training file path.
    val_file: str                                # Validation file path.
    save_dir: str                                # Output directory for models.
    project_name: Optional[str] = None           # Project name for Wandb.
    max_seq_len: int = 512                       # Maximum sequence length. Expressions longer than this value will be skipped.
    max_out_len: int = 100                       # Maximum length to use during decoding.
    max_n_pos: int = 1024                        # Max length for positional embedding.
    max_train_examples: int = -1                 # Maximum number of training examples to read from the file.
    max_val_examples: int = -1                   # Maximum number of validation examples to read from the file.
    seed: Optional[int] = None                   # Random seed value.
    train_batch_size: int = 64                   # Batch size for training.
    val_batch_size: int = 64                     # Batch size for validation.
    lr: float = 0.0001                           # Learning rate for training.
    weight_decay: float = 0.0                    # Weight decay for the optimizer.
    optim: str = "Adam"                          # Optimizer for training.
    n_epochs: int = 20                           # Number of epochs.
    track_grad_norm: int = -1                    # Which grad norm to track.
    run_name: str = "test"                       # Run name.
    grad_clip_val: float = 0.0                   # Gradient clip threshold.
    grad_clip_algo: str = "norm"                 # Gradient clip algorithm.
    precision: int = 16                          # Precision to use for training.
    sympy_timeout: int = 2                       # Timeout for SymPy operations.
    early_stopping: Optional[int] = None         # If provided, the early stopping callback is enabled with patience as the value of this argument.
    semvec: bool = False                         # If job is being run for one of the SemVec datasets.
    n_min_epochs: Optional[int] = None           # Force training to atleast these many epochs (optionally to be used with early_stopping).
    bool_dataset: bool = False                   # If the dataset consists of boolean expressions.
    label_smoothing: float = 0.0                 # Label smoothing to use while computing loss.
    autoencoder: bool = False                    # If it should be run as autoencoder.


@dataclass
class TestingArguments(Serializable):
    """Testing arguments"""
    test_file: str                               # Test file path.
    save_dir: str                                # Output directory for results.
    full_file: Optional[str] = None              # Path to full dataset (only used in SemVec).
    beam_sizes: Optional[List[int]] = None       # Beam sizes to use for testing.
    sympy_timeout: Optional[float] = None        # Timeout for SymPy operations.
    batch_size: int = 64                         # Testing batch size.
    ckpt_name: str = "best"                      # Specify the ckpt name to test.
    max_seq_len: int = 512                       # Maximum sequence length. Expressions longer than this value will be skipped.
    max_test_examples: int = -1                  # Maximum number of examples to read from the file.
    precision: int = 16                          # Precision to use for training.
    semvec: bool = False                         # If job is being run for one of the SemVec datasets.


@dataclass
class EmbMathArguments(Serializable):
    """Embedding mathematics arguments"""
    train_file: str                              # File used for training the model.
    save_dir: str                                # Saved model directory. Results will also be stored in this directory.
    test_file: str                               # Test file path.
    model_cls: str = "ExpEmbTx"                  # Model class. Only ExmEmbTx is supported.
    ckpt_name: str = "best"                      # Specify the ckpt name to test.
    batch_size: int = 1024                       # Batch size to use while computing embedding.


@dataclass
class WandbConfig(Serializable):
    id: str
    name: str


@dataclass
class TestingResults(Serializable):
    args: TestingArguments
    accuracy: List[dict]
