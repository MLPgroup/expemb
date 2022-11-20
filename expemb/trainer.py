import os
import wandb
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .data import ExpressionTupleDataset, ExpressionDataset, FileNameDataset
from .model import ExpEmbTx, SemVecExpEmbTx
from .tokenizer import EquivExpTokenizer, SemVecTokenizer
from .args import *
from .constants import SEMVEC_SCORE_MODES


class TxTrainer:
    def __init__(self, model_args, train_args):
        outdir = train_args.save_dir
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        else:
            print(f"{outdir} already exists!")

        # Save/load command line arguments
        modelargsfile = os.path.join(outdir, "model_args.yaml")
        trainargsfile = os.path.join(outdir, "train_args.yaml")
        if not os.path.exists(modelargsfile):
            model_args.save(modelargsfile)
        else:
            print(f"Model arguments file exists. Ignoring the command line values and using the saved values.")
            model_args = TxModelArguments.load(modelargsfile, drop_extra_fields = True)
        if not os.path.exists(trainargsfile):
            train_args.save(trainargsfile)
        else:
            print(f"Training arguments file exists. Ignoring the command line values and using the saved values.")
            train_args = TrainingArguments.load(trainargsfile, drop_extra_fields = True)

        self.train_args = train_args
        self.model_args = model_args
        self.outdir = outdir

        # Fix seed if applicable
        if self.train_args.seed is not None:
            print(f"Setting random seed to {self.train_args.seed}")
            pl.utilities.seed.seed_everything(self.train_args.seed, workers = True)


    def get_project_name(self):
        return self.train_args.project_name if self.train_args.project_name is not None else "expembtx"


    def get_last_checkpoint_path(self):
        checkpoint_path = os.path.join(self.outdir, "saved_models/last.ckpt")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = None
        return checkpoint_path


    def create_tokenizer(self):
        return EquivExpTokenizer()


    def get_tokenizer(self, checkpoint_path):
        if checkpoint_path is None:
            tokenizer = self.create_tokenizer()
        else:
            tokenizer = torch.load(checkpoint_path)["tokenizer"]

        return tokenizer


    def get_train_dataset(self, tokenizer):
        return ExpressionTupleDataset(
            self.train_args.train_file,
            tokenizer = tokenizer,
            max_seq_len = self.train_args.max_seq_len,
            n_examples = self.train_args.max_train_examples,
        )


    def get_val_dataset(self, tokenizer):
        return ExpressionDataset(
            self.train_args.val_file,
            tokenizer = tokenizer,
            max_seq_len = self.train_args.max_seq_len,
            n_examples = self.train_args.max_val_examples,
        )


    def setup_wandb(self):
        wandbfile = os.path.join(self.outdir, "wandb.yaml")
        if os.path.exists(wandbfile):
            wandb_config = WandbConfig.load(wandbfile)
            print(f"Using {wandb_config} for logging.")
        else:
            wandb_config = WandbConfig(id = wandb.util.generate_id(), name = self.train_args.run_name)
            wandb_config.save(wandbfile)

        return pl.loggers.WandbLogger(
            name = wandb_config.name,
            save_dir = self.outdir,
            project = self.get_project_name(),
            log_model = False,
            id = wandb_config.id
        )


    def get_callbacks(self):
        # Checkpointing callback
        callbacks = []
        checkpoint_dir = os.path.join(self.outdir, "saved_models")
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor = "val/accuracy",
            mode = "max",
            dirpath = checkpoint_dir,
            save_last = True,
            save_on_train_epoch_end = True,
            filename = "best",
        )
        callbacks.append(checkpoint_callback)

        # Early stopping callback
        if self.train_args.early_stopping is not None:
            early_stopping_callback = pl.callbacks.EarlyStopping(
                monitor = "val/accuracy",
                mode = "max",
                min_delta = 0.0005,
                patience = self.train_args.early_stopping,
                verbose = True,
            )
            callbacks.append(early_stopping_callback)

        return callbacks


    def get_model(self, tokenizer):
        return ExpEmbTx(
            n_encoder_layers = self.model_args.n_encoder_layers,
            n_decoder_layers = self.model_args.n_decoder_layers,
            emb_size = self.model_args.emb_size,
            n_heads = self.model_args.n_heads,
            vocab_size = tokenizer.n_comp,
            dim_feedforward = self.model_args.dim_feedforward,
            dropout = self.model_args.dropout,
            max_seq_len = self.train_args.max_seq_len,
            optim = self.train_args.optim,
            lr = self.train_args.lr,
            tokenizer = tokenizer,
            norm_first = self.model_args.norm_first,
            max_out_len = self.train_args.max_out_len,
            sympy_timeout = self.train_args.sympy_timeout,
            is_bool_dataset = False,
            activation = self.model_args.activation,
            label_smoothing = self.train_args.label_smoothing,
            weight_decay = self.train_args.weight_decay,
        )


    def run_training(self):
        # If a checkpoint is saved on disk
        checkpoint_path = self.get_last_checkpoint_path()
        # Initialize dataloaders
        tokenizer = self.get_tokenizer(checkpoint_path)
        train_dataset = self.get_train_dataset(tokenizer)
        val_dataset = self.get_val_dataset(tokenizer)
        train_dataloder = DataLoader(train_dataset, batch_size = self.train_args.train_batch_size, collate_fn = train_dataset.collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size = self.train_args.val_batch_size, collate_fn = val_dataset.collate_fn)
        # Wandb
        logger = self.setup_wandb()
        # Checkpoint and early stopping callbacks
        callbacks = self.get_callbacks()
        # Define model
        model = self.get_model(tokenizer)
        trainer = pl.Trainer(
            accelerator = "auto",
            max_epochs = self.train_args.n_epochs,
            track_grad_norm = self.train_args.track_grad_norm,
            logger = logger,
            callbacks = callbacks,
            gradient_clip_val = self.train_args.grad_clip_val,
            gradient_clip_algorithm = self.train_args.grad_clip_algo,
            val_check_interval = 1.0,
            strategy = pl.strategies.ddp.DDPStrategy(find_unused_parameters = False),
            precision = self.train_args.precision,
            min_epochs = self.train_args.n_min_epochs,
            deterministic = True,
        )
        trainer.fit(model, train_dataloder, val_dataloader, ckpt_path = checkpoint_path)


class SemVecTxTrainer(TxTrainer):
    def get_project_name(self):
        return self.train_args.project_name if self.train_args.project_name is not None else "semvecexpembtx"


    def create_tokenizer(self):
        return SemVecTokenizer()


    def get_callbacks(self):
        callbacks = []
        checkpoint_dir = os.path.join(self.outdir, "saved_models")

        for mode in SEMVEC_SCORE_MODES:
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                monitor = f"val/score@5/{mode}",
                mode = "max",
                dirpath = checkpoint_dir,
                save_last = False,
                save_on_train_epoch_end = True,
                filename = f"best_{mode}",
            )
            callbacks.append(checkpoint_callback)

        last_checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor = f"step",
            mode = "max",
            dirpath = checkpoint_dir,
            save_last = False,
            save_on_train_epoch_end = True,
            filename = "last",
        )
        callbacks.append(last_checkpoint_callback)

        # Early stopping callback
        if self.train_args.early_stopping is not None:
            early_stopping_callback = pl.callbacks.EarlyStopping(
                monitor = "val/score@5/max",
                mode = "max",
                min_delta = 0.0005,
                patience = self.train_args.early_stopping,
                verbose = True,
            )
            callbacks.append(early_stopping_callback)

        return callbacks


    def get_val_dataset(self, tokenizer):
        return FileNameDataset(
            full_datafile = None,
            test_datafile = self.train_args.val_file,
        )


    def get_model(self, tokenizer):
        return SemVecExpEmbTx(
            n_encoder_layers = self.model_args.n_encoder_layers,
            n_decoder_layers = self.model_args.n_decoder_layers,
            emb_size = self.model_args.emb_size,
            n_heads = self.model_args.n_heads,
            vocab_size = tokenizer.n_comp,
            dim_feedforward = self.model_args.dim_feedforward,
            dropout = self.model_args.dropout,
            max_seq_len = self.train_args.max_seq_len,
            optim = self.train_args.optim,
            lr = self.train_args.lr,
            tokenizer = tokenizer,
            norm_first = self.model_args.norm_first,
            max_out_len = self.train_args.max_out_len,
            sympy_timeout = self.train_args.sympy_timeout,
            is_bool_dataset = self.train_args.is_bool_dataset,
            activation = self.model_args.activation,
            label_smoothing = self.train_args.label_smoothing,
            weight_decay = self.train_args.weight_decay,
        )
