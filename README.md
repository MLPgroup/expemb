# Semantic Representations of Mathematical Expressions in a Continuous Vector Space

## Environment Setup

Setup the environment using `conda` as follows:
```
conda env create -n expembtx -f environment.yml
```

## Training and Evaluation
### Setup
To run the training and evaluation pipeline in this repository, [eqnet](https://github.com/mast-group/eqnet/) is required. As it can not be installed as a dependency, clone this repository and add it to `PYTHONPATH`.
```
EQNET_PATH=/tmp/eqnet
git clone https://github.com/mast-group/eqnet.git $EQNET_PATH
export PYTHONPATH=$PYTHONPATH:$EQNET_PATH
```

### Training
To train ExpEmb on the Equivalent Expressions Dataset or the SemVec datasets, `train_expembtx.py` may be used.

Example:
```
python train_expembtx.py \
    --train_file <TRAIN_FILE> \
    --val_file <VAL_FILE> \
    --n_epochs 100 \
    --norm_first True \
    --optim Adam \
    --weight_decay 0 \
    --lr 0.0001 \
    --train_batch_size 128 \
    --run_name <RUN_NAME> \
    --val_batch_size 256 \
    --grad_clip_val 1 \
    --max_out_len 256 \
    --precision 16 \
    --save_dir <OUT_DIR> \
    --early_stopping 5 \
    --n_min_epochs 10 \
    --label_smoothing 0.1 \
    --seed 42
```

Add `--semvec` option to the above-mentioned command for the SemVec datasets.

For all supported options, use `python train_expembtx.py --help` or refer to [TrainingAgruments](expemb/args.py#TestingArguments).

### Evaluation
To evaluate a trained model, `test_expembtx.py` may be used.

Example:
```
python test_expembtx.py \
    --test_file <TEST_FILE> \
    --save_dir <SAVED_MODEL_DIR> \
    --beam_sizes 1 10 50 \
    --max_seq_len 256 \
    --result_file_prefix <RESULT_FILE_PREFIX> \
    --batch_size 32
```

For all supported options, use `python test_expembtx.py --help` or refer to [TestingArguments](expemb/args.py#TestingArguments).

## Embedding Mathematics
`run_embmath.py` may be used to generate embedding mathematics results.

Example:
```
python run_embmath.py \
    --train_file <TRAIN_FILE> \
    --save_dir <SAVED_MODEL_DIR> \
    --test_file <EMB_MATH_TEST_FILE>
```

For all supported options, use `python run_embmath.py --help` or refer to [EmbMathAgruments](expemb/args.py#EmbMathAgruments).

## Distance Analysis
`run_distance_analysis` may be used to run distance analysis on a trained model.

Example:
```
python run_distance_analysis.py \
    --train_file <TRAIN_FILE> \
    --save_dir <SAVED_MODEL_DIR> \
    --test_file <DIST_ANALYSIS_TEST_FILE>
```

For all supported options, use `python run_embmath.py --help` or refer to [DistanceAnalysisArguments](expemb/args.py#DistanceAnalysisArguments).

## Embedding Plots
For embedding plots, refer to [embedding_plots.ipynb](notebooks/embedding_plots.ipynb).

## Wandb Integration
This repository supports wandb integration. To start using it, login to wandb using `wandb login`. To disable wandb, set the environment variable `WANDB_MODE=offline`.