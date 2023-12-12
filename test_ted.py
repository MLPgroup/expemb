import os
import json
from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from expemb.ted import TreeEditDistanceCalculator
from expemb import TrainingArguments


def validate_expemb(filepath, is_autoencoder):
    dirpath = Path(filepath).parent.absolute()
    args_file = os.path.join(dirpath, "train_args.yaml")
    train_args = TrainingArguments.load_yaml(args_file)

    if is_autoencoder:
        assert train_args.autoencoder
    else:
        assert not train_args.autoencoder


def dict_default_val():
    return {"expembe": 0, "expemba": 0, "tie": 0}


if __name__ == "__main__":
    parser = ArgumentParser("Test tree edit distance")
    parser.add_argument("--expembe", type = str, required = True)
    parser.add_argument("--expemba", type = str, required = True)
    parser.add_argument("--allow_same_group_replacement", action = "store_true", default = False)
    parser.add_argument("--remove_const_mul_add", action = "store_true", default = False)
    parser.add_argument("--result_file", type = str, required = True)
    parser.add_argument("--simplify", action = "store_true", default = False)
    args = parser.parse_args()

    validate_expemb(args.expembe, is_autoencoder = False)
    validate_expemb(args.expemba, is_autoencoder = True)

    with open(args.expembe) as f:
        expembe = json.load(f)

    with open(args.expemba) as f:
        expemba = json.load(f)

    common_frms = set(expembe.keys()).intersection(set(expemba.keys()))
    print(f"Total number of keys: {len(common_frms)}")

    calculator = TreeEditDistanceCalculator(
        allow_same_group_replacement = args.allow_same_group_replacement,
        remove_const_mul_add = args.remove_const_mul_add,
        simplify = args.simplify,
    )

    idx2stat = defaultdict(dict_default_val)
    all_results = []
    for frm in tqdm(common_frms):
        for idx, (expembe_to, expemba_to) in enumerate(zip(expembe[frm], expemba[frm])):
            try:
                expembe_dist = calculator.tree_distance(frm, expembe_to)
                expemba_dist = calculator.tree_distance(frm, expemba_to)

                item = {
                    "from": frm,
                    "expembe_to": expembe_to,
                    "expemba_to": expemba_to,
                    "expembe_dist": expembe_dist,
                    "expemba_dist": expemba_dist,
                }
                all_results.append(item)

                if expembe_dist < expemba_dist:
                    idx2stat[idx]["expembe"] += 1
                elif expembe_dist > expemba_dist:
                    idx2stat[idx]["expemba"] += 1
                else:
                    idx2stat[idx]["tie"] += 1
            except Exception as e:
                print(f"Error in frm: {frm}\nexpembe_to: {expembe_to}\nexpemba_to: {expemba_to}. Error: {e}")
                raise e

    print(f"Distance stats: {sorted(idx2stat.items(), key = lambda x : x[0])}")

    with open(args.result_file, "w") as f:
        result = {
            "args": vars(args),
            "n_keys": len(common_frms),
            "stats": idx2stat,
            "all_results": all_results,
        }
        json.dump(result, f, indent = 4)
