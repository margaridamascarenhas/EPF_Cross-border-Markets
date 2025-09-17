# This file is part of epftoolbox (https://github.com/jeslago/epftoolbox)
# Modified by Maria Margarida Mascarenhas, 2025
# Licensed under the GNU Affero General Public License v3 (AGPL-3.0).

from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import csv
import os

from .models import hyperparameter_optimizer

def parse_select_inputs(val: str):
    """Accept JSON list (preferred) or comma-separated list of feature names."""
    try:
        parsed = json.loads(val)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    except json.JSONDecodeError:
        pass
    items = [x.strip() for x in val.split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("select_inputs must be a JSON list or comma-separated list of strings.")
    return items

def slugify_list(items: list[str]) -> str:
    return "-".join(x.replace("/", "-").replace(" ", "_") for x in items)

def parse_args():
    p = argparse.ArgumentParser(description="Run hyperparameter optimization for DNN models.")
    p.add_argument("--select-inputs", required=True, type=parse_select_inputs,
                   help='Feature list, e.g. \'["Solar","Wind"]\' or "Solar,Wind"')
    p.add_argument("--rf", "--recalibrate-frequency", dest="rf", required=True,
                   choices=["once", "weekly", "monthly", "daily"],
                   help="Recalibration frequency (parsed for consistency with other scripts; not used here).")
    p.add_argument("--cw", "--calibration-window", dest="cw", required=True,
                   type=int, choices=[56, 112, 365, 730], help="Calibration window (days).")
    p.add_argument("--market", required=True, choices=["BE", "SE3"], help="Market to run.")
    p.add_argument("--years-test", type=int, default=1)
    p.add_argument("--begin-test-date", default="2023-01-01", help="YYYY-MM-DD (inclusive).")
    p.add_argument("--end-test-date",   default="2023-12-31", help="YYYY-MM-DD (inclusive).")
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--shuffle-train", type=int, default=1)
    p.add_argument("--data-augmentation", type=int, default=0)
    p.add_argument("--new-hyperopt", type=int, default=1)
    p.add_argument("--max-evals", type=int, default=600)
    p.add_argument("--datasets-path", default="./datasets", help="Folder with input datasets.")
    p.add_argument("--outdir", default="Forecast/experimental_files", help="Where to write hyperopt artifacts.")
    p.add_argument("--experiment-id", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    script_start_time = time.time()
    log_entries = []

    dataset = {"BE": "BE_Data_UTC", "SE3": "SE3_Data_UTC"}[args.market]

    datasets_path = Path(args.datasets_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[Hyperopt] market={args.market} dataset={dataset} cw={args.cw} rf={args.rf} "
          f"nlayers={args.nlayers} years_test={args.years_test} "
          f"range={args.begin_test_date}..{args.end_test_date}")
    print(f"[Hyperopt] select_inputs={args.select_inputs}")

    # Run optimizer
    hyperparameter_optimizer(
        path_datasets_folder=str(datasets_path) + ("/" if not str(datasets_path).endswith("/") else ""),
        path_hyperparameters_folder=str(outdir) + ("/" if not str(outdir).endswith("/") else ""),
        new_hyperopt=args.new_hyperopt,
        max_evals=args.max_evals,
        nlayers=args.nlayers,
        dataset=dataset,
        years_test=args.years_test,
        calibration_window=args.cw,
        shuffle_train=args.shuffle_train,
        data_augmentation=args.data_augmentation,
        experiment_id=args.experiment_id,
        begin_test_date=args.begin_test_date,
        end_test_date=args.end_test_date,
        Select_Inputs=args.select_inputs,
    )

    # Logging runtime
    feat_slug = slugify_list(args.select_inputs)
    experiment_label = f"Opt_hyper_DNN_forecast_{dataset}_CW{args.cw}_Exp{args.experiment_id}_{feat_slug}"

    total_elapsed_time = time.time() - script_start_time
    log_entries.append({"Experiment": experiment_label, "runtime_seconds": round(total_elapsed_time, 2)})

    log_file_path = Path("./log_run_duration.csv")
    file_exists = log_file_path.exists()
    with log_file_path.open(mode="a", newline="") as log_file:
        fieldnames = ["Experiment", "runtime_seconds"]
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for entry in log_entries:
            writer.writerow(entry)

    print(f"Run details have been logged to {log_file_path}")


if __name__ == "__main__":
    main()
