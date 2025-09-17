# This file is part of epftoolbox (https://github.com/jeslago/epftoolbox)
# Modified by Maria Margarida Mascarenhas, 2025
# Licensed under the GNU Affero General Public License v3 (AGPL-3.0).

from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta
import time
import csv
import os

import numpy as np
import pandas as pd

from .models import LEAR
from .data import read_data


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
    p = argparse.ArgumentParser(description="Run LEAR day-ahead forecast with recalibration.")
    p.add_argument("--select-inputs", required=True, type=parse_select_inputs,
                   help='Feature list, e.g. \'["Solar","Wind"]\' or "Solar,Wind"')
    p.add_argument("--rf", "--recalibrate-frequency", dest="rf", required=True,
                   choices=["once", "weekly", "monthly", "daily"], help="Recalibration frequency.")
    p.add_argument("--cw", "--calibration-window", dest="cw", required=True,
                   type=int, choices=[56, 112, 365, 730], help="Calibration window (days).")
    p.add_argument("--market", required=True, choices=["BE", "SE3"], help="Market to run.")
    p.add_argument("--begin-test-date", default="2024-01-01", help="YYYY-MM-DD (inclusive).")
    p.add_argument("--end-test-date",   default="2024-31-12", help="YYYY-MM-DD (inclusive).")
    p.add_argument("--datasets-path", default="./datasets", help="Folder with input datasets.")
    p.add_argument("--outdir", default="Forecast/experimental_files", help="Where to write forecasts.")
    p.add_argument("--years-test", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    script_start_time = time.time()
    log_entries = []

    dataset = {"BE": "BE_Data_UTC", "SE3": "SE3_Data_UTC"}[args.market]

    datasets_path = Path(args.datasets_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Running Select_Inputs={args.select_inputs}, rf={args.rf}, cw={args.cw}, market={args.market}")

    # Read data
    df_train, df_test = read_data(
        dataset=dataset,
        years_test=args.years_test,
        path=str(datasets_path),
        begin_test_date=args.begin_test_date,
        end_test_date=args.end_test_date,
        Select_Inputs=args.select_inputs,
    )

    # Prepare forecast containers
    forecast = pd.DataFrame(index=df_test.index[::24], columns=[f"h{k}" for k in range(24)])
    real_values = df_test.loc[:, ["Price"]].values.reshape(-1, 24)  # kept for parity with original
    real_values = pd.DataFrame(real_values, index=forecast.index, columns=forecast.columns)
    forecast_dates = forecast.index

    # Output filename
    feat_slug = slugify_list(args.select_inputs)
    forecast_fname = f"LEAR_forecast_{dataset}_CW{args.cw}_RF_{args.rf}_{feat_slug}.csv"
    forecast_fpath = outdir / forecast_fname

    # Model
    model = LEAR(calibration_window=args.cw, recalibrate_frequency=args.rf)

    # Rolling recalibration + forecast
    for date in forecast_dates:
        print("Forecasting date:", date)
        data_available = pd.concat([df_train, df_test.loc[:date + pd.Timedelta(hours=23), :]], axis=0)
        data_available.loc[date:date + pd.Timedelta(hours=23), "Price"] = np.nan
        Yp = model.recalibrate_and_forecast_next_day(
            df=data_available,
            next_day_date=date,
            calibration_window=args.cw,
        )
        forecast.loc[date, :] = Yp

        # Write incrementally
        forecast.to_csv(forecast_fpath)

    # Logging runtime
    total_elapsed_time = time.time() - script_start_time
    experiment_label = f"LEAR_forecast_{dataset}_CW{args.cw}_RF_{args.rf}_{feat_slug}"
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
