
# Author: Maria Margarida Mascarenhas, 2025
# Licensed under the GNU Affero General Public License v3 (AGPL-3.0).

import subprocess
import os, sys, json
from pathlib import Path

# Root of the repo: the directory containing this file
REPO_ROOT = Path(__file__).resolve().parent

def _venv_python() -> str | None:
    """
    Return the path to the Python interpreter inside ./.venv if it exists,
    preferring Windows layout first, then Unix-like layout. Otherwise None.
    """
    win = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    nix = REPO_ROOT / ".venv" / "bin" / "python"
    if win.exists(): return str(win)
    if nix.exists(): return str(nix)
    return None

def _python_exe() -> str:
    """
    Decide which Python to use, in order of priority:
      1) $PYTHON env var (explicit override)
      2) The .venv interpreter if present
      3) The current interpreter (sys.executable)
    """
    return os.environ.get("PYTHON") or _venv_python() or sys.executable

def _as_module(script_path: str) -> str:
    """
    Convert a filesystem path to a Python module path so it can be run with -m.
    Example: './Forecast/DNN.py' -> 'Forecast.DNN'
    """
    p = Path(script_path).with_suffix('')
    # Join all path parts except '.' and '' with dots
    return ".".join(part for part in p.parts if part not in (".", ""))

def run_script(
    script_name,
    select_inputs,
    freq_type,
    calibration_window,
    market,
    timeout: int = 600
):
    """
    Run one training/forecast script with standardized CLI arguments.

    Args:
        script_name: Path to the script (e.g., './Forecast/DNN.py').
        select_inputs: List of feature names; passed as JSON via --select-inputs.
        freq_type: Recalibration frequency (e.g., 'daily', 'weekly', 'monthly', 'once').
        calibration_window: String/number window length (e.g., '56', '112', '365', '730').
        market: Bidding zone label ('BE' or 'SE3').
        use_module: If True and script_name endswith .py, run via `python -m <module>`.
        timeout: Seconds before the subprocess is aborted.
    """
    py = _python_exe()
    print(f"→ Using Python: {py}")  # log which interpreter will run

    # Serialize the feature list as JSON so the child process can parse it robustly
    select_inputs_arg = json.dumps(select_inputs)

    module = _as_module(script_name)
    cmd = [
        py, "-m", module,
        "--select-inputs", select_inputs_arg,
        "--rf", freq_type,
        "--cw", str(calibration_window),
        "--market", market,
    ]
    # Inherit the current environment; PYTHONPATH usually not needed when using -m from REPO_ROOT
    env = dict(os.environ)

    try:
        # Run with captured stdout/stderr, fail on non-zero exit codes, and set working dir to REPO_ROOT
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(REPO_ROOT),
            env=env,
            timeout=timeout
        )
        print(
            f"Ran {script_name} | inputs={select_inputs} | rf={freq_type} | "
            f"cw={calibration_window} | market={market}"
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            # Note: stderr may contain warnings; printing for visibility
            print(result.stderr)

    except subprocess.TimeoutExpired:
        # Timed out before completion
        print(f"Timeout running {script_name}: {' '.join(cmd)}")

    except subprocess.CalledProcessError as e:
        # Child process exited with non-zero status; show captured output to aid debugging
        print(f"Error running {script_name} (exit {e.returncode})")
        if e.stdout:
            print("--- stdout ---\n" + e.stdout)
        if e.stderr:
            print("--- stderr ---\n" + e.stderr)


# Scripts to execute (must accept the CLI flags used above)
scripts = [
    "./Forecast/LEAR.py",
    "./Forecast/Opt_Hyper.py",
    "./Forecast/DNN.py",
]

# Target market (controls feature sets below)
market = "BE" #Options: "BE" or "SE3" 

# Feature selections per market (comment/uncomment sets as needed)
if market == 'BE':
    select_inputs_list = [
        ['Solar', 'Wind', 'Load', 'Temp', 'Hum'],
        ['Solar','Wind','Load','Temp','Hum', 'Price_AT_15min'],
        ['Solar','Wind','Load','Temp','Hum', 'Price_AT_15min', 'Price_DE_LU_15min'],
        ['Solar','Wind','Load','Temp','Hum', 'Price_AT_15min', 'Price_DE_LU_15min', 'Price_CH'],
    ]
elif market == 'SE3':
    select_inputs_list = [
        ['Wind_onshore', 'Load', 'Temp', 'Hum'],
        ['Wind_onshore', 'Load', 'Temp', 'Hum', 'Price_AT_15min'],
        ['Wind_onshore', 'Load', 'Temp', 'Hum', 'Price_AT_15min', 'Price_DE_LU_15min'],
        ['Wind_onshore', 'Load', 'Temp', 'Hum', 'Price_AT_15min', 'Price_DE_LU_15min', 'Price_CH'],
    ]
else:
    raise ValueError(f"Market '{market}' not recognized. Please use 'BE' or 'SE3'.")

frequency_list = ["daily", 'weekly', 'monthly', 'once']#  Options: once, monthly, weekly or daily (pick one or a combination)

calibration_windows = [56, 112, 365, 730] # Options: 56, 112, 365, 730 (pick one or a combination)

# Forecasts (scripts × frequencies × feature sets × CWs) number of models
for script in scripts:
    for freq in frequency_list:
        for select_inputs in select_inputs_list:
            for cw in calibration_windows:
                run_script(script, select_inputs, freq, cw, market)
