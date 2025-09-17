# EPF Cross Border Markets

This library provides tools for electricity price forecasting in coupled European spot markets, with a particular focus on leveraging differences in gate closure times (GCTs) between bidding zones.
The motivation stems from the increasing integration of European electricity markets, where prices across zones show growing convergence, yet still differ in operational aspects such as GCTs.
The framework implemented here allows forecasts for bidding zones (BDZs) BE and SE3 to be improved using earlier-published prices from Austria (AT), Germany–Luxembourg (DE-LU), and Switzerland (CH).

This repository builds upon the [EPF Toolbox](https://github.com/jeslago/epftoolbox), originally developed by Jesus Lago and collaborators, which provides a benchmark framework for **day-ahead electricity price forecasting (EPF)**.  
The original toolbox is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.  
This work includes **modifications and extensions** for research and operational forecasting workflows.

## Additions and Modifications

- **New entrypoint script: `run.py`**
  - Automatically detects Python interpreter (`$PYTHON` env var, `.venv`, or system default).
  - Allows running forecasting scripts as standalone or modules.
  - Provides CLI arguments for:
    - Selection of input features,
    - Forecast frequency type,
    - Calibration window size,
    - Market selection,
    - Timeout control.

- **Flexible Input Feature Selection**
  - Supports defining feature sets dynamically via dictionaries.
  - Enables easy experimentation with different predictors (load, wind, solar, etc.).

- **Requirements Management**
  - Added `requirements.txt` for reproducibility.
    
- **Portability**
  - Works on **Windows**, **Linux**, and **MacOS**.
  - Virtual environment detection (`.venv/bin/python` or `.venv/Scripts/python.exe`).

---

## Getting Started

#### 1. Clone the repository

git clone https://github.com/margaridamascarenhas/EPF_Cross-border-Markets.git
cd EPF_Cross-border-Markets

#### 2. (Optional) Set up a virtual environment

<pre>python -m venv .venv source
.venv/bin/activate # Linux/MacOS 
.venv\Scripts\activate # Windows </pre>

#### 3. Install dependencies

<pre>pip install -r requirements.txt</pre>

#### 4. Run the entry script

<pre>python run.py</pre>


## License

This repository is licensed under the terms of the
 - GNU Affero General Public License v3.0 (AGPL-3.0).

Full text available in the LICENSE file or at https://www.gnu.org/licenses/agpl-3.0.txt.


## Acknowledgments

 - Original Toolbox: [jeslago/epftoolbox](https://github.com/jeslago/epftoolbox)
 - Modifications: Margarida Mascarenhas (2025)
