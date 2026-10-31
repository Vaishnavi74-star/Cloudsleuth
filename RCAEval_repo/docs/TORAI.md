# TORAI: Multi-Source Root Cause Analysis for Blind Spots in Microservice Service Call Graph

TORAI is a multi-source RCA method that combines metrics, logs, and traces to identify root cause services. It uses per-modality anomaly scoring, Gaussian Mixture Model clustering, and RCD-based within-cluster refinement.

**Table of Contents**

  * [Installation](#installation)
  * [Basic Usage](#basic-usage)
  * [Reproducibility](#reproducibility)
    + [Online Boutique](#online-boutique)
    + [Sock Shop](#sock-shop)
    + [Train Ticket](#train-ticket)
  * [Data Format](#data-format)

## Dataset

The TORAI datasets are available on Figshare: [https://doi.org/10.6084/m9.figshare.31925976](https://doi.org/10.6084/m9.figshare.31925976)

Download and extract the data:

```bash
wget -O torai-data.zip "https://ndownloader.figshare.com/files/63402423"
unzip torai-data.zip -d data/
```

## Installation

TORAI uses Python 3.8 and a patched version of `causal-learn` (for RCD's localized PC algorithm).

```bash
# create and activate the virtual environment
python3.8 -m venv .venv-torai
source .venv-torai/bin/activate

# install dependencies
pip install --upgrade pip
pip install -r requirements_torai.lock

# install RCAEval
pip install -e .

# IMPORTANT: link the patched causal-learn files
bash script/link-torai.sh
```

## Basic Usage

```bash
source .venv-torai/bin/activate

python main.py --method torai --dataset torai-ob --length 10
```

## Reproducibility

The TORAI datasets (`torai-OB`, `torai-SS`, `torai-TT`) contain multi-modal telemetry data (metrics, logs, and optionally traces) for three microservice systems. Each dataset has 30 fault scenarios with 3 repetitions each (90 cases total).

### Online Boutique

```bash
python main.py --method torai --dataset torai-ob --length 10
```

<details>
<summary>Expected output</summary>

```
--- Evaluation results ---
Avg@5-CPU:   0.96
Avg@5-MEM:   0.93
Avg@5-DISK:  1.0
Avg@5-SOCKET: 0.93
Avg@5-DELAY: 0.8
Avg@5-LOSS:  0.84
```
</details>

### Sock Shop

```bash
python main.py --method torai --dataset torai-ss --length 10
```

### Train Ticket

```bash
python main.py --method torai --dataset torai-tt --length 10
```

## Data Format

TORAI expects a directory structure with multi-modal telemetry files per fault case:

```
data/torai-OB/
  {service}_{fault_type}/
    {run}/
      simple_metrics.csv    # System metrics (CPU, memory, disk, latency) at 1s sampling
      logts.csv             # Log event counts aggregated at 15s intervals
      tracets_err.csv       # Trace error counts at 15s intervals (OB and TT only)
      tracets_lat.csv       # Trace latency at 15s intervals (OB and TT only)
      inject_time.txt       # Unix timestamp of fault injection
```

