#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Generating FX regime CSV exports for EURUSD, AUDUSD, USDJPY..."

python - <<'PY'
import importlib.util
import os
import sys
from pathlib import Path

os.environ["FIRSTRATEDATA_ROOT"] = "/home/matrillo/apps/jupyter-notebooks/histdata/firstratedata"

repo = Path('/home/matrillo/apps/regime-classification')
sys.path.insert(0, str(repo))

module_path = repo / 'utils' / 'fx_regime_dataset_export.py'
spec = importlib.util.spec_from_file_location('fx_export', str(module_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

export_dir = repo / 'exports' / 'forex'
symbols = ['EURUSD', 'AUDUSD', 'USDJPY']

for symbol in symbols:
    out_path = mod.build_regime_dataset_for_symbol(symbol, str(export_dir))
    print(f'wrote: {out_path}')
PY

echo
echo "Done. Expected files:"
echo "  /home/matrillo/apps/regime-classification/exports/forex/EURUSD_regime_ohlcv.csv"
echo "  /home/matrillo/apps/regime-classification/exports/forex/AUDUSD_regime_ohlcv.csv"
echo "  /home/matrillo/apps/regime-classification/exports/forex/USDJPY_regime_ohlcv.csv"
