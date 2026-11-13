#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Live LTR universe: TPIV v1.01 (AUDUSD/NZDUSD) + MHAL v1.02
# (NZDUSD/EURCAD/EURCHF). The original EURUSD/USDJPY exports are retained.
SYMBOLS="${REGIME_EXPORT_SYMBOLS:-EURUSD AUDUSD USDJPY NZDUSD EURCAD EURCHF}"

echo "Generating FX regime CSV exports for: ${SYMBOLS}"

python - <<'PY'
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "FIRSTRATEDATA_ROOT", "/home/matrillo/apps/jupyter-notebooks/histdata/firstratedata"
)

repo = Path('/home/matrillo/apps/regime-classification')
sys.path.insert(0, str(repo))

module_path = repo / 'utils' / 'fx_regime_dataset_export.py'
spec = importlib.util.spec_from_file_location('fx_export', str(module_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

export_dir = repo / 'exports' / 'forex'
symbols = os.environ.get(
    "REGIME_EXPORT_SYMBOLS", "EURUSD AUDUSD USDJPY NZDUSD EURCAD EURCHF"
).split()

for symbol in symbols:
    out_path = mod.build_regime_dataset_for_symbol(symbol, str(export_dir))
    print(f'wrote: {out_path}')
PY

echo
echo "Done. Expected files:"
for s in ${SYMBOLS}; do
  echo "  ${SCRIPT_DIR}/exports/forex/${s}_regime_ohlcv.csv"
done
