#!/bin/bash
# Launch the toolkit with a source-installed GSAS-II on PYTHONPATH.
# Activate your conda env first, e.g. `conda activate catalysis`.
# Override the location with: GSAS2_ROOT=/path/to/GSAS-II bash run_local.sh
GSAS2_ROOT="${GSAS2_ROOT:-$HOME/g2full/GSAS-II}"

if [ -d "$GSAS2_ROOT/GSASII" ]; then
    # GSAS-II locates its own GSASII-bin/ subdirectory from here.
    export PYTHONPATH="$GSAS2_ROOT/GSASII:$PYTHONPATH"
else
    echo "GSAS-II not found at $GSAS2_ROOT - starting without it." >&2
fi

python app.py
