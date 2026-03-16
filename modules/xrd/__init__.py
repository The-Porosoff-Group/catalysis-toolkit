"""
modules/xrd/__init__.py
XRD module package init + data file parsers.
"""

import os, re
import numpy as np

MODULE_INFO = {
    'name':        'XRD',
    'description': 'Le Bail profile fitting and phase identification',
    'status':      'active',
    'icon':        '🔬',
}

COMMON_WAVELENGTHS = {
    'Cu Kα':  1.54056,
    'Cu Kα1': 1.54056,
    'Cu Kα2': 1.54439,
    'Mo Kα':  0.71073,
    'Mo Kα1': 0.70930,
    'Co Kα':  1.78897,
    'Cr Kα':  2.28970,
    'Fe Kα':  1.93604,
    'Ag Kα':  0.55941,
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA FILE PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_xrd_file(filepath):
    """
    Parse XRD data file. Supports:
      .dat  — PowderGraph / Bruker format (2theta d intx sigx counts)
      .xy   — two-column (2theta intensity)
      .xye  — three-column (2theta intensity error)
      .csv  — comma-separated, auto-detected columns
      .txt  — whitespace-separated, auto-detected
    Returns dict:
      tt, intensity, sigma, metadata
    """
    ext = os.path.splitext(filepath)[1].lower()

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()

    lines = raw.splitlines()

    # Detect PowderGraph .dat format
    if '[PowderGraph' in raw or (len(lines) > 1 and '2thetadeg' in lines[1].lower()):
        return _parse_powdergraph(lines)

    # Try generic whitespace/CSV
    return _parse_generic(lines, ext)


def _parse_powdergraph(lines):
    """Parse the PowderGraph .dat format used by your instrument."""
    metadata = {}

    # Header
    if '[PowderGraph' in lines[0]:
        metadata['format'] = 'PowderGraph'
    col_line = lines[1] if len(lines) > 1 else ''
    cols = col_line.lower().split()
    # cols: 2thetadeg d-value intx sigx count

    tt_list, int_list, sig_list = [], [], []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            tt  = float(parts[0])
            d   = float(parts[1])
            ix  = float(parts[2])
            sx  = float(parts[3])
            ct  = float(parts[4])
            if ct > 0 and tt > 0:
                tt_list.append(tt)
                int_list.append(ix)
                sig_list.append(sx if sx > 0 else math.sqrt(max(ix, 1)))
        except ValueError:
            continue

    return {
        'tt':        np.array(tt_list),
        'intensity': np.array(int_list),
        'sigma':     np.array(sig_list),
        'metadata':  metadata,
    }


def _parse_generic(lines, ext):
    """Parse generic two- or three-column XRD data."""
    import math
    tt_list, int_list, sig_list = [], [], []
    sep = ',' if ext == '.csv' else None

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('!'):
            continue
        # Skip header lines
        try:
            float(line.split(sep)[0])
        except (ValueError, IndexError):
            continue

        parts = line.split(sep)
        try:
            tt  = float(parts[0])
            ix  = float(parts[1])
            sx  = float(parts[2]) if len(parts) > 2 else math.sqrt(max(ix, 1))
            if tt > 0 and ix >= 0:
                tt_list.append(tt)
                int_list.append(ix)
                sig_list.append(sx)
        except (ValueError, IndexError):
            continue

    return {
        'tt':        np.array(tt_list),
        'intensity': np.array(int_list),
        'sigma':     np.array(sig_list),
        'metadata':  {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT (called by app.py)
# ─────────────────────────────────────────────────────────────────────────────

def run(filepath, output_dir, metadata, params):
    """
    Called by app.py after user has confirmed phases and settings.

    params dict keys:
      phases       : list of phase dicts (from COD + user confirmation)
      wavelength   : float (Å)
      tt_min       : float
      tt_max       : float
      n_bg_coeffs  : int (default 6)
    """
    import os
    from .lebail  import run_lebail
    from .xrd_plots import make_xrd_plot

    os.makedirs(output_dir, exist_ok=True)

    # Parse data
    data = parse_xrd_file(filepath)
    tt        = data['tt']
    intensity = data['intensity']
    sigma     = data['sigma']

    wavelength  = params.get('wavelength', 1.54056)
    tt_min      = params.get('tt_min', float(tt.min()))
    tt_max      = params.get('tt_max', float(tt.max()))
    n_bg        = params.get('n_bg_coeffs', 6)
    phases      = params['phases']

    # Run refinement
    result = run_lebail(
        tt, intensity, sigma, phases, wavelength,
        tt_min=tt_min, tt_max=tt_max,
        n_bg_coeffs=n_bg, max_outer=15,
    )

    # Plot
    plot_path = os.path.join(output_dir, 'xrd_refinement.png')
    make_xrd_plot(result, {
        'sample_id':       metadata.get('sample_id', 'Sample'),
        'wavelength_label': params.get('wavelength_label', f"λ={wavelength:.5f} Å"),
    }, plot_path)

    # Summary CSV
    import pandas as pd
    rows = []
    for ph in result['phase_results']:
        row = {'sample': metadata.get('sample_id', ''), **ph}
        row.update(result['statistics'])
        row['zero_shift'] = result['zero_shift']
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary_path = os.path.join(output_dir, 'xrd_summary.csv')
    summary.to_csv(summary_path, index=False)

    return {
        'plot_path':    plot_path,
        'summary_path': summary_path,
        'statistics':   result['statistics'],
        'phase_results': result['phase_results'],
        'zero_shift':   result['zero_shift'],
        'result':       result,
    }
