"""
modules/xrd/__init__.py
XRD module — entry point, file parsers, run() called by app.py.
"""

import os, re, math
import numpy as np

MODULE_INFO = {
    'name':        'XRD',
    'description': 'Le Bail profile fitting with COD phase identification',
    'status':      'active',
    'icon':        '🔬',
}

COMMON_WAVELENGTHS = {
    'Cu Kα  (1.54056 Å)':  1.54056,
    'Cu Kα1 (1.54056 Å)':  1.54056,
    'Cu Kα2 (1.54439 Å)':  1.54439,
    'Mo Kα  (0.71073 Å)':  0.71073,
    'Mo Kα1 (0.70930 Å)':  0.70930,
    'Co Kα  (1.78897 Å)':  1.78897,
    'Cr Kα  (2.28970 Å)':  2.28970,
    'Fe Kα  (1.93604 Å)':  1.93604,
    'Ag Kα  (0.55941 Å)':  0.55941,
}


# ─────────────────────────────────────────────────────────────────────────────
# FILE PARSERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_xrd_file(filepath):
    """
    Parse XRD data file. Supports:
      .dat  — PowderGraph / Bruker (2theta d intx sigx counts)
      .xy   — two-column
      .xye  — three-column with errors
      .csv  — comma-separated
      .txt  — whitespace-separated
    Returns dict: tt, intensity, sigma, metadata
    """
    ext = os.path.splitext(filepath)[1].lower()
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()
    lines = raw.splitlines()

    if '[PowderGraph' in raw or (len(lines) > 1 and '2thetadeg' in lines[1].lower()):
        return _parse_powdergraph(lines)
    return _parse_generic(lines, ext)


def _parse_powdergraph(lines):
    tt_list, int_list, sig_list = [], [], []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 5: continue
        try:
            tt  = float(parts[0])
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
        'metadata':  {'format': 'PowderGraph'},
    }


def _parse_generic(lines, ext):
    tt_list, int_list, sig_list = [], [], []
    sep = ',' if ext == '.csv' else None
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('!'): continue
        try:
            float(line.split(sep)[0])
        except (ValueError, IndexError):
            continue
        parts = line.split(sep)
        try:
            tt = float(parts[0]); ix = float(parts[1])
            sx = float(parts[2]) if len(parts) > 2 else math.sqrt(max(ix, 1))
            if tt > 0 and ix >= 0:
                tt_list.append(tt); int_list.append(ix); sig_list.append(sx)
        except (ValueError, IndexError):
            continue
    return {
        'tt':        np.array(tt_list),
        'intensity': np.array(int_list),
        'sigma':     np.array(sig_list),
        'metadata':  {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# PHASE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_phases(phases, fetch_missing=True):
    """
    Validate and fill missing cell parameters.
    If a phase has cif_text, parse it to get the full structure.
    If cell params are missing and cod_id is set, fetch CIF from COD.
    """
    from .cod_api import fetch_cif
    from .crystallography import parse_cif

    validated = []
    for ph in phases:
        ph = dict(ph)

        # If CIF text is provided, parse it for authoritative cell params
        cif_text = ph.get('cif_text', '')
        if cif_text:
            try:
                parsed = parse_cif(cif_text)
                # Only fill fields that are missing or None
                for key in ['a','b','c','alpha','beta','gamma',
                            'spacegroup_number','system']:
                    if ph.get(key) is None and parsed.get(key) is not None:
                        ph[key] = parsed[key]
                if not ph.get('name'):
                    ph['name'] = parsed.get('formula', 'Phase')
            except Exception:
                pass

        # If still missing cell params, try fetching from COD
        a = ph.get('a')
        c = ph.get('c')
        if (a is None or c is None) and ph.get('cod_id') and str(ph.get('cod_id')) != 'manual':
            try:
                full = fetch_cif(str(ph['cod_id']))
                for key in ['a','b','c','alpha','beta','gamma',
                            'spacegroup_number','system','formula']:
                    if ph.get(key) is None and full.get(key) is not None:
                        ph[key] = full[key]
                # Also keep the CIF text for pymatgen
                if not ph.get('cif_text') and full.get('cif_text'):
                    ph['cif_text'] = full['cif_text']
            except Exception:
                pass

        # Safe float helper
        def _f(val, default):
            try: return float(val) if val is not None else default
            except (TypeError, ValueError): return default

        ph['a']     = _f(ph.get('a'),     4.0)
        ph['b']     = _f(ph.get('b'),     ph['a'])
        ph['c']     = _f(ph.get('c'),     ph['a'])
        ph['alpha'] = _f(ph.get('alpha'), 90.0)
        ph['beta']  = _f(ph.get('beta'),  90.0)
        ph['gamma'] = _f(ph.get('gamma'), 90.0)

        sg = ph.get('spacegroup_number')
        ph['spacegroup_number'] = int(sg) if sg is not None else 1
        ph['system'] = ph.get('system') or 'triclinic'
        ph['name']   = ph.get('name') or ph.get('formula') or 'Phase'

        validated.append(ph)
    return validated


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run(filepath, output_dir, metadata, params):
    """
    Called by app.py. Runs the full Le Bail pipeline.

    params keys:
      phases           list of phase dicts
      wavelength       float (Å)
      tt_min / tt_max  float
      n_bg_coeffs      int (default 6)
      wavelength_label str
    """
    from .lebail    import run_lebail
    from .xrd_plots import make_xrd_plot

    os.makedirs(output_dir, exist_ok=True)

    # Validate phases
    phases = validate_phases(params.get('phases', []))
    if not phases:
        raise ValueError('No valid phases provided for refinement.')

    # Parse data
    data      = parse_xrd_file(filepath)
    tt        = data['tt']
    intensity = data['intensity']
    sigma     = data['sigma']

    wavelength = params.get('wavelength', 1.54056)
    tt_min     = params.get('tt_min', float(tt.min()))
    tt_max     = params.get('tt_max', float(tt.max()))
    n_bg       = params.get('n_bg_coeffs', 6)
    max_outer  = params.get('max_outer', 10)

    # Run refinement
    result = run_lebail(
        tt, intensity, sigma, phases, wavelength,
        tt_min=tt_min, tt_max=tt_max,
        n_bg_coeffs=n_bg, max_outer=max_outer,
    )

    # Plot
    plot_path = os.path.join(output_dir, 'xrd_refinement.png')
    make_xrd_plot(result, {
        'sample_id':       metadata.get('sample_id', 'Sample'),
        'wavelength_label': params.get('wavelength_label',
                                        f"λ={wavelength:.5f} Å"),
    }, plot_path)

    # Summary CSV
    import pandas as pd
    rows = []
    for ph in result['phase_results']:
        row = {'sample': metadata.get('sample_id', ''), **ph}
        row.update(result['statistics'])
        row['zero_shift']    = result['zero_shift']
        row['pymatgen_used'] = result.get('pymatgen_used', False)
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
        'pymatgen_used': result.get('pymatgen_used', False),
        'result':       result,
    }
