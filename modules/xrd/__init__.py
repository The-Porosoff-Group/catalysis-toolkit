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
      .dat  — PowderGraph 5-col, OR Rigaku step-scan (header + single intensity per line)
      .xy   — two-column (2theta intensity)
      .xye  — three-column (2theta intensity error)
      .csv  — comma-separated
      .txt  — whitespace-separated
    Returns dict: tt, intensity, sigma, metadata
    """
    ext = os.path.splitext(filepath)[1].lower()
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()
    # Normalise line endings
    raw   = raw.replace('\r\n', '\n').replace('\r', '\n')
    lines = [l for l in raw.splitlines() if l.strip()]

    # PowderGraph format
    if '[PowderGraph' in raw or (len(lines) > 1 and '2thetadeg' in lines[1].lower()):
        return _parse_powdergraph(lines)

    # Rigaku/Bruker step-scan: first line = "start step end", rest = single values
    if _is_step_scan(lines):
        return _parse_step_scan(lines)

    return _parse_generic(lines, ext)


def _is_step_scan(lines):
    """
    Detect Rigaku/Bruker step-scan format:
    Line 0: exactly 3 numbers (start, step, end)
    Lines 1+: exactly 1 number each (intensity)
    """
    if len(lines) < 10:
        return False
    try:
        parts = lines[0].split()
        if len(parts) != 3:
            return False
        start, step, end = float(parts[0]), float(parts[1]), float(parts[2])
        if step <= 0 or start >= end:
            return False
        # Check that at least 80% of remaining lines are single numbers
        sample = lines[1:min(50, len(lines))]
        single = sum(1 for l in sample if len(l.split()) == 1
                     and _safe_float(l) is not None)
        return single / len(sample) > 0.8
    except Exception:
        return False


def _safe_float(s):
    try: return float(s.strip())
    except: return None


def _parse_step_scan(lines):
    """Parse Rigaku/Bruker step-scan: header line + one intensity per line."""
    parts = lines[0].split()
    start = float(parts[0])
    step  = float(parts[1])

    intensities = []
    for line in lines[1:]:
        v = _safe_float(line)
        if v is not None:
            intensities.append(v)

    n  = len(intensities)
    tt = np.array([start + i * step for i in range(n)])
    iy = np.array(intensities)
    sg = np.sqrt(np.maximum(iy, 1.0))
    return {
        'tt':        tt,
        'intensity': iy,
        'sigma':     sg,
        'metadata':  {'format': 'StepScan',
                      'start': start, 'step': step, 'n_points': n},
    }


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

def _sg_symbol_to_number(symbol):
    """
    Map common space group symbol strings to International Tables numbers.
    Handles common variations in formatting.
    """
    # Normalise: remove spaces, underscores around subscripts
    s = str(symbol).strip().replace(' ', '').replace('_', '')
    # Direct lookup table for phases common in catalysis research
    _MAP = {
        'Im-3m': 229, 'Im3m': 229, 'Im-3m': 229,
        'Fm-3m': 225, 'Fm3m': 225,
        'Pm-3n': 223, 'Pm3n': 223,
        'Pm-3m': 221, 'Pm3m': 221,
        'Pa-3':  205, 'Pa3':  205,
        'Ia-3':  206, 'Ia3':  206,
        'Ia-3d': 230, 'Ia3d': 230,
        'Fd-3m': 227, 'Fd3m': 227,
        'P63/mmc': 194, 'P6_3/mmc': 194, 'P63mmc': 194,
        'P6/mmm': 191, 'P6mmm': 191,
        'P63/m':  176, 'P6_3/m': 176,
        'P-6m2':  187, 'P6m2':  187,
        'P-6':    174,
        'P63mc':  186,
        'P6322':  182,
        'P63':    173,
        'R-3m':   166, 'R3m': 166,
        'R-3':    148, 'R3':  148,
        'Pnma':    62, 'Pbnm': 62,
        'Cmcm':    63, 'Cmce': 64,
        'P21/c':   14, 'P2_1/c': 14,
        'C2/m':    12,
        'P-1':      2, 'P1':  1,
    }
    # Try as-is first, then stripped
    return _MAP.get(symbol.strip()) or _MAP.get(s)


def _to_conventional(ph):
    """
    Convert primitive cell settings to conventional.
    pymatgen often returns primitive cells which our reflection generator
    can't handle — we need conventional cells with standard angles.

    BCC primitive:  a_prim, alpha=beta=gamma=109.471° → a_conv = a_prim * 2/√3
    FCC primitive:  a_prim, alpha=beta=gamma=60°      → a_conv = a_prim * √2
    Rhombohedral:   treated as hexagonal if SG 146-167
    """
    import math
    a   = ph['a']
    al  = ph.get('alpha', 90.0)
    be  = ph.get('beta',  90.0)
    ga  = ph.get('gamma', 90.0)
    sys_ = (ph.get('system') or 'triclinic').lower()
    sg   = ph.get('spacegroup_number', 1)

    # BCC primitive: a=b=c, alpha=beta=gamma ≈ 109.47°
    if (sys_ == 'cubic' and
            abs(al - 109.471) < 0.6 and
            abs(be - 109.471) < 0.6 and
            abs(ga - 109.471) < 0.6):
        a_conv = a * 2.0 / math.sqrt(3.0)
        return {**ph, 'a': round(a_conv, 5), 'b': round(a_conv, 5),
                'c': round(a_conv, 5), 'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0}

    # FCC primitive: a=b=c, alpha=beta=gamma ≈ 60°
    if (sys_ == 'cubic' and
            abs(al - 60.0) < 0.6 and
            abs(be - 60.0) < 0.6 and
            abs(ga - 60.0) < 0.6):
        a_conv = a * math.sqrt(2.0)
        return {**ph, 'a': round(a_conv, 5), 'b': round(a_conv, 5),
                'c': round(a_conv, 5), 'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0}

    # Rhombohedral primitive stored as trigonal: force correct angles
    if sys_ in ('trigonal', 'hexagonal') and 143 <= sg <= 167:
        return {**ph, 'alpha': 90.0, 'beta': 90.0, 'gamma': 120.0}

    return ph


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
                            'spacegroup_number','system','Z']:
                    if ph.get(key) is None and parsed.get(key) is not None:
                        ph[key] = parsed[key]
                if not ph.get('name'):
                    ph['name'] = parsed.get('formula', 'Phase')
                if not ph.get('formula') and parsed.get('formula'):
                    ph['formula'] = parsed['formula']
            except Exception:
                pass

        # If still missing cell params, try fetching from COD
        a = ph.get('a')
        c = ph.get('c')
        if (a is None or c is None) and ph.get('cod_id') and str(ph.get('cod_id')) != 'manual':
            try:
                full = fetch_cif(str(ph['cod_id']))
                for key in ['a','b','c','alpha','beta','gamma',
                            'spacegroup_number','system','formula','Z']:
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

        # If spacegroup_number is 1 (default/unknown), try to infer from symbol
        if ph['spacegroup_number'] == 1 and ph.get('spacegroup'):
            inferred = _sg_symbol_to_number(ph['spacegroup'])
            if inferred:
                ph['spacegroup_number'] = inferred

        # If system is still triclinic but we have a real spacegroup number, fix it
        if ph['system'] == 'triclinic' and ph['spacegroup_number'] > 2:
            from .cod_api import infer_system
            ph['system'] = infer_system(ph['spacegroup_number'])

        # Convert primitive cells to conventional based on space group
        # e.g. pymatgen returns primitive BCC (a=2.746, alpha=109.47) for mp-91
        ph = _to_conventional(ph)

        validated.append(ph)
    return validated


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def _write_summary_xlsx(result, metadata, method_label, output_dir):
    """Write an Excel workbook with two sheets:
      Sheet 1 ('Summary')   – transposed: parameters as rows, phases as columns
      Sheet 2 ('Plot Data') – X/Y arrays + per-phase peak positions with [hkl]
    """
    import pandas as pd

    phases = result['phase_results']
    stats  = result['statistics']

    # ── Sheet 1: Transposed summary ──────────────────────────────────────
    # Define the parameters we want to show (order matters)
    param_keys = [
        ('sample',             'Sample'),
        ('method',             'Method'),
        ('name',               'Phase name'),
        ('cod_id',             'COD / MP ID'),
        ('formula',            'Formula'),
        ('spacegroup',         'Space group'),
        ('spacegroup_number',  'Space group #'),
        ('system',             'Crystal system'),
        ('a',                  'a (Å)'),
        ('b',                  'b (Å)'),
        ('c',                  'c (Å)'),
        ('alpha',              'α (°)'),
        ('beta',               'β (°)'),
        ('gamma',              'γ (°)'),
        ('weight_fraction_%',  'Weight fraction (%)'),
        ('crystallite_size_nm','Crystallite size (nm)'),
        ('scale',              'Scale factor'),
        ('U',                  'U (Caglioti)'),
        ('V',                  'V (Caglioti)'),
        ('W',                  'W (Caglioti)'),
        ('X',                  'X (Lorentzian)'),
        ('Y',                  'Y (Lorentzian)'),
        ('eta_at_strongest',   'η at strongest'),
        ('fwhm_deg',           'FWHM (°)'),
        ('n_reflections',      'N reflections'),
        ('seeded_by',          'Seeded by'),
    ]
    # Add statistics
    stat_keys = [
        ('Rwp',  'Rwp (%)'),
        ('Rp',   'Rp (%)'),
        ('chi2', 'χ²'),
        ('GoF',  'GoF'),
    ]

    rows_data = []
    for key, label in param_keys:
        row = {'Parameter': label}
        for i, ph in enumerate(phases):
            col_name = ph.get('name', f'Phase {i+1}')
            if key == 'sample':
                row[col_name] = metadata.get('sample_id', '')
            elif key == 'method':
                row[col_name] = method_label
            else:
                row[col_name] = ph.get(key, '')
        rows_data.append(row)

    # Add zero shift (shared)
    zs_row = {'Parameter': 'Zero shift (°)'}
    for i, ph in enumerate(phases):
        zs_row[ph.get('name', f'Phase {i+1}')] = result['zero_shift']
    rows_data.append(zs_row)

    for key, label in stat_keys:
        row = {'Parameter': label}
        for i, ph in enumerate(phases):
            row[ph.get('name', f'Phase {i+1}')] = stats.get(key, '')
        rows_data.append(row)

    df_summary = pd.DataFrame(rows_data)

    # ── Sheet 2: Plot data + peak positions ──────────────────────────────
    tt = result.get('tt', [])
    plot_dict = {
        '2theta':     tt,
        'Y_obs':      result.get('y_obs', []),
        'Y_calc':     result.get('y_calc', []),
        'Background': result.get('y_background', []),
        'Residual':   result.get('residuals', []),
    }
    # Per-phase component curves
    phase_patterns = result.get('phase_patterns', [])
    for i, ph in enumerate(phases):
        col = ph.get('name', f'Phase {i+1}')
        if i < len(phase_patterns):
            plot_dict[col] = phase_patterns[i]

    df_plot = pd.DataFrame(plot_dict)

    # Per-phase reflection positions with Miller indices
    # Generate hkl data from cell parameters
    from .crystallography import generate_reflections, parse_cif

    wavelength = result.get('wavelength', 1.54056)
    tt_min = min(tt) if tt else 5.0
    tt_max = max(tt) if tt else 90.0

    # Find max rows needed for reflection columns
    phase_ref_data = []
    for ph in phases:
        refs = []
        try:
            sys_ = (ph.get('system') or 'triclinic').lower()
            sg   = ph.get('spacegroup_number', 1)
            sites = None
            cif_text = ph.get('cif_text', '')
            if cif_text:
                try:
                    parsed = parse_cif(cif_text)
                    sites = parsed.get('sites') or None
                except Exception:
                    pass
            ref_list = generate_reflections(
                ph.get('a', 1), ph.get('b', 1), ph.get('c', 1),
                ph.get('alpha', 90), ph.get('beta', 90), ph.get('gamma', 90),
                sys_, sg, wavelength, tt_min, tt_max, hkl_max=12,
                sites=sites)
            for r in ref_list:
                h, k, l = r[2]
                refs.append((round(r[0], 3), f'[{h}{k}{l}]'))
        except Exception:
            pass
        phase_ref_data.append(refs)

    # Append reflection columns to plot dataframe (separate column block per phase)
    max_refs = max((len(r) for r in phase_ref_data), default=0)
    for i, ph in enumerate(phases):
        pname = ph.get('name', f'Phase {i+1}')
        tt_col = f'{pname} peak 2θ'
        hkl_col = f'{pname} [hkl]'
        refs = phase_ref_data[i] if i < len(phase_ref_data) else []
        # Pad to dataframe length
        tt_vals  = [r[0] for r in refs] + [None] * (len(df_plot) - len(refs))
        hkl_vals = [r[1] for r in refs] + [None] * (len(df_plot) - len(refs))
        df_plot[tt_col]  = tt_vals[:len(df_plot)]
        df_plot[hkl_col] = hkl_vals[:len(df_plot)]

    # ── Write xlsx (fall back to CSV if openpyxl not installed) ─────────
    try:
        import openpyxl  # noqa: F401 – just check availability
        xlsx_path = os.path.join(output_dir, 'xrd_summary.xlsx')
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            df_plot.to_excel(writer, sheet_name='Plot Data', index=False)
        return xlsx_path
    except ImportError:
        # openpyxl not available — write two CSV files instead
        summary_csv = os.path.join(output_dir, 'xrd_summary.csv')
        df_summary.to_csv(summary_csv, index=False)
        plot_csv = os.path.join(output_dir, 'xrd_plot_data.csv')
        df_plot.to_csv(plot_csv, index=False)
        return summary_csv


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run(filepath, output_dir, metadata, params):
    """
    Called by app.py. Runs Le Bail or Rietveld pipeline.

    params keys:
      phases           list of phase dicts
      wavelength       float (Å)
      tt_min / tt_max  float
      n_bg_coeffs      int (default 6)
      wavelength_label str
      method           'lebail' (default) or 'rietveld'
    """
    from .lebail    import run_lebail, run_rietveld
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
    method     = params.get('method', 'lebail').lower()

    # Run refinement
    if method == 'gsas2':
        from .gsasii_backend import run_gsas2, is_available as gsas2_available
        if not gsas2_available():
            from .gsasii_backend import import_error
            raise RuntimeError(
                f"GSAS-II is not available: {import_error()}\n"
                f"Install with: conda install gsas2full -c briantoby")
        # GSAS-II needs CIF text — pull from cache if stripped
        from .cif_cache import get_cif
        for ph in phases:
            if not ph.get('cif_text'):
                cid = ph.get('cod_id') or ph.get('mp_id')
                if cid:
                    ph['cif_text'] = get_cif(cid)
        missing = [ph.get('name', '?') for ph in phases
                   if not ph.get('cif_text')]
        if missing:
            raise ValueError(
                f"GSAS-II requires CIF with atom coordinates for all phases. "
                f"Missing for: {', '.join(missing)}.")
        result = run_gsas2(
            tt, intensity, sigma, phases, wavelength,
            tt_min=tt_min, tt_max=tt_max,
            n_bg_coeffs=n_bg, max_cycles=max_outer * 3,
        )
    elif method == 'rietveld':
        # Check that all phases have atom sites (CIF text)
        missing = [ph.get('name','?') for ph in phases
                   if not ph.get('sites') and not ph.get('cif_text')]
        if missing:
            raise ValueError(
                f"Rietveld requires CIF with atom coordinates for all phases. "
                f"Missing for: {', '.join(missing)}. "
                f"Try fetching the CIF first, or use Le Bail instead.")
        result = run_rietveld(
            tt, intensity, sigma, phases, wavelength,
            tt_min=tt_min, tt_max=tt_max,
            n_bg_coeffs=n_bg, max_iter=max_outer * 5,
        )
    else:
        result = run_lebail(
            tt, intensity, sigma, phases, wavelength,
            tt_min=tt_min, tt_max=tt_max,
            n_bg_coeffs=n_bg, max_outer=max_outer,
        )

    # Plot
    method_label = {'rietveld': 'Rietveld', 'gsas2': 'GSAS-II',
                    }.get(method, 'Le Bail')
    plot_path = os.path.join(output_dir, 'xrd_refinement.png')
    make_xrd_plot(result, {
        'sample_id':       metadata.get('sample_id', 'Sample'),
        'wavelength_label': params.get('wavelength_label',
                                        f"λ={wavelength:.5f} Å"),
        'method':           method_label,
    }, plot_path)

    # Summary Excel (two sheets: transposed summary + plot data with peaks)
    import pandas as pd
    summary_path = _write_summary_xlsx(result, metadata, method_label, output_dir)

    return {
        'plot_path':    plot_path,
        'summary_path': summary_path,
        'statistics':   result['statistics'],
        'phase_results': result['phase_results'],
        'zero_shift':   result['zero_shift'],
        'pymatgen_used': result.get('pymatgen_used', False),
        'method':        method_label,
        'result':       result,
    }
