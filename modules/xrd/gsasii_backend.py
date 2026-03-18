"""
modules/xrd/gsasii_backend.py
GSAS-II integration for Rietveld/Le Bail refinement via GSASIIscriptable.

Requires GSAS-II installed in the Python environment.
Install via:  git clone https://github.com/AdvancedPhotonSource/GSAS-II.git
              cd GSAS-II && pip install .
This module wraps GSASIIscriptable to provide a refinement backend compatible
with the toolkit's result format (same keys as run_lebail / run_rietveld).
"""

import math, os, sys, tempfile, warnings
import numpy as np

# ── GSAS-II availability check ──────────────────────────────────────────────

_GSASII_AVAILABLE = False
_GSASII_IMPORT_ERROR = None

def _add_gsas2pkg_paths():
    """Add gsas2pkg conda install paths to sys.path if not already present.
    gsas2pkg installs to {conda_prefix}/GSAS-II/ rather than site-packages."""
    prefix = sys.prefix  # e.g. C:\catalysis-toolkit\.conda_env
    candidates = [
        os.path.join(prefix, 'GSAS-II', 'GSASII'),   # for import GSASIIscriptable
        os.path.join(prefix, 'GSAS-II'),               # for import GSASII.GSASIIscriptable
        os.path.join(prefix, 'GSAS-II', 'backcompat'), # backcompat shim
    ]
    for p in candidates:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

_add_gsas2pkg_paths()

try:
    # New-style import (pip-installed from GitHub, or gsas2pkg via GSAS-II subpackage)
    import GSASII.GSASIIscriptable as G2sc
    _GSASII_AVAILABLE = True
except ImportError:
    try:
        # Direct import (gsas2pkg installs GSASII/ dir; backcompat or GSASII on path)
        import GSASIIscriptable as G2sc
        _GSASII_AVAILABLE = True
    except ImportError as e:
        _GSASII_IMPORT_ERROR = str(e)

try:
    from .crystallography import (
        compute_fit_statistics, cell_volume, molar_mass_from_formula,
        tch_fwhm_eta, size_from_Y, scherrer_size,
    )
except ImportError:
    from crystallography import (
        compute_fit_statistics, cell_volume, molar_mass_from_formula,
        tch_fwhm_eta, size_from_Y, scherrer_size,
    )


def is_available():
    """Return True if GSASIIscriptable is importable and functional."""
    if not _GSASII_AVAILABLE:
        return False
    # Quick functional check: verify G2Project is callable
    # (catches partial installs where Fortran extensions are missing)
    try:
        if not hasattr(G2sc, 'G2Project'):
            return False
    except Exception:
        return False
    return True


def import_error():
    """Return the import error message, or None if available."""
    return _GSASII_IMPORT_ERROR


# ─────────────────────────────────────────────────────────────────────────────
# SPACE GROUP TABLE  (number → H-M symbol for CIF generation)
# ─────────────────────────────────────────────────────────────────────────────

_SG_HM = {
    # Common catalyst phases — add entries as needed
    1: 'P 1', 2: 'P -1', 12: 'C 2/m', 14: 'P 21/c', 15: 'C 2/c',
    62: 'P n m a', 63: 'C m c m', 139: 'I 4/m m m', 141: 'I 41/a m d',
    148: 'R -3', 166: 'R -3 m', 167: 'R -3 c',
    173: 'P 63', 176: 'P 63/m', 186: 'P 63 m c', 187: 'P -6 m 2',
    191: 'P 6/m m m', 194: 'P 63/m m c',
    196: 'F 2 3', 202: 'F m -3', 216: 'F -4 3 m', 225: 'F m -3 m',
    227: 'F d -3 m', 229: 'I m -3 m', 223: 'P m -3 n', 221: 'P m -3 m',
    230: 'I a -3 d',
}


def _build_conventional_cif(ph):
    """
    Build a synthetic CIF string using the phase dict's (conventional) cell
    parameters and atom sites.

    This ensures GSAS-II always sees a CIF consistent with the conventional
    cell, even when the original CIF used a primitive setting (common with
    Materials Project data).  The space group is written explicitly so that
    GSAS-II applies the correct cell-parameter constraints.
    """
    try:
        from .crystallography import parse_cif
    except ImportError:
        from crystallography import parse_cif

    a   = ph.get('a', 4.0)
    b   = ph.get('b', a)
    c   = ph.get('c', a)
    al  = ph.get('alpha', 90.0)
    be  = ph.get('beta', 90.0)
    ga  = ph.get('gamma', 90.0)
    sg  = ph.get('spacegroup_number', 1)
    formula = ph.get('formula', '')
    Z   = ph.get('Z', '')

    # H-M symbol — try phase dict first, then lookup table
    hm = ph.get('spacegroup', '') or ph.get('spacegroup_name', '')
    if not hm:
        hm = _SG_HM.get(sg, 'P 1')

    # Get atom sites from the original CIF text
    sites = []
    cif_text = ph.get('cif_text', '')
    if cif_text:
        try:
            parsed = parse_cif(cif_text)
            sites = parsed.get('sites', [])
        except Exception:
            pass

    lines = [
        'data_phase',
        f"_chemical_formula_sum '{formula}'" if formula else '',
        f"_cell_formula_units_Z {Z}" if Z else '',
        f'_cell_length_a {a:.5f}',
        f'_cell_length_b {b:.5f}',
        f'_cell_length_c {c:.5f}',
        f'_cell_angle_alpha {al:.3f}',
        f'_cell_angle_beta  {be:.3f}',
        f'_cell_angle_gamma {ga:.3f}',
        f'_symmetry_Int_Tables_number {sg}',
        f"_symmetry_space_group_name_H-M '{hm}'",
    ]

    if sites:
        lines += [
            '',
            'loop_',
            '_atom_site_type_symbol',
            '_atom_site_fract_x',
            '_atom_site_fract_y',
            '_atom_site_fract_z',
            '_atom_site_occupancy',
        ]
        for el, x, y, z, occ in sites:
            lines.append(f'{el}  {x:.6f}  {y:.6f}  {z:.6f}  {occ:.4f}')

    return '\n'.join(ln for ln in lines if ln is not None) + '\n'


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _write_xye(path, tt, y_obs, sigma):
    """Write a .xye file (2theta  intensity  sigma) for GSAS-II import."""
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for i in range(len(tt)):
            f.write(f"{tt[i]:.6f}  {y_obs[i]:.4f}  {sigma[i]:.4f}\n")


def _write_instprm(work_dir, wavelength):
    """Write a minimal GSAS-II .instprm file. Returns path."""
    path = os.path.join(work_dir, 'instrument.instprm')
    lines = [
        '#GSAS-II instrument parameter file; do not add/delete items!',
        'Type:PXC',
        f'Lam:{wavelength:.6f}',
        'Zero:0.0',
        'Polariz.:0.99',
        'U:2.0',
        'V:-2.0',
        'W:5.0',
        'X:0.0',
        'Y:0.0',
        'Z:0.0',
        'SH/L:0.002',
        'Azimuth:0.0',
    ]
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def _write_temp_cif(cif_text, phase_name='phase', work_dir=None, index=0):
    """Write CIF text to a temporary file. Returns path.

    If *work_dir* is given the file is created there (avoids Windows
    short-path / permission issues with the system temp directory).
    The *index* parameter ensures unique filenames when multiple phases
    share the same name.
    """
    if work_dir is not None:
        # Sanitise phase_name for use as a filename component
        safe = "".join(c if c.isalnum() or c in ('_', '-') else '_'
                       for c in phase_name)
        path = os.path.join(work_dir, f'gsas_{index}_{safe}.cif')
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(cif_text)
        return path
    # Fallback: system temp
    fd, path = tempfile.mkstemp(suffix='.cif', prefix=f'gsas_{index}_{phase_name}_')
    with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as f:
        f.write(cif_text)
    return path


def _extract_profile_params(phase_obj):
    """
    Extract TCH profile parameters from a GSAS-II phase object.
    GSAS-II stores profile coefficients in the histogram data.
    Returns dict with U, V, W, X, Y and derived quantities.
    """
    try:
        # GSAS-II stores profile params per histogram in the phase
        hapData = list(phase_obj.data['Histograms'].values())[0]
        size_data = hapData.get('Size', [])
        strain_data = hapData.get('Mustrain', [])

        # GSAS-II uses isotropic size/strain by default
        # Size[1][0] = isotropic size in Å, Mustrain[1][0] = microstrain (%)
        cryst_size_A = None
        if size_data and len(size_data) > 1:
            cryst_size_A = float(size_data[1][0]) if size_data[1][0] > 0 else None

        microstrain = None
        if strain_data and len(strain_data) > 1:
            microstrain = float(strain_data[1][0])

        return {
            'crystallite_size_A': cryst_size_A,
            'microstrain_pct': microstrain,
        }
    except Exception:
        return {'crystallite_size_A': None, 'microstrain_pct': None}


def _extract_instrument_params(histogram):
    """Extract instrument/profile parameters from a GSAS-II histogram."""
    try:
        inst_params = histogram.data['Instrument Parameters'][0]
        U = float(inst_params.get('U', [0, 0])[1])
        V = float(inst_params.get('V', [0, 0])[1])
        W = float(inst_params.get('W', [0, 0])[1])
        X = float(inst_params.get('X', [0, 0])[1])
        Y = float(inst_params.get('Y', [0, 0])[1])
        return {'U': U, 'V': V, 'W': W, 'X': X, 'Y': Y}
    except Exception:
        return {'U': 0, 'V': 0, 'W': 0.1, 'X': 0, 'Y': 0}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_gsas2(tt, y_obs, sigma, phases, wavelength,
              tt_min=None, tt_max=None, n_bg_coeffs=6,
              max_cycles=10, progress_callback=None):
    """
    Run GSAS-II Rietveld refinement via GSASIIscriptable.

    Parameters
    ----------
    tt, y_obs, sigma : np.ndarray — data arrays
    phases   : list of dicts — must include 'cif_text' for atom positions
    wavelength : float (Å)
    tt_min/max : float — fitting range
    n_bg_coeffs : int — number of background coefficients
    max_cycles : int — max refinement cycles

    Returns dict compatible with Le Bail / Rietveld output (same keys).
    """
    if not _GSASII_AVAILABLE:
        raise RuntimeError(
            f"GSAS-II is not installed or not importable in this Python environment. "
            f"Install with: git clone https://github.com/AdvancedPhotonSource/GSAS-II.git "
            f"&& cd GSAS-II && pip install .\n"
            f"Import error: {_GSASII_IMPORT_ERROR}")

    # Validate: all phases need CIF text
    missing = [ph.get('name', '?') for ph in phases if not ph.get('cif_text')]
    if missing:
        raise ValueError(
            f"GSAS-II refinement requires CIF with atom coordinates for all phases. "
            f"Missing for: {', '.join(missing)}.")

    if tt_min is None: tt_min = float(tt.min())
    if tt_max is None: tt_max = float(tt.max())

    # Apply range mask
    mask = (tt >= tt_min) & (tt <= tt_max)
    tt_r = tt[mask]
    y_r = y_obs[mask]
    sig_r = sigma[mask] if sigma is not None else np.sqrt(np.maximum(y_r, 1.0))

    if progress_callback:
        progress_callback('GSAS-II: setting up project...')

    # ── Create temporary files ───────────────────────────────────────────
    # Use a temp dir inside the app directory to avoid spaces / short-path
    # issues in the user-profile temp folder (GSAS-II can't read files
    # whose path contains spaces).
    _app_tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.pardir, os.pardir, '.gsas_tmp')
    _app_tmp = os.path.normpath(_app_tmp)
    os.makedirs(_app_tmp, exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix='gsas2_', dir=_app_tmp)
    gpx_path = os.path.join(work_dir, 'refine.gpx')
    data_path = os.path.join(work_dir, 'data.xye')
    instprm_path = _write_instprm(work_dir, wavelength)
    _write_xye(data_path, tt_r, y_r, sig_r)

    cif_paths = []
    for i, ph in enumerate(phases):
        # Build a synthetic CIF from the phase dict's (conventional) cell
        # parameters.  This guarantees GSAS-II sees the correct space group
        # and cell geometry even when the original CIF used a primitive
        # setting (common with Materials Project data).
        cif_for_gsas = _build_conventional_cif(ph)
        cif_path = _write_temp_cif(cif_for_gsas, ph.get('name', 'phase'),
                                   work_dir=work_dir, index=i)
        cif_paths.append(cif_path)

    try:
        # ── Build GSAS-II project ────────────────────────────────────────
        try:
            gpx = G2sc.G2Project(newgpx=gpx_path)
        except Exception as e:
            raise RuntimeError(
                f"GSAS-II failed to create a project. This usually means GSAS-II's "
                f"compiled Fortran extensions are not installed (requires a Fortran "
                f"compiler during installation).\n\n"
                f"Try using the built-in Rietveld refinement instead, or install "
                f"GSAS-II via: conda install gsas2pkg -c briantoby\n\n"
                f"Original error: {e}"
            ) from e

        # Add histogram (powder data)
        histogram = gpx.add_powder_histogram(
            data_path, instprm_path,
            databank=None,
        )

        # Set data range
        histogram.data['Limits'] = [[tt_min, tt_max], [tt_min, tt_max]]

        # Set background
        bkg_data = histogram.data['Background']
        bkg_data[0] = ['chebyschev-1', True, n_bg_coeffs,
                        1.0] + [0.0] * (n_bg_coeffs - 1)

        # Add phases from CIF files — use unique phasenames to avoid
        # GSAS-II internal name collisions.
        gsas_phases = []
        used_names = set()
        for i, (ph, cif_path) in enumerate(zip(phases, cif_paths)):
            base_name = ph.get('name', f'Phase_{i+1}')
            phasename = base_name
            if phasename in used_names:
                phasename = f"{base_name}_{i+1}"
            used_names.add(phasename)
            try:
                phase_obj = gpx.add_phase(
                    cif_path,
                    phasename=phasename,
                    histograms=[histogram],
                )
            except Exception as e:
                cif_size = os.path.getsize(cif_path) if os.path.exists(cif_path) else -1
                cif_head = ''
                if os.path.exists(cif_path):
                    with open(cif_path, 'r', encoding='utf-8') as _f:
                        cif_head = _f.read(500)
                raise RuntimeError(
                    f"GSAS-II could not read CIF for phase '{ph.get('name', '?')}' "
                    f"(file: {cif_path}, size: {cif_size} bytes).\n"
                    f"CIF preview:\n{cif_head}\n\n"
                    f"Original error: {e}"
                ) from e
            gsas_phases.append(phase_obj)

        # Track which stage succeeded (for fallback on failure)
        last_good_stage = 0

        # ── Helper to safely run a refinement stage ──────────────────────
        def _safe_refine(stage_name, refinement_dicts, stage_num):
            nonlocal last_good_stage
            try:
                gpx.do_refinements(refinement_dicts)
                last_good_stage = stage_num
                return True
            except Exception as e:
                # GSAS-II internally restores from .bak0.gpx on failure,
                # so the project state reverts to pre-refinement. Safe to continue.
                warnings.warn(f"GSAS-II: {stage_name} failed ({e}). "
                             f"Continuing with results from stage {last_good_stage}.")
                return False

        if progress_callback:
            progress_callback('GSAS-II: stage 1 — refining background + scale...')

        # ── Stage 1: Background + scale (one phase at a time) ────────────
        # Fix all scales first, then refine them one at a time to break
        # the correlation that causes SVD singularities.
        for phase_obj in gsas_phases:
            hapData = list(phase_obj.data['Histograms'].values())[0]
            hapData['Scale'] = [1.0, False]  # start fixed

        # First: refine background only
        _safe_refine('background', [{
            'set': {
                'Background': {'type': 'chebyschev-1', 'refine': True,
                                'no. coeffs': n_bg_coeffs},
            },
            'cycles': min(max_cycles, 5),
        }], 1)

        # Then: refine each phase's scale one at a time
        for idx, phase_obj in enumerate(gsas_phases):
            hapData = list(phase_obj.data['Histograms'].values())[0]
            hapData['Scale'] = [hapData['Scale'][0], True]  # turn on
            _safe_refine(f'scale (phase {idx})', [{
                'set': {},
                'cycles': 3,
            }], 1)

        # Now refine all scales together (they have good starting values)
        _safe_refine('all scales', [{
            'set': {'Scale': True},
            'cycles': min(max_cycles, 5),
        }], 1)

        if progress_callback:
            progress_callback('GSAS-II: stage 2 — refining profile parameters...')

        # ── Stage 2: Profile parameters ──────────────────────────────────
        _safe_refine('profile', [{
            'set': {
                'Instrument Parameters': ['U', 'V', 'W', 'X', 'Y'],
            },
            'cycles': min(max_cycles, 5),
        }], 2)

        if progress_callback:
            progress_callback('GSAS-II: stage 3 — refining cell parameters...')

        # ── Stage 3: Cell parameters (one phase at a time) ───────────────
        # Refining all cells at once with many atoms can cause arccos
        # errors when cell angles go unphysical. Do it per phase.
        # For cubic phases, lock angles to 90° to prevent arccos errors.
        for idx, (phase_obj, ph_input) in enumerate(zip(gsas_phases, phases)):
            sys_ = (ph_input.get('system') or 'triclinic').lower()
            # For high-symmetry systems, clamp angles before refining
            # to prevent them from drifting to unphysical values.
            if sys_ in ('cubic', 'tetragonal', 'orthorhombic'):
                try:
                    _cell = phase_obj.data['General']['Cell']
                    _cell[4] = 90.0  # alpha
                    _cell[5] = 90.0  # beta
                    _cell[6] = 90.0  # gamma
                except Exception:
                    pass
            elif sys_ in ('hexagonal', 'trigonal'):
                try:
                    _cell = phase_obj.data['General']['Cell']
                    _cell[4] = 90.0   # alpha
                    _cell[5] = 90.0   # beta
                    _cell[6] = 120.0  # gamma
                except Exception:
                    pass
            phase_obj.set_refinements({'Cell': True})
            ok = _safe_refine(f'cell (phase {idx})', [{
                'set': {},
                'cycles': min(max_cycles, 8),
            }], 3)
            if not ok:
                # Turn cell refinement back off for this phase
                phase_obj.set_refinements({'Cell': False})

        if progress_callback:
            progress_callback('GSAS-II: stage 4 — refining atomic displacement...')

        # ── Stage 4: Atomic displacement (Uiso) ─────────────────────────
        for idx, phase_obj in enumerate(gsas_phases):
            try:
                phase_obj.set_refinements({'Atoms': {'all': 'U'}})
            except Exception:
                pass
        _safe_refine('Uiso', [{
            'set': {},
            'cycles': min(max_cycles, 5),
        }], 4)

        if progress_callback:
            progress_callback('GSAS-II: extracting results...')

        # ── Extract results ──────────────────────────────────────────────
        # Get calculated pattern
        y_calc_full = np.array(histogram.getdata('ycalc'))
        y_obs_full = np.array(histogram.getdata('yobs'))
        y_bg_full = np.array(histogram.getdata('background'))
        tt_full = np.array(histogram.getdata('x'))

        # Trim to our range
        rmask = (tt_full >= tt_min) & (tt_full <= tt_max)
        tt_out = tt_full[rmask]
        y_obs_out = y_obs_full[rmask]
        y_calc_out = y_calc_full[rmask]
        y_bg_out = y_bg_full[rmask]
        diff_out = y_obs_out - y_calc_out

        # Weights for statistics
        weights_out = 1.0 / np.maximum(
            np.where(sig_r is not None, sig_r**2,
                     np.maximum(y_obs_out, 1.0)), 1e-6)

        # Compute statistics
        n_params_est = sum(
            len(list(phase_obj.atoms())) + 7  # atoms + cell + scale + profile
            for phase_obj in gsas_phases
        ) + n_bg_coeffs + 1
        stats = compute_fit_statistics(y_obs_out, y_calc_out,
                                        weights_out, n_params_est)

        # Get GSAS-II's own R-factors if available
        try:
            gsas_stats = histogram.get_statistics()
            if gsas_stats:
                stats['Rwp'] = round(gsas_stats.get('Rwp', stats['Rwp']), 2)
                stats['Rp'] = round(gsas_stats.get('Rp', stats['Rp']), 2)
        except Exception:
            pass

        # Extract instrument params
        inst = _extract_instrument_params(histogram)

        # Zero shift
        try:
            zero_shift = float(
                histogram.data['Instrument Parameters'][0].get('Zero', [0, 0])[1])
        except Exception:
            zero_shift = 0.0

        # ── Per-phase results ────────────────────────────────────────────
        phase_patterns = []
        phase_results = []

        # ── Weight fractions via Hill & Howard (1987) ────────────────────
        # W_α = S_α · Z_α · M_α · V_α  /  Σ(S_i · Z_i · M_i · V_i)
        # If Z or M are unavailable, fall back to raw scale normalisation.
        raw_scales = {}
        try:
            for phase_obj in gsas_phases:
                hapData = list(phase_obj.data['Histograms'].values())[0]
                scale_entry = hapData.get('Scale', [1.0])
                scale_val = scale_entry[0] if isinstance(scale_entry, (list, tuple)) else float(scale_entry)
                raw_scales[phase_obj.name] = scale_val
        except Exception as e:
            warnings.warn(f"GSAS-II: could not read scale factors: {e}")

        zmv_values = {}
        use_zmv = True
        for ph_input, phase_obj in zip(phases, gsas_phases):
            S = raw_scales.get(phase_obj.name, 1.0)
            _cell = phase_obj.data['General']['Cell']
            V_ph = cell_volume(float(_cell[1]), float(_cell[2]), float(_cell[3]),
                               float(_cell[4]), float(_cell[5]), float(_cell[6]))
            Z = ph_input.get('Z')
            M = molar_mass_from_formula(ph_input.get('formula', ''))
            if Z and M and V_ph > 0:
                zmv_values[phase_obj.name] = float(S) * float(Z) * float(M) * V_ph
            else:
                use_zmv = False
                break

        if not use_zmv:
            warnings.warn("GSAS-II: Z or molar mass unavailable for one or more "
                         "phases — falling back to raw scale factor weighting.")
            zmv_values = dict(raw_scales)

        total_zmv = sum(zmv_values.values()) or 1e-10

        # Warn if all scale factors are identical (common with failed refinement)
        scale_vals = list(raw_scales.values())
        if len(scale_vals) >= 2 and len(set(round(v, 6) for v in scale_vals)) == 1:
            warnings.warn("GSAS-II: all phase scale factors are identical — "
                         "refinement may not have converged properly.")

        for i, (ph, phase_obj) in enumerate(zip(phases, gsas_phases)):
            # Cell parameters — read directly from GSAS-II data structure
            # to avoid get_cell() API differences across versions.
            # General['Cell'] = [mustRefine, a, b, c, alpha, beta, gamma, volume]
            _cell = phase_obj.data['General']['Cell']
            a, b, c = float(_cell[1]), float(_cell[2]), float(_cell[3])
            alpha, beta, gamma = float(_cell[4]), float(_cell[5]), float(_cell[6])
            V = cell_volume(a, b, c, alpha, beta, gamma)

            # Profile / size info
            prof = _extract_profile_params(phase_obj)
            cryst_A = prof.get('crystallite_size_A')

            # If GSAS-II size extraction failed, estimate from Y
            if cryst_A is None and inst['Y'] > 0.01:
                cryst_A = size_from_Y(inst['Y'], wavelength)

            # FWHM and eta at a representative angle (40°)
            fwhm_rep, eta_rep = tch_fwhm_eta(
                40.0, inst['U'], inst['V'], inst['W'],
                inst['X'], inst['Y'])

            # Weight fraction (Hill & Howard or raw-scale fallback)
            scale_val = raw_scales.get(phase_obj.name, 1.0)
            zmv_val = zmv_values.get(phase_obj.name, scale_val)
            wt_pct = (zmv_val / total_zmv) * 100 if total_zmv > 0 else 0

            # Tick positions (reflection list)
            # GSAS-II stores reflections in several possible locations
            # depending on the version and how the phase was added.
            # Standard reflection array: [h,k,l,mult,d,2theta,sig,gam,Fobs²,Fcalc²,...]
            tick_positions = []
            try:
                hapData = list(phase_obj.data['Histograms'].values())[0]
                # Try 'Reflection Lists' first (newer GSAS-II)
                refDict = hapData.get('Reflection Lists', {})
                if not refDict:
                    # Fallback: older GSAS-II stores it directly
                    refDict = hapData.get('RefList', {})
                if isinstance(refDict, dict) and refDict:
                    for key in refDict:
                        rlist = refDict[key]
                        # rlist may be a dict with 'RefList' key or a direct list
                        if isinstance(rlist, dict):
                            rlist = rlist.get('RefList', [])
                        if isinstance(rlist, (list, np.ndarray)) and len(rlist) > 0:
                            # Determine 2theta column index by checking if
                            # index 5 contains plausible 2theta values (0-180°).
                            # Fall back to index 4 if not.
                            tt_col = 5
                            try:
                                test_val = float(rlist[0][5])
                                if not (0 < test_val < 180):
                                    tt_col = 4
                            except (IndexError, TypeError, ValueError):
                                tt_col = 4
                            for ref in rlist:
                                try:
                                    tt_ref = float(ref[tt_col])
                                    if tt_min <= tt_ref <= tt_max:
                                        tick_positions.append(round(tt_ref, 3))
                                except (IndexError, TypeError, ValueError):
                                    continue
                            if tick_positions:
                                break  # found reflections, stop searching
            except Exception as e:
                warnings.warn(f"GSAS-II: could not extract tick positions for "
                            f"'{ph.get('name', '?')}': {e}")

            # Fallback: compute tick positions from cell parameters + space group
            # Always pass sites so structure-factor-extinct peaks are excluded.
            if not tick_positions:
                try:
                    from .crystallography import generate_reflections, parse_cif
                    sys_ = (ph.get('system') or 'triclinic').lower()
                    sg = ph.get('spacegroup_number', 1)
                    # Try to get atom sites from the CIF for accurate F² filtering
                    sites = None
                    cif_text = ph.get('cif_text', '')
                    if cif_text:
                        try:
                            parsed = parse_cif(cif_text)
                            sites = parsed.get('sites') or None
                        except Exception:
                            pass
                    fallback_refs = generate_reflections(
                        a, b, c, alpha, beta, gamma, sys_, sg,
                        wavelength, tt_min, tt_max, hkl_max=12,
                        sites=sites)
                    tick_positions = [round(r[0], 3) for r in fallback_refs]
                except Exception:
                    pass

            # B_iso (average over atoms)
            b_iso_avg = 0.5
            try:
                atoms = list(phase_obj.atoms())
                if atoms:
                    uisos = [at.uiso for at in atoms if hasattr(at, 'uiso')]
                    if uisos:
                        b_iso_avg = 8 * math.pi**2 * np.mean(uisos)
            except Exception:
                pass

            # Phase pattern: we don't get per-phase patterns from GSAS-II
            # easily in scriptable mode. Approximate by scaling the total
            # calculated pattern by weight fraction.
            phase_pat = (y_calc_out - y_bg_out) * (wt_pct / 100.0)
            phase_patterns.append(phase_pat.tolist())

            phase_results.append({
                'name':              ph.get('name', f'Phase {i+1}'),
                'cod_id':            ph.get('cod_id', ph.get('mp_id', '')),
                'formula':           ph.get('formula', ''),
                'a': round(a, 5), 'b': round(b, 5), 'c': round(c, 5),
                'alpha': round(alpha, 3), 'beta': round(beta, 3),
                'gamma': round(gamma, 3),
                'system':            (ph.get('system') or 'triclinic').lower(),
                'spacegroup_number': ph.get('spacegroup_number', 1),
                'spacegroup':        ph.get('spacegroup', ''),
                'scale':             round(scale_val, 5),
                'B_iso':             round(b_iso_avg, 4),
                'U': round(inst['U'], 5), 'V': round(inst['V'], 5),
                'W': round(inst['W'], 5),
                'X': round(inst['X'], 5), 'Y': round(inst['Y'], 5),
                'eta_at_strongest':  round(eta_rep, 3),
                'fwhm_deg':          round(fwhm_rep, 4),
                'crystallite_size_A':  round(cryst_A, 1) if cryst_A else None,
                'crystallite_size_nm': round(cryst_A / 10, 2) if cryst_A else None,
                'weight_fraction_%':   round(wt_pct, 1),
                'n_reflections':       len(tick_positions),
                'tick_positions':      tick_positions,
                'seeded_by':           'gsas2',
            })

        result = {
            'tt':             tt_out.tolist(),
            'y_obs':          y_obs_out.tolist(),
            'y_calc':         y_calc_out.tolist(),
            'y_background':   y_bg_out.tolist(),
            'phase_patterns': phase_patterns,
            'residuals':      diff_out.tolist(),
            'statistics':     stats,
            'phase_results':  phase_results,
            'zero_shift':     round(zero_shift, 5),
            'wavelength':     wavelength,
            'pymatgen_used':  False,
            'method':         'GSAS-II',
        }

        return result

    finally:
        # Clean up temporary files
        for p in cif_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        try:
            os.unlink(data_path)
        except OSError:
            pass
        # Keep .gpx for debugging; clean up on next run
