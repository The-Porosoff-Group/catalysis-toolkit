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
        generate_reflections, parse_cif, caglioti_fwhm,
        expand_sites_from_cif,
    )
except ImportError:
    from crystallography import (
        compute_fit_statistics, cell_volume, molar_mass_from_formula,
        tch_fwhm_eta, size_from_Y, scherrer_size,
        generate_reflections, parse_cif, caglioti_fwhm,
        expand_sites_from_cif,
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


def _get_expanded_sites(cif_text, spacegroup_number=None):
    """Get symmetry-expanded unit-cell sites for structure factor computation.

    Uses the shared expand_sites_from_cif (pymatgen) first, then falls back
    to raw parse_cif (may be asymmetric unit only).
    """
    if not cif_text:
        return None
    sites = expand_sites_from_cif(cif_text)
    if sites:
        return sites
    # Fallback: raw parse_cif (may be asymmetric unit only)
    try:
        parsed = parse_cif(cif_text)
        return parsed.get('sites') or None
    except Exception:
        return None


def _reduce_to_asymmetric_unit(cif_text):
    """Reduce a full-unit-cell CIF to asymmetric unit for GSAS-II.

    GSAS-II applies symmetry operations itself, so it expects only the
    asymmetric unit in the CIF.  If we give it the full unit cell plus
    the space group, it over-expands and generates wrong reflections.

    Uses pymatgen's symmetrized structure to get the unique sites.
    Falls back to the raw sites if pymatgen is unavailable.
    """
    if not cif_text:
        return []

    try:
        from pymatgen.io.cif import CifParser
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        try:
            parser = CifParser.from_str(cif_text, occupancy_tolerance=100.0)
        except (AttributeError, TypeError):
            import tempfile as _tf, os as _os
            _fd, _tmp = _tf.mkstemp(suffix='.cif')
            with _os.fdopen(_fd, 'w') as _f:
                _f.write(cif_text)
            parser = CifParser(_tmp, occupancy_tolerance=100.0)
            _os.unlink(_tmp)
        structs = parser.parse_structures(primitive=False)
        if structs:
            sga = SpacegroupAnalyzer(structs[0], symprec=0.1)
            sym_struct = sga.get_symmetrized_structure()
            sites = []
            for equiv_sites in sym_struct.equivalent_sites:
                site = equiv_sites[0]  # take first of each equivalent group
                frac = site.frac_coords % 1.0
                el = str(site.specie)
                sites.append((el, float(frac[0]), float(frac[1]),
                              float(frac[2]), 1.0))
            if sites:
                return sites
    except Exception:
        pass

    # Fallback: raw parse_cif
    try:
        parsed = parse_cif(cif_text)
        return parsed.get('sites') or []
    except Exception:
        return []


def _build_conventional_cif(ph):
    """
    Build a synthetic CIF string using the phase dict's (conventional) cell
    parameters and atom sites.

    This ensures GSAS-II always sees a CIF consistent with the conventional
    cell, even when the original CIF used a primitive setting (common with
    Materials Project data).  The space group is written explicitly so that
    GSAS-II applies the correct cell-parameter constraints.
    """
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

    # Get atom sites reduced to the asymmetric unit.
    # GSAS-II applies space-group symmetry itself, so we must NOT give
    # it the full unit cell — that would cause double-counting and wrong
    # reflections (e.g. ghost peaks at ~25° for A15-W).
    cif_text = ph.get('cif_text', '')
    sites = _reduce_to_asymmetric_unit(cif_text) if cif_text else []

    lines = [
        'data_phase',
        f"_chemical_formula_sum '{formula}'" if formula else '',
        f"_cell_formula_units_Z {Z}" if Z else '',
        f'_cell_length_a {a:.5f}',
        f'_cell_length_b {b:.5f}',
        f'_cell_length_c {c:.5f}',
        f'_cell_angle_alpha {al:.3f}',
        f'_cell_angle_beta {be:.3f}',
        f'_cell_angle_gamma {ga:.3f}',
        f'_symmetry_Int_Tables_number {sg}',
        f"_symmetry_space_group_name_H-M '{hm}'",
        f"_space_group_IT_number {sg}",
        f"_space_group_name_H-M_alt '{hm}'",
    ]

    if sites:
        lines += [
            '',
            'loop_',
            '_atom_site_label',
            '_atom_site_type_symbol',
            '_atom_site_fract_x',
            '_atom_site_fract_y',
            '_atom_site_fract_z',
            '_atom_site_occupancy',
        ]
        site_counts = {}
        for el, x, y, z, occ in sites:
            # Generate unique labels like W1, W2, O1, O2...
            site_counts[el] = site_counts.get(el, 0) + 1
            label = f'{el}{site_counts[el]}'
            # Clamp negative zeros to 0.0 — GSAS-II CIF reader rejects "-0.000000"
            x = abs(x) if abs(x) < 1e-8 else x
            y = abs(y) if abs(y) < 1e-8 else y
            z = abs(z) if abs(z) < 1e-8 else z
            lines.append(f'{label}  {el}  {x:.6f}  {y:.6f}  {z:.6f}  {occ:.4f}')

    return '\n'.join(ln for ln in lines if ln is not None) + '\n'


# ─────────────────────────────────────────────────────────────────────────────
# NON-NEGATIVE LEAST SQUARES (NNLS)
# ─────────────────────────────────────────────────────────────────────────────

def _nnls(A, b):
    """Solve  min‖Ax − b‖²  subject to  x ≥ 0.

    Uses scipy.optimize.nnls if available, otherwise falls back to the
    Lawson-Hanson active-set algorithm (pure numpy).
    """
    try:
        from scipy.optimize import nnls as _scipy_nnls
        x, _ = _scipy_nnls(A, b)
        return x
    except ImportError:
        pass
    # ── Fallback: Lawson-Hanson active-set NNLS ──────────────────────
    m, n = A.shape
    P = set()               # passive set (variables free to be > 0)
    Z = set(range(n))       # zero set (variables fixed at 0)
    x = np.zeros(n)
    max_outer = 3 * n
    for _ in range(max_outer):
        w = A.T @ (b - A @ x)  # negative gradient
        if not Z or max(w[j] for j in Z) <= 1e-10:
            break
        # Move variable with largest positive gradient into passive set
        t = max(Z, key=lambda j: w[j])
        P.add(t); Z.discard(t)
        # Inner loop: solve unconstrained on P, handle negatives
        for __ in range(max_outer):
            cols = sorted(P)
            s = np.zeros(n)
            s[cols] = np.linalg.lstsq(A[:, cols], b, rcond=None)[0]
            if all(s[j] >= 0 for j in cols):
                x = s
                break
            # Find the tightest step that hits zero
            alpha = 1.0
            j_remove = None
            for j in cols:
                if s[j] < 0 and x[j] > x[j] - s[j]:
                    a = x[j] / (x[j] - s[j])
                    if a < alpha:
                        alpha = a
                        j_remove = j
            if j_remove is None:
                x = s
                break
            x = x + alpha * (s - x)
            x[j_remove] = 0.0
            Z.add(j_remove); P.discard(j_remove)
        else:
            x = np.maximum(x, 0.0)  # safety clamp
    return x


# ─────────────────────────────────────────────────────────────────────────────
# PER-PHASE PATTERN FROM GSAS-II REFLECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _compute_raw_phase_profile(tt_arr, refs, U_deg, V_deg, W_deg,
                                X_deg, Y_deg):
    """Compute a raw (unscaled) profile shape for one phase.

    Returns an array of the same length as *tt_arr* whose values are
    proportional to the diffracted intensity at each 2θ point.  The
    absolute scale is arbitrary — what matters is the *relative* shape
    compared to other phases so that the proportional decomposition
    can correctly partition the total pattern.

    refs : list of [two_theta, d, (h,k,l), intensity_weight] from
           generate_reflections (with structure-factor filtering).
    """
    n = len(tt_arr)
    pattern = np.zeros(n, dtype=np.float64)

    use_tch = (X_deg != 0.0 or Y_deg != 0.0)

    for tt_pos, d, hkl, weight in refs:
        if weight <= 0:
            continue

        if use_tch:
            fwhm, eta = tch_fwhm_eta(tt_pos, U_deg, V_deg, W_deg,
                                      X_deg, Y_deg)
        else:
            fwhm = max(caglioti_fwhm(tt_pos, U_deg, V_deg, W_deg), 0.005)
            eta = 0.5

        sigma_g = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        gamma_l = fwhm / 2.0
        # Tight window (3× FWHM) to prevent cross-phase bleed
        window = 3.0 * fwhm

        msk = np.abs(tt_arr - tt_pos) < window
        if not msk.any():
            continue

        dx = tt_arr[msk] - tt_pos
        G = np.exp(-0.5 * (dx / sigma_g) ** 2)
        L = 1.0 / (1.0 + (dx / gamma_l) ** 2)
        prof = eta * L + (1.0 - eta) * G
        # Normalise
        area = np.trapz(prof, tt_arr[msk]) if np.trapz(prof, tt_arr[msk]) > 0 else 1.0
        pattern[msk] += weight * prof / area

    return pattern


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

        # Add phases from CIF files — always embed the space group number in
        # the GSAS-II phasename so that two phases with the same formula
        # (e.g. W Pm-3n vs W Im-3m) are never treated as the same phase.
        gsas_phases = []
        used_names = set()
        for i, (ph, cif_path) in enumerate(zip(phases, cif_paths)):
            formula  = ph.get('formula', '') or ph.get('name', '') or 'Phase'
            sg_num   = ph.get('spacegroup_number', 1)
            # Always include SG number for unambiguous internal naming
            phasename = f"{formula}_sg{sg_num}"
            # Deduplicate in case two phases have the same formula + SG (rare)
            if phasename in used_names:
                phasename = f"{phasename}_{i+1}"
            used_names.add(phasename)
            try:
                phase_obj = gpx.add_phase(
                    cif_path,
                    phasename=phasename,
                    histograms=[histogram],
                )
            except Exception as e1:
                # Synthetic CIF failed — fall back to original CIF text
                orig_cif = ph.get('cif_text', '')
                if orig_cif:
                    try:
                        orig_path = _write_temp_cif(
                            orig_cif, ph.get('name', 'phase'),
                            work_dir=work_dir, index=100+i)
                        cif_paths.append(orig_path)  # for cleanup
                        phase_obj = gpx.add_phase(
                            orig_path,
                            phasename=phasename,
                            histograms=[histogram],
                        )
                        warnings.warn(
                            f"Synthetic CIF failed for '{ph.get('name', '?')}', "
                            f"using original CIF text (error: {e1})")
                    except Exception as e2:
                        cif_size = os.path.getsize(cif_path) if os.path.exists(cif_path) else -1
                        cif_head = ''
                        if os.path.exists(cif_path):
                            with open(cif_path, 'r', encoding='utf-8') as _f:
                                cif_head = _f.read(500)
                        raise RuntimeError(
                            f"GSAS-II could not read CIF for phase '{ph.get('name', '?')}' "
                            f"(file: {cif_path}, size: {cif_size} bytes).\n"
                            f"CIF preview:\n{cif_head}\n\n"
                            f"Original error: {e2}"
                        ) from e2
                else:
                    cif_size = os.path.getsize(cif_path) if os.path.exists(cif_path) else -1
                    cif_head = ''
                    if os.path.exists(cif_path):
                        with open(cif_path, 'r', encoding='utf-8') as _f:
                            cif_head = _f.read(500)
                    raise RuntimeError(
                        f"GSAS-II could not read CIF for phase '{ph.get('name', '?')}' "
                        f"(file: {cif_path}, size: {cif_size} bytes).\n"
                        f"CIF preview:\n{cif_head}\n\n"
                        f"Original error: {e1}"
                    ) from e1
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
        raw_phase_profiles = []   # for proportional decomposition
        phase_refs_list = []      # reflection lists per phase

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
        print(f"GSAS-II scale factors: {raw_scales}", flush=True)
        print(f"GSAS-II ZMV values: {zmv_values}", flush=True)
        print(f"GSAS-II use_zmv: {use_zmv}", flush=True)

        # Warn if all scale factors are identical (common with failed refinement)
        scale_vals = list(raw_scales.values())
        if len(scale_vals) >= 2 and len(set(round(v, 6) for v in scale_vals)) == 1:
            warnings.warn("GSAS-II: all phase scale factors are identical — "
                         "refinement may not have converged properly.")

        # ── Extract GSAS-II RefList for physics-based profile generation ────
        # The RefList contains GSAS-II's refined Fc² values for each reflection,
        # giving more accurate phase envelopes than our generate_reflections output.
        # RefList columns: 0=h, 1=k, 2=l, 3=mult, 4=dsp, 5=2theta, 6=Fo²,
        #                  7=sig, 8=Fc², ...
        gsas_refs = {}   # phase_name → [(two_theta, d, (h,k,l), mult*Fc²)]
        try:
            raw_refl_lists = histogram.data.get('Reflection Lists', {})
            for ph_name, refl_data in raw_refl_lists.items():
                ref_arr = refl_data.get('RefList')
                if ref_arr is not None and len(ref_arr) > 0 and ref_arr.shape[1] > 8:
                    refs = []
                    for row in ref_arr:
                        h, k, l = int(row[0]), int(row[1]), int(row[2])
                        mult      = float(row[3])
                        d_sp      = float(row[4])
                        two_theta = float(row[5])
                        fc2       = float(row[8])   # Fc²
                        weight    = mult * fc2
                        if weight > 0 and tt_min <= two_theta <= tt_max:
                            refs.append((two_theta, d_sp, (h, k, l), weight))
                    if refs:
                        gsas_refs[ph_name] = refs
                        print(f"  Loaded GSAS-II RefList for '{ph_name}': "
                              f"{len(refs)} reflections with Fc²", flush=True)
        except Exception as e:
            print(f"  (Could not extract GSAS-II RefList: {e})", flush=True)

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

            # GSAS-II stores profile params in centidegrees² (sig) and
            # centidegrees (gam).  Convert to degrees² / degrees for
            # Caglioti/TCH compatibility with our functions.
            U_deg = inst['U'] / 100.0   # centideg² → deg²
            V_deg = inst['V'] / 100.0
            W_deg = inst['W'] / 100.0
            X_deg = inst['X'] / 1.0     # X is already in degrees (or dimensionless)
            Y_deg = inst['Y'] / 1.0

            # If GSAS-II size extraction failed, estimate from Y
            if cryst_A is None and Y_deg > 0.01:
                cryst_A = size_from_Y(Y_deg, wavelength)

            # FWHM and eta at a representative angle (40°)
            fwhm_rep, eta_rep = tch_fwhm_eta(
                40.0, U_deg, V_deg, W_deg, X_deg, Y_deg)

            # Weight fraction (Hill & Howard or raw-scale fallback)
            scale_val = raw_scales.get(phase_obj.name, 1.0)
            zmv_val = zmv_values.get(phase_obj.name, scale_val)
            wt_pct = (zmv_val / total_zmv) * 100 if total_zmv > 0 else 0

            # ── Generate reflections using our own crystallography code ──
            # This is the same approach used by the in-house Rietveld and
            # is independent of GSAS-II's internal data structures.
            # Each phase gets its own reflection list based on its own
            # space group, cell parameters, and atom sites from CIF.
            sys_ = (ph.get('system') or 'triclinic').lower()
            sg = ph.get('spacegroup_number', 1)
            # Get symmetry-expanded sites for correct structure factors
            sites = _get_expanded_sites(ph.get('cif_text', ''), sg)

            phase_refs = generate_reflections(
                a, b, c, alpha, beta, gamma, sys_, sg,
                wavelength, tt_min, tt_max, hkl_max=12,
                sites=sites)
            tick_positions = [round(r[0], 3) for r in phase_refs]

            # Raw profile shape for scale-fitting and decomposition.
            # Prefer GSAS-II's refined Fc² values (from RefList) over our
            # generate_reflections output — the Rietveld code has the correct
            # structure factors and systematic extinctions baked in.
            refs_for_profile = gsas_refs.get(phase_obj.name, phase_refs)
            raw_prof = _compute_raw_phase_profile(
                tt_out, refs_for_profile, U_deg, V_deg, W_deg, X_deg, Y_deg)
            raw_phase_profiles.append(raw_prof)
            phase_refs_list.append(phase_refs)

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

            # Build a display name that always includes the space group symbol
            # so that phases with the same formula are clearly distinguishable.
            _sg_sym = (ph.get('spacegroup', '') or
                       _SG_HM.get(ph.get('spacegroup_number', 1), '') or
                       f"SG{ph.get('spacegroup_number', 1)}")
            _base_name = ph.get('name') or ph.get('formula') or f'Phase {i+1}'
            # Only append SG if not already present in the name
            if _sg_sym and _sg_sym.replace(' ', '') not in _base_name.replace(' ', ''):
                display_name = f"{_base_name} {_sg_sym}"
            else:
                display_name = _base_name

            phase_results.append({
                'name':              display_name,
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
                'U': round(U_deg, 5), 'V': round(V_deg, 5),
                'W': round(W_deg, 5),
                'X': round(X_deg, 5), 'Y': round(Y_deg, 5),
                'eta_at_strongest':  round(eta_rep, 3),
                'fwhm_deg':          round(fwhm_rep, 4),
                'crystallite_size_A':  round(cryst_A, 1) if cryst_A else None,
                'crystallite_size_nm': round(cryst_A / 10, 2) if cryst_A else None,
                'weight_fraction_%':   round(wt_pct, 1),
                'n_reflections':       len(tick_positions),
                'tick_positions':      tick_positions,
                'seeded_by':           'gsas2',
            })

        # ── Phase decomposition via unique-peak scale fitting ───────────
        # GSAS-II's scale factors can be unreliable when peaks overlap
        # (SVD singularity seen in the log).  Instead, we determine each
        # phase's absolute scale by looking at peaks UNIQUE to that phase
        # (no other phase has a reflection within ±MIN_SEP degrees).
        # At those points all signal belongs to that phase:
        #     scale_p = median( y_obs_at_peak / raw_profile_at_peak )
        # We then render each phase's pattern INDEPENDENTLY as
        #     phase_pattern_p = scale_p × raw_profile_p
        # This gives physically distinct envelopes (correct peak positions
        # per phase) rather than just proportional slices of the total.
        total_above_bg = np.maximum(y_calc_out - y_bg_out, 0.0)
        if raw_phase_profiles:
            n_ph = len(raw_phase_profiles)
            # Collect all reflection 2θ positions per phase
            all_peak_pos = []
            for refs in phase_refs_list:
                all_peak_pos.append([r[0] for r in refs])

            # Fit scale from unique peaks
            MIN_SEP = 1.5  # degrees — peaks closer than this are "shared"
            fitted_scales = np.zeros(n_ph)
            for i in range(n_ph):
                ratios = []
                for pos in all_peak_pos[i]:
                    if pos < tt_min or pos > tt_max:
                        continue
                    # Check if this peak is unique (no other phase nearby)
                    is_unique = True
                    for j in range(n_ph):
                        if j == i:
                            continue
                        for other_pos in all_peak_pos[j]:
                            if abs(pos - other_pos) < MIN_SEP:
                                is_unique = False
                                break
                        if not is_unique:
                            break
                    if is_unique:
                        idx_peak = np.argmin(np.abs(tt_out - pos))
                        raw_val = raw_phase_profiles[i][idx_peak]
                        obs_val = total_above_bg[idx_peak]
                        if raw_val > 1e-6 and obs_val > 0:
                            ratios.append(obs_val / raw_val)
                if ratios:
                    fitted_scales[i] = np.median(ratios)
                    ph_name = phase_results[i]['name'] if i < len(phase_results) else f'Phase {i}'
                    print(f"  Phase {i} ({ph_name}): unique-peak scale={fitted_scales[i]:.6e} "
                          f"(from {len(ratios)} unique peaks)", flush=True)

            # If any phase has no unique peaks, fall back to NNLS for that phase
            missing = [i for i in range(n_ph) if fitted_scales[i] <= 0]
            if missing:
                print(f"  Phases {missing} have no unique peaks — using NNLS fallback", flush=True)
                A = np.column_stack(raw_phase_profiles)
                x = _nnls(A, total_above_bg)
                for i in missing:
                    fitted_scales[i] = x[i]

            # Build scaled profiles that preserve each phase's own peak
            # shapes.  Each raw_scaled[i] has the correct peak positions
            # and relative intensities from that phase's own reflections;
            # the fitted_scales make the absolute magnitude match the
            # observed data at that phase's unique peaks.
            raw_scaled = [fitted_scales[i] * raw_phase_profiles[i] for i in range(n_ph)]
            total_raw = sum(raw_scaled)

            # Global rescale so sum(phase_patterns) == total_above_bg in
            # total integrated intensity, but WITHOUT pointwise
            # normalisation (which destroys per-phase peak shapes when
            # phases share peak positions).
            total_raw_sum = np.sum(total_raw) or 1e-30
            total_obs_sum = np.sum(total_above_bg) or 1e-30
            global_scale = total_obs_sum / total_raw_sum
            weighted = [global_scale * rs for rs in raw_scaled]

            # Soft clip: where the stacked sum exceeds total_above_bg,
            # scale all phases down proportionally at that point to avoid
            # the fills extending beyond the observed envelope.
            stacked = sum(weighted)
            excess = np.maximum(stacked - total_above_bg, 0.0)
            clip_factor = np.where(
                stacked > 1e-30,
                np.maximum(1.0 - excess / stacked, 0.0),
                1.0,
            )
            weighted = [w * clip_factor for w in weighted]

            # Compute weight fractions from integrated intensities
            total_integ = sum(np.sum(wp) for wp in weighted) or 1e-30
            for i_wp, wp in enumerate(weighted):
                integ_frac = np.sum(wp) / total_integ * 100
                print(f"  Phase {i_wp}: integrated fraction = {integ_frac:.1f}%", flush=True)
                phase_patterns.append(wp.tolist())
                # Update weight fractions in phase_results
                if i_wp < len(phase_results):
                    phase_results[i_wp]['weight_fraction_%'] = round(integ_frac, 1)
        else:
            # Single-phase fallback
            phase_patterns.append(total_above_bg.tolist())

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
