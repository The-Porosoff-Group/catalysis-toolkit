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
    # Triclinic / Monoclinic
    1: 'P 1', 2: 'P -1', 3: 'P 2', 4: 'P 21', 5: 'C 2',
    6: 'P m', 7: 'P c', 8: 'C m', 9: 'C c', 10: 'P 2/m',
    11: 'P 21/m', 12: 'C 2/m', 13: 'P 2/c', 14: 'P 21/c', 15: 'C 2/c',
    # Orthorhombic (common catalyst phases)
    16: 'P 2 2 2', 19: 'P 21 21 21', 26: 'P m c 21', 29: 'P c a 21',
    31: 'P m n 21', 33: 'P n a 21', 36: 'C m c 21', 40: 'A m a 2',
    44: 'I m m 2', 46: 'I m a 2',
    47: 'P m m m', 51: 'P m m a', 55: 'P b a m', 57: 'P b c m',
    58: 'P n n m', 59: 'P m m n', 60: 'P b c n', 61: 'P b c a',
    62: 'P n m a', 63: 'C m c m', 64: 'C m c a', 65: 'C m m m',
    66: 'C c c m', 69: 'F m m m', 70: 'F d d d', 71: 'I m m m',
    72: 'I b a m', 74: 'I m m a',
    # Tetragonal
    75: 'P 4', 82: 'I -4', 83: 'P 4/m', 84: 'P 42/m',
    85: 'P 4/n', 87: 'I 4/m', 88: 'I 41/a',
    99: 'P 4 m m', 107: 'I 4 m m', 115: 'P -4 m 2',
    119: 'I -4 m 2', 121: 'I -4 2 m', 122: 'I -4 2 d',
    123: 'P 4/m m m', 129: 'P 4/n m m', 131: 'P 42/m m c',
    136: 'P 42/m n m', 139: 'I 4/m m m', 140: 'I 4/m c m',
    141: 'I 41/a m d', 142: 'I 41/a c d',
    # Trigonal / Hexagonal
    143: 'P 3', 146: 'R 3', 147: 'P -3', 148: 'R -3',
    150: 'P 3 2 1', 152: 'P 31 2 1', 154: 'P 32 2 1',
    155: 'R 3 2', 156: 'P 3 m 1', 157: 'P 3 1 m',
    160: 'R 3 m', 161: 'R 3 c', 162: 'P -3 1 m', 163: 'P -3 1 c',
    164: 'P -3 m 1', 165: 'P -3 c 1', 166: 'R -3 m', 167: 'R -3 c',
    168: 'P 6', 173: 'P 63', 174: 'P -6', 175: 'P 6/m', 176: 'P 63/m',
    183: 'P 6 m m', 186: 'P 63 m c', 187: 'P -6 m 2', 189: 'P -6 2 m',
    191: 'P 6/m m m', 193: 'P 63/m c m', 194: 'P 63/m m c',
    # Cubic
    195: 'P 2 3', 196: 'F 2 3', 197: 'I 2 3', 198: 'P 21 3',
    199: 'I 21 3', 200: 'P m -3', 201: 'P n -3', 202: 'F m -3',
    203: 'F d -3', 204: 'I m -3', 205: 'P a -3', 206: 'I a -3',
    207: 'P 4 3 2', 209: 'F 4 3 2', 211: 'I 4 3 2',
    212: 'P 43 3 2', 213: 'P 41 3 2', 214: 'I 41 3 2',
    215: 'P -4 3 m', 216: 'F -4 3 m', 217: 'I -4 3 m',
    218: 'P -4 3 n', 219: 'F -4 3 c', 220: 'I -4 3 d',
    221: 'P m -3 m', 222: 'P n -3 n', 223: 'P m -3 n',
    224: 'P n -3 m', 225: 'F m -3 m', 226: 'F m -3 c',
    227: 'F d -3 m', 228: 'F d -3 c', 229: 'I m -3 m',
    230: 'I a -3 d',
}


def _get_hm_symbol(sg_num):
    """Get H-M symbol for a space group number.

    Uses the static table first, then tries pymatgen as a fallback.
    """
    if sg_num in _SG_HM:
        return _SG_HM[sg_num]
    try:
        from pymatgen.symmetry.groups import SpaceGroup
        sg = SpaceGroup.from_int_number(sg_num)
        return sg.symbol
    except Exception:
        pass
    warnings.warn(f"Space group {sg_num} not in lookup table — using 'P 1'. "
                  f"Add it to _SG_HM in gsasii_backend.py for correct results.")
    return 'P 1'


# ── Default instrument / refinement parameters ────────────────────────────
# These are used when the user does not provide their own .instprm file.
# U, V, W, X, Y are initial guesses — GSAS-II refines them during fitting.
# Polariz. and SH/L are NOT refined and directly affect calculated
# intensities; users with known instrument profiles should override them.
DEFAULT_POLARIZ = 0.99    # Monochromator polarization (0.99 for graphite)
DEFAULT_SH_L = 0.002      # Finger-Cox-Jephcoat asymmetry parameter
DEFAULT_U = 2.0            # Caglioti U initial guess (centideg²) — refined
DEFAULT_V = -2.0           # Caglioti V initial guess (centideg²) — refined
DEFAULT_W = 5.0            # Caglioti W initial guess (centideg²) — refined
DEFAULT_X = 0.0            # Lorentzian X initial guess (centideg) — refined
DEFAULT_Y = 0.0            # Lorentzian Y initial guess (centideg) — refined
DEFAULT_B_ISO = 0.5        # Fallback B_iso (Å²) when GSAS-II extraction fails


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

    Validates the result by re-expanding and comparing site counts.
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
            full_cell_n = len(structs[0])  # total sites in full unit cell

            # Try tight tolerance first (0.01 Å), fall back to looser (0.1 Å).
            # Tight tolerance avoids merging non-equivalent sites in compact
            # cells like W2C Pbcn where heavy atoms are close together.
            sites = None
            for symprec in (0.01, 0.05, 0.1):
                try:
                    sga = SpacegroupAnalyzer(structs[0], symprec=symprec)
                    sym_struct = sga.get_symmetrized_structure()
                    candidate = []
                    for equiv_sites in sym_struct.equivalent_sites:
                        site = equiv_sites[0]
                        frac = site.frac_coords % 1.0
                        el = str(site.specie)
                        occ = 1.0
                        if hasattr(site, 'species'):
                            sp = site.species
                            if hasattr(sp, 'num_atoms'):
                                occ = float(sp.num_atoms)
                        candidate.append((el, float(frac[0]), float(frac[1]),
                                          float(frac[2]), occ))

                    # Validate: the number of equivalent sites summed across
                    # all groups should equal the full unit cell site count.
                    # This catches bad equivalence groupings.
                    expanded_n = sum(
                        len(eq) for eq in sym_struct.equivalent_sites)
                    if expanded_n == full_cell_n and candidate:
                        sites = candidate
                        break
                    # If counts don't match, try a looser tolerance
                except Exception:
                    continue

            if sites:
                return sites
    except Exception as e:
        print(f"  Warning: pymatgen asymmetric-unit reduction failed: {e}",
              flush=True)
        print("  Falling back to raw CIF sites — if the CIF contains the "
              "full unit cell, GSAS-II may over-expand.", flush=True)

    # Fallback: raw parse_cif (usually fine for COD CIFs which list
    # asymmetric unit; may cause over-expansion for MP/pymatgen CIFs
    # which list the full unit cell).
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

    # H-M symbol — try phase dict first, then dynamic lookup
    hm = ph.get('spacegroup', '') or ph.get('spacegroup_name', '')
    if not hm:
        hm = _get_hm_symbol(sg)

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
                                X_deg, Y_deg, gaussian_only=False):
    """Compute a raw (unscaled) profile shape for one phase.

    Returns an array of the same length as *tt_arr* whose values are
    proportional to the diffracted intensity at each 2θ point.  The
    absolute scale is arbitrary — what matters is the *relative* shape
    compared to other phases so that the proportional decomposition
    can correctly partition the total pattern.

    Parameters
    ----------
    gaussian_only : bool
        If True, use pure Gaussian profiles (eta=0) with a tighter
        window (3×FWHM).  This eliminates the long Lorentzian tails
        that cause cross-phase leakage artifacts in the proportional
        decomposition, while preserving accurate peak localisation
        for genuinely overlapping reflections.

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

        if gaussian_only:
            eta = 0.0

        sigma_g = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        gamma_l = fwhm / 2.0
        # Gaussian-only: 3×FWHM is sufficient (G ≈ 10⁻¹¹ at edge).
        # Pseudo-Voigt: 6×FWHM captures Lorentzian tails.
        window = 3.0 * fwhm if gaussian_only else 6.0 * fwhm

        msk = np.abs(tt_arr - tt_pos) < window
        if not msk.any():
            continue

        dx = tt_arr[msk] - tt_pos
        G = np.exp(-0.5 * (dx / sigma_g) ** 2)
        if eta > 0:
            L = 1.0 / (1.0 + (dx / gamma_l) ** 2)
            prof = eta * L + (1.0 - eta) * G
        else:
            prof = G
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


def _estimate_profile_params(tt, y_obs):
    """Estimate initial Caglioti U, V, W from observed peak widths.

    Finds the strongest peaks in the data, measures their approximate FWHM,
    and fits the Caglioti equation FWHM² = U tan²θ + V tanθ + W to get
    reasonable starting values.  Returns (U, V, W) in centidegrees² / centideg²
    (GSAS-II internal units), or the module defaults if estimation fails.

    This is critical for complex phases like W2C where the default U/V/W
    may be far from the true broadening, causing GSAS-II to converge to
    a local minimum.
    """
    try:
        from scipy.signal import find_peaks

        # Subtract a simple baseline (percentile)
        baseline = np.percentile(y_obs, 10)
        y_corr = y_obs - baseline

        # Find prominent peaks (height > 10% of max, well separated)
        height_thresh = 0.1 * y_corr.max()
        step = float(tt[1] - tt[0]) if len(tt) > 1 else 0.02
        distance = max(1, int(0.5 / step))  # at least 0.5° apart
        peaks, props = find_peaks(y_corr, height=height_thresh, distance=distance)

        if len(peaks) < 2:
            return DEFAULT_U, DEFAULT_V, DEFAULT_W

        # Measure FWHM for each peak
        fwhm_data = []  # (tan_theta, fwhm_deg)
        for pk in peaks:
            half_max = y_corr[pk] / 2.0
            # Walk left
            left = pk
            while left > 0 and y_corr[left] > half_max:
                left -= 1
            # Walk right
            right = pk
            while right < len(y_corr) - 1 and y_corr[right] > half_max:
                right += 1
            fwhm_deg = float(tt[right] - tt[left])
            if 0.02 < fwhm_deg < 3.0:  # reasonable range
                theta_rad = math.radians(float(tt[pk]) / 2.0)
                tan_th = math.tan(theta_rad)
                fwhm_data.append((tan_th, fwhm_deg))

        if len(fwhm_data) < 2:
            return DEFAULT_U, DEFAULT_V, DEFAULT_W

        # Fit Caglioti: FWHM² = U tan²θ + V tanθ + W
        # In GSAS-II, U/V/W are in centideg² so FWHM is in centideg.
        # We work in degrees then convert.
        tan_arr = np.array([d[0] for d in fwhm_data])
        fwhm_sq = np.array([d[1] ** 2 for d in fwhm_data])

        # Build design matrix [tan²θ, tanθ, 1]
        A = np.column_stack([tan_arr ** 2, tan_arr, np.ones_like(tan_arr)])
        try:
            result, _, _, _ = np.linalg.lstsq(A, fwhm_sq, rcond=None)
            U_deg2, V_deg2, W_deg2 = float(result[0]), float(result[1]), float(result[2])
        except np.linalg.LinAlgError:
            return DEFAULT_U, DEFAULT_V, DEFAULT_W

        # Convert degrees² → centideg²  (multiply by 10000)
        U_cdeg2 = U_deg2 * 10000.0
        V_cdeg2 = V_deg2 * 10000.0
        W_cdeg2 = W_deg2 * 10000.0

        # Sanity clamp: U and W should be positive, V typically negative
        U_cdeg2 = max(0.1, min(U_cdeg2, 500.0))
        V_cdeg2 = max(-200.0, min(V_cdeg2, 200.0))
        W_cdeg2 = max(0.1, min(W_cdeg2, 500.0))

        return U_cdeg2, V_cdeg2, W_cdeg2

    except Exception:
        return DEFAULT_U, DEFAULT_V, DEFAULT_W


def _estimate_lorentzian_params(tt, y_obs, U_cdeg2, V_cdeg2, W_cdeg2):
    """Estimate initial Lorentzian profile parameters X and Y from peak shapes.

    Measures the Lorentzian fraction of observed peaks by comparing their
    half-max width to their quarter-max width (Gaussian ratio ~1.48,
    Lorentzian ratio ~1.73).  Decomposes each peak's FWHM into Gaussian and
    Lorentzian components using the Thompson-Cox-Hastings (TCH) relationship,
    then fits H_L = X*tan(theta) + Y/cos(theta) across peaks.

    Returns (X, Y) in centidegrees, or moderate defaults if estimation fails.
    Critical for carbides/oxides with significant crystallite-size (Y) and
    micro-strain (X) Lorentzian broadening.
    """
    try:
        from scipy.signal import find_peaks

        baseline = np.percentile(y_obs, 10)
        y_corr = y_obs - baseline

        height_thresh = 0.1 * y_corr.max()
        step = float(tt[1] - tt[0]) if len(tt) > 1 else 0.02
        distance = max(1, int(0.5 / step))
        peaks, _ = find_peaks(y_corr, height=height_thresh, distance=distance)

        if len(peaks) < 2:
            return DEFAULT_X, DEFAULT_Y

        lor_data = []  # (tan_theta, cos_theta, H_L_deg)
        for pk in peaks:
            half_max = y_corr[pk] / 2.0
            quarter_max = y_corr[pk] / 4.0

            # Measure half-max width
            left_h, right_h = pk, pk
            while left_h > 0 and y_corr[left_h] > half_max:
                left_h -= 1
            while right_h < len(y_corr) - 1 and y_corr[right_h] > half_max:
                right_h += 1
            fwhm_deg = float(tt[right_h] - tt[left_h])

            # Measure quarter-max width
            left_q, right_q = pk, pk
            while left_q > 0 and y_corr[left_q] > quarter_max:
                left_q -= 1
            while right_q < len(y_corr) - 1 and y_corr[right_q] > quarter_max:
                right_q += 1
            fwqm_deg = float(tt[right_q] - tt[left_q])

            if fwhm_deg < 0.02 or fwhm_deg > 3.0 or fwqm_deg <= fwhm_deg:
                continue

            # Estimate Lorentzian fraction eta from FWQM/FWHM ratio.
            # Gaussian: FWQM/FWHM = sqrt(2) ≈ 1.414
            # Lorentzian: FWQM/FWHM = sqrt(3) ≈ 1.732
            # Linear interpolation gives eta (Lorentzian fraction).
            ratio = fwqm_deg / fwhm_deg
            # Linear interpolation for eta
            eta = (ratio - 1.4142) / (1.7321 - 1.4142)
            eta = max(0.0, min(eta, 1.0))

            if eta < 0.01:
                continue  # essentially pure Gaussian, no Lorentzian info

            # TCH decomposition: total FWHM ≈ eta*H_L + (1-eta)*H_G (simplified)
            # More precise: use the 5th-order TCH formula inverse.
            # For a practical estimate, the pseudo-Voigt approximation gives:
            #   H_L ≈ FWHM * eta (leading-order)
            #   H_G ≈ FWHM * (1 - eta)
            # Subtract the Gaussian Caglioti contribution to isolate Lorentzian.
            theta_rad = math.radians(float(tt[pk]) / 2.0)
            tan_th = math.tan(theta_rad)
            cos_th = math.cos(theta_rad)

            # Caglioti Gaussian FWHM² in deg² (convert from centideg²)
            H_G_sq_deg2 = (U_cdeg2 * tan_th**2 + V_cdeg2 * tan_th
                           + W_cdeg2) / 10000.0
            H_G_deg = math.sqrt(max(H_G_sq_deg2, 0.001))

            # Lorentzian FWHM: from total FWHM and Gaussian contribution
            # Using TCH relation: H_total^5 ≈ H_G^5 + ... + H_L^5
            # Simplified: H_L ≈ max(0, FWHM - H_G) when eta > 0
            # More robust: H_L = FWHM * eta (pseudo-Voigt definition)
            H_L_deg = fwhm_deg * eta
            if H_L_deg > 0.005:
                lor_data.append((tan_th, cos_th, H_L_deg))

        if len(lor_data) < 2:
            return DEFAULT_X, DEFAULT_Y

        # Fit H_L = X*tan(theta) + Y/cos(theta)
        # X and Y are in degrees here; convert to centideg at the end.
        tan_arr = np.array([d[0] for d in lor_data])
        inv_cos_arr = np.array([1.0 / d[1] for d in lor_data])
        hl_arr = np.array([d[2] for d in lor_data])

        A = np.column_stack([tan_arr, inv_cos_arr])
        try:
            result, _, _, _ = np.linalg.lstsq(A, hl_arr, rcond=None)
            X_deg, Y_deg = float(result[0]), float(result[1])
        except np.linalg.LinAlgError:
            return DEFAULT_X, DEFAULT_Y

        # Convert degrees → centidegrees (multiply by 100)
        X_cdeg = X_deg * 100.0
        Y_cdeg = Y_deg * 100.0

        # Sanity clamp: both should be non-negative, moderate magnitude
        X_cdeg = max(0.0, min(X_cdeg, 50.0))
        Y_cdeg = max(0.0, min(Y_cdeg, 50.0))

        return X_cdeg, Y_cdeg

    except Exception:
        return DEFAULT_X, DEFAULT_Y


def _write_instprm(work_dir, wavelength, polariz=None, sh_l=None,
                   u=None, v=None, w=None, x=None, y=None):
    """Write a minimal GSAS-II .instprm file. Returns path.

    Uses module-level DEFAULT_* constants unless overridden.
    U, V, W, X, Y are initial guesses that GSAS-II will refine.
    Polariz. and SH/L are NOT refined — they should match the instrument.
    """
    path = os.path.join(work_dir, 'instrument.instprm')
    pol = polariz if polariz is not None else DEFAULT_POLARIZ
    shl = sh_l if sh_l is not None else DEFAULT_SH_L
    _u = u if u is not None else DEFAULT_U
    _v = v if v is not None else DEFAULT_V
    _w = w if w is not None else DEFAULT_W
    _x = x if x is not None else DEFAULT_X
    _y = y if y is not None else DEFAULT_Y
    lines = [
        '#GSAS-II instrument parameter file; do not add/delete items!',
        'Type:PXC',
        f'Lam:{wavelength:.6f}',
        'Zero:0.0',
        f'Polariz.:{pol}',
        f'U:{_u}',
        f'V:{_v}',
        f'W:{_w}',
        f'X:{_x}',
        f'Y:{_y}',
        'Z:0.0',
        f'SH/L:{shl}',
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
              max_cycles=10, progress_callback=None,
              instprm_file=None, polariz=None, sh_l=None):
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
    instprm_file : str, optional — path to user-provided .instprm file;
        if given, this is used INSTEAD of the auto-generated defaults
    polariz : float, optional — monochromator polarization (default 0.99)
    sh_l : float, optional — Finger-Cox-Jephcoat asymmetry (default 0.002)

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
    # Use user-provided .instprm if available; otherwise generate defaults
    if instprm_file and os.path.isfile(instprm_file):
        import shutil
        instprm_path = os.path.join(work_dir, 'instrument.instprm')
        shutil.copy2(instprm_file, instprm_path)
        print(f"Using user-provided instrument parameters: {instprm_file}",
              flush=True)
    else:
        # Estimate initial profile parameters from observed peak widths
        est_u, est_v, est_w = _estimate_profile_params(tt_r, y_r)
        est_x, est_y = _estimate_lorentzian_params(tt_r, y_r,
                                                    est_u, est_v, est_w)
        instprm_path = _write_instprm(work_dir, wavelength,
                                       polariz=polariz, sh_l=sh_l,
                                       u=est_u, v=est_v, w=est_w,
                                       x=est_x, y=est_y)
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

        # Fix histogram scale to 1.0 — NEVER refine it.
        # GSAS-II has N+1 scale parameters (N phase scales + 1 histogram
        # scale).  Only N are independent.  Refining all N+1 creates a
        # perfect correlation (100%) that causes SVD singularity and the
        # refinement gets stuck with zero peak intensity.
        # Standard practice: fix histogram scale, refine only phase scales.
        try:
            histogram.data['Sample Parameters']['Scale'] = [1.0, False]
        except (KeyError, TypeError):
            pass

        # Set background
        bkg_data = histogram.data['Background']
        bg_init = float(np.percentile(y_r, 2))
        bkg_data[0] = ['chebyschev-1', True, n_bg_coeffs,
                        bg_init] + [0.0] * (n_bg_coeffs - 1)

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
        _failed_stages = []

        # ── Helper to safely run a refinement stage ──────────────────────
        def _safe_refine(stage_name, refinement_dicts, stage_num):
            nonlocal last_good_stage
            try:
                gpx.do_refinements(refinement_dicts)
                last_good_stage = stage_num
                # Guard: ensure histogram scale stays fixed at 1.0.
                # GSAS-II's do_refinements can re-enable it internally
                # when 'Scale' appears in any refinement dict.
                try:
                    histogram.data['Sample Parameters']['Scale'] = [1.0, False]
                except (KeyError, TypeError):
                    pass
                return True
            except Exception as e:
                # GSAS-II internally restores from .bak0.gpx on failure,
                # so the project state reverts to pre-refinement. Safe to continue.
                _failed_stages.append(stage_name)
                warnings.warn(f"GSAS-II: {stage_name} failed ({e}). "
                             f"Continuing with results from stage {last_good_stage}.")
                return False

        # Determine structural complexity → boost cycles for complex phases
        max_asym_atoms = max(
            (len(_reduce_to_asymmetric_unit(ph.get('cif_text', '')))
             for ph in phases if ph.get('cif_text')),
            default=1)
        _complex = max_asym_atoms > 6  # W2C has ~6 asym sites
        _cyc_mult = 2 if _complex else 1  # double cycles for complex phases

        if progress_callback:
            progress_callback('GSAS-II: stage 1 — refining background + scale...')

        # ── Stage 1: Background + scale (one phase at a time) ────────────
        # Fix all scales first, then refine them one at a time to break
        # the correlation that causes SVD singularities.
        # Estimate a data-driven initial scale for each phase.
        # With the histogram scale fixed at 1.0, the phase scale must absorb
        # the full intensity.  A rough estimate based on peak height and
        # structural complexity helps GSAS-II converge from the right region.
        peak_height = float(np.max(y_r) - np.percentile(y_r, 5))
        n_phases = len(gsas_phases)
        for phase_obj, ph_input in zip(gsas_phases, phases):
            hapData = list(phase_obj.data['Histograms'].values())[0]
            cif_text = ph_input.get('cif_text', '')
            n_asym = len(_reduce_to_asymmetric_unit(cif_text)) if cif_text else 1
            # Scale ∝ peak_height / (n_atoms² × n_phases).  The n_atoms²
            # factor approximates how F² grows with atom count.
            init_scale = peak_height / max(1.0, (n_asym ** 2) * n_phases * 10.0)
            init_scale = max(init_scale, 0.001)  # floor to avoid zero
            hapData['Scale'] = [init_scale, False]  # start fixed

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

        # Re-refine all phase scales together (they have good starting values).
        # Do NOT use {'Scale': True} here — that enables the histogram scale
        # refinement flag, which is degenerate with phase scales and causes
        # 100% correlation / SVD singularity.
        _safe_refine('all phase scales', [{
            'set': {},
            'cycles': min(max_cycles, 5),
        }], 1)

        if progress_callback:
            progress_callback('GSAS-II: stage 2 — refining profile parameters...')

        # ── Stage 2: Profile parameters ──────────────────────────────────
        _safe_refine('profile', [{
            'set': {
                'Instrument Parameters': ['U', 'V', 'W', 'X', 'Y'],
            },
            'cycles': min(max_cycles, 5 * _cyc_mult),
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
                'cycles': min(max_cycles, 8 * _cyc_mult),
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
            'cycles': min(max_cycles, 5 * _cyc_mult),
        }], 4)

        if progress_callback:
            progress_callback('GSAS-II: stage 5 — refining atom positions (XYZ)...')

        # ── Stage 5: Atom positions (XYZ) ─────────────────────────────
        # For complex structures (carbides, oxides) the COD database atom
        # coordinates may not exactly match this sample.  Wrong positions →
        # wrong structure factors F(hkl) → the Rietveld model cannot match
        # observed peak intensity ratios → convergence failure.
        # GSAS-II automatically constrains atoms on special positions, so
        # this is safe for simple metals (no-op) while critical for W2C etc.
        #
        # Save atom positions before refinement for damping check.
        saved_xyz = {}
        for idx, phase_obj in enumerate(gsas_phases):
            phase_atoms = phase_obj.data.get('Atoms', [])
            saved_xyz[idx] = [(a[3], a[4], a[5]) if len(a) > 5 else None
                              for a in phase_atoms]
            try:
                phase_obj.set_refinements({'Atoms': {'all': 'XU'}})
            except Exception:
                pass
        xyz_ok = _safe_refine('atom XYZ', [{
            'set': {},
            'cycles': min(max_cycles, 3 * _cyc_mult),
        }], 5)

        # Damping check: if any atom jumped > 0.5 fractional units,
        # it likely hopped to a symmetry-equivalent site — revert it.
        if xyz_ok:
            for idx, phase_obj in enumerate(gsas_phases):
                phase_atoms = phase_obj.data.get('Atoms', [])
                for j, atom in enumerate(phase_atoms):
                    if (len(atom) > 5 and idx in saved_xyz
                            and j < len(saved_xyz[idx])
                            and saved_xyz[idx][j] is not None):
                        ox, oy, oz = saved_xyz[idx][j]
                        dx = abs(atom[3] - ox)
                        dy = abs(atom[4] - oy)
                        dz = abs(atom[5] - oz)
                        if max(dx, dy, dz) > 0.5:
                            warnings.warn(
                                f"GSAS-II: atom {j} in phase {idx} jumped "
                                f"by ({dx:.3f},{dy:.3f},{dz:.3f}) — "
                                f"reverting to original position.")
                            atom[3], atom[4], atom[5] = ox, oy, oz

        # Turn off XYZ refinement flags, keep only Uiso for remaining stages
        for phase_obj in gsas_phases:
            try:
                phase_obj.set_refinements({'Atoms': {'all': 'U'}})
            except Exception:
                pass

        if progress_callback:
            progress_callback('GSAS-II: stage 6 — final background + scale polish...')

        # ── Stage 6: Final polish — re-refine background + scale ────────
        # Background was first refined in Stage 1 when profile/cell/Uiso
        # were still at initial values. Now that all structural parameters
        # have converged, re-optimize background to remove systematic
        # misfit (e.g. Chebyshev polynomial dipping on right side).
        _safe_refine('final polish', [{
            'set': {
                'Background': {'type': 'chebyschev-1', 'refine': True,
                                'no. coeffs': n_bg_coeffs},
            },
            'cycles': min(max_cycles, 5 * _cyc_mult),
        }], 6)

        # Warn if multiple refinement stages failed — result may be unreliable
        if len(_failed_stages) >= 3:
            warnings.warn(
                f"GSAS-II: {len(_failed_stages)} stages failed "
                f"({', '.join(_failed_stages)}). The refinement result may be "
                f"unreliable — check CIF quality or try simpler fitting range.")

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
        _y_bg_gsas = y_bg_out.copy()   # preserve for unbiased phase isolation

        # ── Background dip correction ──────────────────────────────────
        # The Chebyshev polynomial can create local dips where the
        # optimizer trades intensity with peak tails.  Fix: smooth
        # and raise dips.  We detrend first (remove a low-order poly)
        # so the Gaussian doesn't blur steep slopes at low 2θ into
        # the 30-50° region.
        if len(y_bg_out) >= 3:
            try:
                _trend_coeffs = np.polyfit(tt_out, y_bg_out, 2)
                _trend = np.polyval(_trend_coeffs, tt_out)
                _resid = y_bg_out - _trend

                _step = float(tt_out[1] - tt_out[0]) if len(tt_out) > 1 else 0.02
                _sig  = max(3, int(10.0 / _step))          # 10° Gaussian sigma
                _k    = min(3 * _sig, len(_resid) // 2)
                if _k >= 1:
                    _kx   = np.arange(-_k, _k + 1, dtype=float)
                    _kern = np.exp(-0.5 * (_kx / _sig) ** 2)
                    _kern /= _kern.sum()
                    _padded = np.pad(_resid, _k, mode='edge')
                    _smooth_resid = np.convolve(_padded, _kern, mode='valid')
                    y_bg_out = _trend + np.maximum(_resid, _smooth_resid)
            except (np.linalg.LinAlgError, ValueError):
                pass  # skip dip correction if polyfit fails

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
        all_phase_refs = []   # per-phase reflection lists from generate_reflections

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
                    # First pass: collect all reflections and find max weight
                    # for relative threshold filtering (matches preview behavior)
                    raw_refs = []
                    max_weight = 0.0
                    for row in ref_arr:
                        h, k, l = int(row[0]), int(row[1]), int(row[2])
                        mult      = float(row[3])
                        d_sp      = float(row[4])
                        two_theta = float(row[5])
                        fc2       = float(row[8])   # Fc²
                        weight    = mult * fc2
                        if weight > 0 and tt_min <= two_theta <= tt_max:
                            raw_refs.append((two_theta, d_sp, (h, k, l), weight))
                            if weight > max_weight:
                                max_weight = weight
                    # Second pass: apply relative F² threshold to remove
                    # ghost reflections (Fc² near zero compared to strongest)
                    rel_thresh = max_weight * 1e-3  # 0.1% of strongest
                    refs = [r for r in raw_refs if r[3] >= max(1e-4, rel_thresh)]
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
            # 1 deg = 100 centideg  →  1 deg² = 10 000 centideg²
            U_deg = inst['U'] / 10000.0   # centideg² → deg²
            V_deg = inst['V'] / 10000.0
            W_deg = inst['W'] / 10000.0
            X_deg = inst['X'] / 100.0     # centideg → deg
            Y_deg = inst['Y'] / 100.0

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

            # ── Generate tick positions / reflection list ─────────────────
            # Prefer GSAS-II's refined Fc² values (gsas_refs) when available
            # — they correctly account for all atoms and symmetry.  Ghost
            # reflections with Fc²≈0 are removed by a relative threshold
            # (0.1% of strongest) matching the preview stick filter.
            # Fall back to generate_reflections when GSAS-II data is missing.
            gsas_phase_refs = gsas_refs.get(phase_obj.name)
            if gsas_phase_refs:
                phase_refs = gsas_phase_refs
                tick_positions = [round(r[0], 3) for r in phase_refs]
            else:
                sys_ = (ph.get('system') or 'triclinic').lower()
                sg = ph.get('spacegroup_number', 1)
                sites = _get_expanded_sites(ph.get('cif_text', ''), sg)
                phase_refs = generate_reflections(
                    a, b, c, alpha, beta, gamma, sys_, sg,
                    wavelength, tt_min, tt_max, hkl_max=12,
                    sites=sites)
                tick_positions = [round(r[0], 3) for r in phase_refs]
            all_phase_refs.append(phase_refs)

            # B_iso (average over atoms; falls back to DEFAULT_B_ISO)
            b_iso_avg = DEFAULT_B_ISO
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

        # ── Per-phase patterns ─────────────────────────────────────────
        # Primary method: GSAS-II phase isolation — zero all phases
        # except one, let GSAS-II recompute (0 cycles), extract
        # ycalc - background.  This uses GSAS-II's own profile
        # functions (correct peak shapes, asymmetry, everything).
        #
        # Fallback: reconstruct per-phase profiles from reflection
        # lists and refined instrument parameters (may not exactly
        # match GSAS-II's profile shapes).
        #
        # Last resort: equal split (unreliable weight fractions).
        if len(gsas_phases) > 1:
            total_above_bg = np.maximum(y_calc_out - y_bg_out, 0.0)
            decomp_ok = False

            # ── Primary: GSAS-II phase isolation ─────────────────────
            print("  Phase isolation: computing GSAS-II per-phase "
                  "patterns...", flush=True)

            # Save ALL refinement flags — the main refinement left many
            # params refinable (background, U/V/W/X/Y, cell, Uiso).
            # We must turn them ALL off before isolation, otherwise
            # do_refinements will try to refine them with one phase
            # zeroed, causing that phase's parameters to diverge.
            orig_hap_scales = []
            for phase_obj in gsas_phases:
                hapData = list(phase_obj.data['Histograms'].values())[0]
                orig_hap_scales.append(list(hapData['Scale']))

            # Save & turn off background refinement flag
            saved_bg_flag = histogram.data['Background'][0][1]
            histogram.data['Background'][0][1] = False

            # Save & turn off instrument parameter refinement flags.
            # GSAS-II stores refine flags in TWO possible locations:
            # (a) inline: inst_params[0][key][2] (3-element lists)
            # (b) separate dict: inst_params[1][key] (boolean)
            inst_params_all = histogram.data['Instrument Parameters']
            inst_params_raw = inst_params_all[0]
            inst_refine_dict = (inst_params_all[1]
                                if len(inst_params_all) > 1
                                and isinstance(inst_params_all[1], dict)
                                else {})
            saved_inst_flags = {}
            for key in ['U', 'V', 'W', 'X', 'Y', 'SH/L', 'Zero']:
                # Location (a): inline in parameter list
                if key in inst_params_raw and len(inst_params_raw[key]) >= 3:
                    saved_inst_flags[(key, 'inline')] = \
                        inst_params_raw[key][2]
                    inst_params_raw[key][2] = False
                # Location (b): separate refine-flags dict
                if key in inst_refine_dict:
                    saved_inst_flags[(key, 'dict')] = \
                        inst_refine_dict[key]
                    inst_refine_dict[key] = False

            # Save & turn off cell and atom refinement flags.
            # atom[2] = refinement flags string ('XU', 'U', '' etc.)
            # atom[9] = multiplicity — NOT the refinement flag.
            saved_cell_flags = []
            saved_atom_flags = []
            for phase_obj in gsas_phases:
                cell = phase_obj.data['General']['Cell']
                saved_cell_flags.append(cell[0])
                cell[0] = False
                atom_flags = []
                for atom in phase_obj.data['Atoms']:
                    if len(atom) > 2:
                        atom_flags.append(str(atom[2]))
                        atom[2] = ''   # clear XYZ + Uiso flags
                saved_atom_flags.append(atom_flags)

            # Belt-and-suspenders: force 0 cycles in Controls so GSAS-II
            # cannot iterate even if a flag was somehow missed.
            _controls = gpx.data.get('Controls', {}).get('data', {})
            _saved_maxcyc = _controls.get('max cyc', 3)
            _controls['max cyc'] = 0

            try:
                phase_patterns = []
                for i in range(len(gsas_phases)):
                    # Activate only phase i; zero all others
                    for j, phase_obj in enumerate(gsas_phases):
                        hapData = list(phase_obj.data['Histograms'].values())[0]
                        if j == i:
                            hapData['Scale'] = [orig_hap_scales[j][0], False]
                        else:
                            hapData['Scale'] = [0.0, False]

                    # Persist to disk so do_refinements picks up changes.
                    gpx.save()

                    # Evaluate only (cycles=0) — all refinement flags are
                    # OFF, so nothing can diverge.
                    gpx.do_refinements([{'set': {}, 'cycles': 0}])

                    # Re-fetch histogram — do_refinements reloads the
                    # project from disk, so the old reference may be stale.
                    histogram = gpx.histograms()[0]

                    y_calc_iso = np.array(histogram.getdata('ycalc'))
                    # Use GSAS-II's original background (before our dip
                    # correction) so the correction delta doesn't bias
                    # phase ratios toward the dominant phase.
                    phase_pat = np.maximum(
                        y_calc_iso[rmask] - _y_bg_gsas, 0.0)
                    phase_patterns.append(phase_pat.tolist())

                    ph_name = (phase_results[i]['name']
                               if i < len(phase_results) else f'Phase {i}')
                    print(f"  Phase {i} ({ph_name}): "
                          f"max={np.max(phase_pat):.1f}, "
                          f"integrated={np.sum(phase_pat):.1f}", flush=True)

                # Diagnostic: verify isolation produced correct patterns.
                sum_iso = np.zeros_like(tt_out, dtype=np.float64)
                for pp in phase_patterns:
                    sum_iso += np.array(pp)
                max_diff = float(np.max(np.abs(
                    sum_iso - total_above_bg)))
                sum_total = float(np.sum(total_above_bg))
                sum_iso_total = float(np.sum(sum_iso))
                print(f"  Diagnostic: sum(iso)={sum_iso_total:.1f}, "
                      f"sum(above_bg)={sum_total:.1f}, "
                      f"max_pointwise_diff={max_diff:.2f}", flush=True)

                # Proportional normalization: ensure stacked fills sum
                # exactly to total_above_bg at every point so that the
                # fills match the I_calc line perfectly.
                for i_pp, pp in enumerate(phase_patterns):
                    pp_arr = np.array(pp)
                    ratio = np.zeros_like(sum_iso)
                    np.divide(pp_arr, sum_iso, out=ratio,
                              where=sum_iso > 0)
                    phase_patterns[i_pp] = np.maximum(
                        ratio * total_above_bg, 0.0).tolist()

                decomp_ok = True
                print("  Phase isolation succeeded.", flush=True)
            except Exception as e_iso:
                print(f"  Phase isolation failed: {e_iso}", flush=True)
                phase_patterns = []
            finally:
                # Always restore ALL original flags and scales
                histogram.data['Background'][0][1] = saved_bg_flag
                for (key, loc), flag in saved_inst_flags.items():
                    if loc == 'inline':
                        inst_params_raw[key][2] = flag
                    else:
                        inst_refine_dict[key] = flag
                for idx_r, phase_obj in enumerate(gsas_phases):
                    phase_obj.data['General']['Cell'][0] = \
                        saved_cell_flags[idx_r]
                    for j_a, atom in enumerate(phase_obj.data['Atoms']):
                        if (len(atom) > 2
                                and j_a < len(saved_atom_flags[idx_r])):
                            atom[2] = saved_atom_flags[idx_r][j_a]
                    hapData = list(phase_obj.data['Histograms'].values())[0]
                    hapData['Scale'] = orig_hap_scales[idx_r]
                # Restore max cycles to original value
                _controls['max cyc'] = _saved_maxcyc
                gpx.save()
                histogram = gpx.histograms()[0]

            # ── Fallback: profile reconstruction from reflections ────
            if not decomp_ok:
                print("  Falling back to profile reconstruction...",
                      flush=True)
                if all_phase_refs and len(all_phase_refs) == len(gsas_phases):
                    try:
                        U_d = inst['U'] / 10000.0
                        V_d = inst['V'] / 10000.0
                        W_d = inst['W'] / 10000.0
                        X_d = inst['X'] / 100.0
                        Y_d = inst['Y'] / 100.0

                        raw_profiles = []
                        for i_ph, (phase_obj_r, fallback_refs) in enumerate(
                                zip(gsas_phases, all_phase_refs)):
                            refs_to_use = gsas_refs.get(
                                phase_obj_r.name, fallback_refs) or fallback_refs
                            if not refs_to_use:
                                raise ValueError(
                                    f"No reflections for phase {i_ph}")
                            raw_profiles.append(
                                _compute_raw_phase_profile(
                                    tt_out, refs_to_use,
                                    U_d, V_d, W_d, X_d, Y_d,
                                    gaussian_only=True))

                        scaled = []
                        for phase_obj, prof in zip(gsas_phases, raw_profiles):
                            s = raw_scales.get(phase_obj.name, 1.0)
                            scaled.append(prof * s)

                        sum_raw = np.zeros_like(tt_out, dtype=np.float64)
                        for sp in scaled:
                            sum_raw += sp
                        peak_max = (np.max(sum_raw)
                                    if np.max(sum_raw) > 0 else 1.0)
                        threshold = peak_max * 1e-10
                        for sp in scaled:
                            ratio = np.zeros_like(sum_raw)
                            np.divide(sp, sum_raw, out=ratio,
                                      where=sum_raw > threshold)
                            phase_patterns.append(
                                np.maximum(
                                    ratio * total_above_bg, 0.0).tolist())
                        decomp_ok = True
                        print("  Profile reconstruction succeeded.",
                              flush=True)
                    except Exception as e_fc2:
                        print(f"  Profile reconstruction failed: {e_fc2}",
                              flush=True)
                        phase_patterns = []

            # ── Last resort: equal split ───────────────────────────────
            if not decomp_ok:
                warnings.warn(
                    "GSAS-II phase decomposition: both isolation and "
                    "profile reconstruction failed. Falling back to "
                    "equal split — weight fractions will be UNRELIABLE.")
                phase_patterns = []
                for _ in gsas_phases:
                    share = (total_above_bg / len(gsas_phases)).tolist()
                    phase_patterns.append(share)

            # Update weight fractions from isolated patterns
            total_integ = sum(
                np.sum(np.array(pp)) for pp in phase_patterns) or 1e-30
            for i_wp, pp in enumerate(phase_patterns):
                integ_frac = np.sum(np.array(pp)) / total_integ * 100
                print(f"  Phase {i_wp}: "
                      f"integrated fraction = {integ_frac:.1f}%", flush=True)
                if i_wp < len(phase_results):
                    phase_results[i_wp]['weight_fraction_%'] = round(
                        integ_frac, 1)
        else:
            # Single phase — entire signal above background
            total_above_bg = np.maximum(y_calc_out - y_bg_out, 0.0)
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
