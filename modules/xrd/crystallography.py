"""
modules/xrd/crystallography.py
Pure-Python crystallography engine.
No pymatgen dependency — implements everything needed for Le Bail refinement.
"""

import math, re, itertools
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# UNIT CELL
# ─────────────────────────────────────────────────────────────────────────────

CRYSTAL_SYSTEMS = {
    'cubic':        lambda a,b,c,al,be,ga: (a, a, a, 90, 90, 90),
    'tetragonal':   lambda a,b,c,al,be,ga: (a, a, c, 90, 90, 90),
    'orthorhombic': lambda a,b,c,al,be,ga: (a, b, c, 90, 90, 90),
    'hexagonal':    lambda a,b,c,al,be,ga: (a, a, c, 90, 90,120),
    'trigonal':     lambda a,b,c,al,be,ga: (a, a, c, 90, 90,120),
    'monoclinic':   lambda a,b,c,al,be,ga: (a, b, c, 90, be, 90),
    'triclinic':    lambda a,b,c,al,be,ga: (a, b, c, al, be, ga),
}

def cell_volume(a, b, c, al, be, ga):
    """Unit cell volume in Å³."""
    al_r = math.radians(al)
    be_r = math.radians(be)
    ga_r = math.radians(ga)
    return a*b*c*math.sqrt(
        1 - math.cos(al_r)**2 - math.cos(be_r)**2 - math.cos(ga_r)**2
        + 2*math.cos(al_r)*math.cos(be_r)*math.cos(ga_r)
    )

def d_spacing(h, k, l, a, b, c, al, be, ga, system):
    """Calculate d-spacing for given hkl and cell parameters."""
    system = system.lower()
    try:
        if system in ('cubic',):
            return a / math.sqrt(h**2 + k**2 + l**2)
        elif system in ('tetragonal',):
            return 1 / math.sqrt((h**2+k**2)/a**2 + l**2/c**2)
        elif system in ('orthorhombic',):
            return 1 / math.sqrt(h**2/a**2 + k**2/b**2 + l**2/c**2)
        elif system in ('hexagonal', 'trigonal'):
            return 1 / math.sqrt((4/3)*(h**2+h*k+k**2)/a**2 + l**2/c**2)
        elif system in ('monoclinic',):
            be_r = math.radians(be)
            sin_be = math.sin(be_r)
            cos_be = math.cos(be_r)
            return 1 / math.sqrt(
                (1/sin_be**2) * (h**2/a**2 + k**2*sin_be**2/b**2 + l**2/c**2
                                 - 2*h*l*cos_be/(a*c))
            )
        elif system in ('triclinic',):
            al_r, be_r, ga_r = math.radians(al), math.radians(be), math.radians(ga)
            V = cell_volume(a, b, c, al, be, ga)
            S11 = b**2*c**2*math.sin(al_r)**2
            S22 = a**2*c**2*math.sin(be_r)**2
            S33 = a**2*b**2*math.sin(ga_r)**2
            S12 = a*b*c**2*(math.cos(al_r)*math.cos(be_r)-math.cos(ga_r))
            S23 = a**2*b*c*(math.cos(be_r)*math.cos(ga_r)-math.cos(al_r))
            S13 = a*b**2*c*(math.cos(ga_r)*math.cos(al_r)-math.cos(be_r))
            return V / math.sqrt(S11*h**2 + S22*k**2 + S33*l**2
                                  + 2*S12*h*k + 2*S23*k*l + 2*S13*h*l)
        else:
            # Fallback: orthorhombic
            return 1 / math.sqrt(h**2/a**2 + k**2/b**2 + l**2/c**2)
    except (ValueError, ZeroDivisionError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEMATIC ABSENCE RULES
# ─────────────────────────────────────────────────────────────────────────────

def is_allowed(h, k, l, spacegroup_number):
    """
    Simple systematic absence filter based on lattice type.
    Not complete for all space groups — handles the most common catalyst phases.
    Extendable.
    """
    sg = spacegroup_number

    # Face-centred (F): h,k,l all odd or all even
    if sg in range(196, 230):  # F lattice space groups
        if not ((h%2==k%2==l%2==0) or (h%2==1 and k%2==1 and l%2==1)):
            return False

    # Body-centred (I): h+k+l even
    if sg in list(range(197,200)) + list(range(204,207)) + list(range(211,215)) + \
             list(range(217,221)) + list(range(229,231)):
        if (h+k+l) % 2 != 0:
            return False

    # Hexagonal: for (00l), l must be even (P63)
    if sg in (173, 176, 186, 194):  # P63 family
        if h == 0 and k == 0 and l % 2 != 0:
            return False

    # Always remove (000)
    if h == 0 and k == 0 and l == 0:
        return False

    return True

def multiplicity(h, k, l, system):
    """Approximate multiplicity for common systems."""
    system = system.lower()
    zeros = sum(1 for x in [h,k,l] if x == 0)
    if system == 'cubic':
        # Very rough
        unique = len(set([abs(h),abs(k),abs(l)]))
        if unique == 1: return 6
        elif unique == 2: return 24 if zeros == 0 else 12
        else: return 48
    elif system in ('hexagonal', 'trigonal'):
        if h == 0 and k == 0: return 2
        elif h == 0 or k == 0 or h == k: return 12
        else: return 24
    elif system == 'tetragonal':
        if h == 0 and k == 0: return 2
        elif h == 0 or k == 0: return 8 if l != 0 else 4
        elif h == k: return 8
        else: return 16
    else:
        return 8  # fallback


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_reflections(a, b, c, al, be, ga, system, spacegroup_number,
                          wavelength, two_theta_min, two_theta_max, hkl_max=10):
    """
    Generate list of (two_theta, d, h, k, l, multiplicity) for all
    allowed reflections in the given 2θ range.
    """
    reflections = []
    seen_d = {}  # d -> list of (h,k,l) for deduplication

    for h in range(0, hkl_max+1):
        for k in range(-hkl_max, hkl_max+1):
            for l in range(-hkl_max, hkl_max+1):
                if h == 0 and k <= 0 and l <= 0:
                    continue  # avoid duplicates
                if h == 0 and k == 0 and l <= 0:
                    continue
                if not is_allowed(abs(h), abs(k), abs(l), spacegroup_number):
                    continue

                d = d_spacing(h, k, l, a, b, c, al, be, ga, system)
                if d is None or d <= 0:
                    continue

                sin_theta = wavelength / (2 * d)
                if sin_theta > 1 or sin_theta < 0:
                    continue

                two_theta = 2 * math.degrees(math.asin(sin_theta))
                if two_theta < two_theta_min or two_theta > two_theta_max:
                    continue

                d_key = round(d, 4)
                if d_key in seen_d:
                    seen_d[d_key][3] += 1  # increment multiplicity
                else:
                    m = multiplicity(abs(h), abs(k), abs(l), system)
                    seen_d[d_key] = [two_theta, d, (h,k,l), m]

    reflections = sorted(seen_d.values(), key=lambda x: x[0])
    return reflections  # [(two_theta, d, (h,k,l), multiplicity), ...]


# ─────────────────────────────────────────────────────────────────────────────
# PEAK SHAPE: PSEUDO-VOIGT + CAGLIOTI
# ─────────────────────────────────────────────────────────────────────────────

def caglioti_fwhm(two_theta_deg, U, V, W):
    """
    Caglioti equation: FWHM² = U·tan²θ + V·tanθ + W
    Returns FWHM in degrees.
    """
    theta = math.radians(two_theta_deg / 2)
    tan_t = math.tan(theta)
    fwhm2 = U * tan_t**2 + V * tan_t + W
    if fwhm2 <= 0:
        return 0.01
    return math.sqrt(fwhm2)

def pseudo_voigt(x, x0, fwhm, eta):
    """
    Pseudo-Voigt profile: η·Lorentzian + (1-η)·Gaussian
    eta = 0 → pure Gaussian, eta = 1 → pure Lorentzian
    """
    eta = max(0.0, min(1.0, eta))
    sigma_g = fwhm / (2 * math.sqrt(2 * math.log(2)))
    gamma_l = fwhm / 2

    dx = x - x0
    gauss = np.exp(-0.5 * (dx / sigma_g)**2)
    lor   = 1.0 / (1.0 + (dx / gamma_l)**2)
    return eta * lor + (1 - eta) * gauss

def compute_phase_pattern(two_theta_array, reflections, scale,
                           U, V, W, eta, two_theta_zero=0.0):
    """
    Compute the simulated intensity pattern for one phase.
    reflections: output of generate_reflections()
    Returns array of intensities same shape as two_theta_array.
    """
    pattern = np.zeros_like(two_theta_array)
    tt_shifted = two_theta_array - two_theta_zero

    for tt_peak, d, hkl, mult in reflections:
        fwhm = caglioti_fwhm(tt_peak, U, V, W)
        fwhm = max(fwhm, 0.005)
        # Only compute within ±20*fwhm of peak centre (efficiency)
        window = 20 * fwhm
        mask = np.abs(tt_shifted - tt_peak) < window
        if not mask.any():
            continue
        profile = pseudo_voigt(tt_shifted[mask], tt_peak, fwhm, eta)
        # Lorentz-polarisation factor
        theta_r = math.radians(tt_peak / 2)
        cos2t    = math.cos(math.radians(tt_peak))
        Lp = (1 + cos2t**2) / (math.sin(theta_r)**2 * math.cos(theta_r))
        pattern[mask] += scale * mult * Lp * profile

    return pattern


# ─────────────────────────────────────────────────────────────────────────────
# CHEBYSHEV POLYNOMIAL BACKGROUND
# ─────────────────────────────────────────────────────────────────────────────

def chebyshev_background(two_theta, coeffs, tt_min, tt_max):
    """
    Evaluate Chebyshev polynomial background.
    coeffs: list of polynomial coefficients [c0, c1, c2, ...]
    Normalises 2θ to [-1, 1] for numerical stability.
    """
    x = 2 * (two_theta - tt_min) / (tt_max - tt_min) - 1
    bg = np.zeros_like(two_theta)
    T_prev2 = np.ones_like(x)      # T0
    T_prev1 = x.copy()             # T1
    if len(coeffs) > 0:
        bg += coeffs[0] * T_prev2
    if len(coeffs) > 1:
        bg += coeffs[1] * T_prev1
    for i, c in enumerate(coeffs[2:], 2):
        T_curr = 2 * x * T_prev1 - T_prev2
        bg += c * T_curr
        T_prev2, T_prev1 = T_prev1, T_curr
    return bg


# ─────────────────────────────────────────────────────────────────────────────
# SCHERRER CRYSTALLITE SIZE
# ─────────────────────────────────────────────────────────────────────────────

def scherrer_size(fwhm_deg, two_theta_deg, wavelength, K=0.94):
    """
    Scherrer equation: D = K·λ / (β·cos θ)
    fwhm_deg: FWHM in degrees (instrumental broadening NOT subtracted here)
    Returns crystallite size D in Å.
    """
    beta_rad = math.radians(fwhm_deg)
    theta_rad = math.radians(two_theta_deg / 2)
    if beta_rad <= 0 or math.cos(theta_rad) <= 0:
        return None
    return K * wavelength / (beta_rad * math.cos(theta_rad))


# ─────────────────────────────────────────────────────────────────────────────
# CIF PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_cif(cif_text):
    """
    Minimal CIF parser — extracts unit cell, space group, and formula.
    Returns dict with keys: a, b, c, alpha, beta, gamma, spacegroup_number,
    spacegroup_name, system, formula, cod_id
    """
    result = {
        'a': None, 'b': None, 'c': None,
        'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0,
        'spacegroup_number': 1,
        'spacegroup_name': 'P 1',
        'system': 'triclinic',
        'formula': '',
        'cod_id': '',
    }

    def parse_val(s):
        """Strip uncertainty e.g. '3.002(5)' → 3.002"""
        s = s.strip().strip("'\"")
        m = re.match(r'^([0-9\.\-\+eE]+)', s)
        return float(m.group(1)) if m else None

    for line in cif_text.splitlines():
        line = line.strip()
        if line.startswith('_cell_length_a'):
            v = parse_val(line.split()[-1])
            if v: result['a'] = v
        elif line.startswith('_cell_length_b'):
            v = parse_val(line.split()[-1])
            if v: result['b'] = v
        elif line.startswith('_cell_length_c'):
            v = parse_val(line.split()[-1])
            if v: result['c'] = v
        elif line.startswith('_cell_angle_alpha'):
            v = parse_val(line.split()[-1])
            if v: result['alpha'] = v
        elif line.startswith('_cell_angle_beta'):
            v = parse_val(line.split()[-1])
            if v: result['beta'] = v
        elif line.startswith('_cell_angle_gamma'):
            v = parse_val(line.split()[-1])
            if v: result['gamma'] = v
        elif line.startswith('_symmetry_Int_Tables_number') or \
             line.startswith('_space_group_IT_number'):
            v = parse_val(line.split()[-1])
            if v: result['spacegroup_number'] = int(v)
        elif line.startswith('_symmetry_space_group_name_H-M') or \
             line.startswith('_space_group_name_H-M_alt'):
            parts = line.split(None, 1)
            if len(parts) > 1:
                result['spacegroup_name'] = parts[1].strip().strip("'\"")
        elif line.startswith('_chemical_formula_sum'):
            parts = line.split(None, 1)
            if len(parts) > 1:
                result['formula'] = parts[1].strip().strip("'\"")

    # Infer crystal system from space group number
    sg = result['spacegroup_number']
    if   1  <= sg <=   2: result['system'] = 'triclinic'
    elif 3  <= sg <=  15: result['system'] = 'monoclinic'
    elif 16 <= sg <=  74: result['system'] = 'orthorhombic'
    elif 75 <= sg <= 142: result['system'] = 'tetragonal'
    elif 143 <= sg <= 167: result['system'] = 'trigonal'
    elif 168 <= sg <= 194: result['system'] = 'hexagonal'
    elif 195 <= sg <= 230: result['system'] = 'cubic'

    return result


# ─────────────────────────────────────────────────────────────────────────────
# GOODNESS-OF-FIT STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_fit_statistics(y_obs, y_calc, y_weights, n_params):
    """
    Compute Rwp, Rp, chi-squared, and GoF.
    y_weights = 1/sigma^2
    """
    diff     = y_obs - y_calc
    Rwp      = math.sqrt(np.sum(y_weights * diff**2) / np.sum(y_weights * y_obs**2))
    Rp       = np.sum(np.abs(diff)) / np.sum(np.abs(y_obs))
    chi2     = np.sum(y_weights * diff**2) / max(len(y_obs) - n_params, 1)
    GoF      = math.sqrt(chi2)
    return {'Rwp': round(Rwp*100,2), 'Rp': round(Rp*100,2),
            'chi2': round(chi2,3), 'GoF': round(GoF,3)}
