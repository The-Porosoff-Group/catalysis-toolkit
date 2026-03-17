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
    # Guard against None cell parameters
    if any(v is None for v in [a, b, c, al, be, ga]):
        return None
    a = float(a); b = float(b); c = float(c)
    al = float(al); be = float(be); ga = float(ga)
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
    Systematic absence filter based on lattice centering.
    Covers the most common catalyst phases correctly.
    """
    sg = spacegroup_number

    # Always remove (000)
    if h == 0 and k == 0 and l == 0:
        return False

    # ── Lattice centering conditions ──────────────────────────────────────────

    # F-centred (FCC): h,k,l must be all odd or all even
    # F-lattice cubic SGs: 196,202,203,209,210,216,219,225,226,227,228
    _F_cubic = {196,202,203,209,210,216,219,225,226,227,228}
    if sg in _F_cubic:
        if not ((h%2 == k%2 == l%2 == 0) or (h%2==1 and k%2==1 and l%2==1)):
            return False

    # I-centred (BCC): h+k+l must be even
    # I-lattice cubic SGs: 197,199,204,206,211,214,217,220,229,230
    _I_cubic = {197,199,204,206,211,214,217,220,229,230}
    if sg in _I_cubic:
        if (h+k+l) % 2 != 0:
            return False

    # I-centred tetragonal: h+k+l even (SGs 79-88, 97-98, 107-110, 119-122, 139-142)
    _I_tetrag = set(range(79,89)) | set(range(97,99)) | set(range(107,111)) | \
                set(range(119,123)) | set(range(139,143))
    if sg in _I_tetrag:
        if (h+k+l) % 2 != 0:
            return False

    # I-centred orthorhombic: h+k+l even (SGs 23-24, 44-46, 71-74)
    _I_ortho = {23,24,44,45,46,71,72,73,74}
    if sg in _I_ortho:
        if (h+k+l) % 2 != 0:
            return False

    # C-centred: h+k even
    _C_centred = set(range(5,6)) | set(range(8,10)) | set(range(12,16)) | \
                 set(range(20,22)) | set(range(35,42)) | set(range(63,69)) | \
                 set(range(63,69))
    if sg in _C_centred:
        if (h+k) % 2 != 0:
            return False

    # ── Screw axis / glide plane conditions for common phases ─────────────────

    # Pm-3n (#223, beta-W A15, Cr3Si type):
    # In cubic m-3m, the HM positions are ⟨100⟩, ⟨111⟩, ⟨110⟩.
    # "n" is in the third position → n-glide ⊥ ⟨110⟩.
    # Reflection conditions (ITC Vol A):
    #   hhl: l = 2n   (and equivalents hkh, khh, etc. by cubic symmetry)
    #   00l: l = 2n   (implied serial condition)
    if sg == 223:
        # hhl-type: any two indices equal → the third must be even
        ah, ak, al = abs(h), abs(k), abs(l)
        if ah == ak and al % 2 != 0: return False   # hhl: l=2n
        if ah == al and ak % 2 != 0: return False   # hlh → k=2n
        if ak == al and ah % 2 != 0: return False   # hkk → h=2n
        # Serial: 00l, h00, 0k0 → index must be even (implied by above)
        if h == 0 and k == 0 and l % 2 != 0: return False
        if k == 0 and l == 0 and h % 2 != 0: return False
        if h == 0 and l == 0 and k % 2 != 0: return False

    # P63/mmc (#194, e.g. Mo2C, beta-W2C, graphite, WC):
    # (00l): l must be even (63 screw)
    if sg == 194:
        if h == 0 and k == 0 and l % 2 != 0:
            return False

    # P6/mmm (#191), P63/m (#176), P63 (#173), P6322 (#182), P63mc (#186)
    if sg in (173, 176, 182, 186, 191):
        if h == 0 and k == 0 and l % 2 != 0:
            return False

    # P-6m2 (#187, alpha-WC, hexagonal): no screw absences beyond lattice
    # P-43m (#215), F-43m (#216) — already handled by F centering above

    # Ia-3d (#230, garnets): I-centred + d-glide
    if sg == 230:
        if h == 0 and k == 0 and l % 4 != 0: return False  # 00l: l=4n
        if h == 0 and l == 0 and k % 4 != 0: return False
        if k == 0 and l == 0 and h % 4 != 0: return False

    # Im-3m (#229, alpha-W, alpha-Mo, alpha-Fe): pure I-centring, no screw absences
    # Already handled by _I_cubic set above

    return True

# ─────────────────────────────────────────────────────────────────────────────
# ATOMIC SCATTERING FACTORS  (Cromer-Mann 9-parameter coefficients)
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (a1,b1, a2,b2, a3,b3, a4,b4, c)
# Source: International Tables for Crystallography, Vol. C, Table 6.1.1.4
# f0(s) = c + Σ_i a_i * exp(-b_i * s²)  where s = sin(θ)/λ in Å⁻¹
CROMER_MANN = {
    'H':  (0.489918,20.6593, 0.262003,7.74039, 0.196767,49.5519, 0.049879,2.20159, 0.001305),
    'C':  (2.31000,20.8439, 1.02000,10.2075, 1.58860,0.568700, 0.865000,51.6512, 0.215600),
    'N':  (12.2126,0.005700, 3.13220,9.89330, 2.01250,28.9975, 1.16630,0.582600,-11.529),
    'O':  (3.04850,13.2771, 2.28680,5.70110, 1.54630,0.323900, 0.867000,32.9089, 0.250800),
    'Si': (6.29150,2.43860, 3.03530,32.3337, 1.98910,0.678500, 1.54100,81.6937, 1.14070),
    'P':  (6.43450,1.90670, 4.17910,27.1570, 1.78000,0.526000, 1.49080,68.1645, 1.11490),
    'S':  (6.90530,1.46790, 5.20340,22.2151, 1.43790,0.253600, 1.58630,56.1720, 0.866900),
    'Ti': (9.75950,7.85080, 7.35580,0.500000, 1.69910,35.6338, 1.90210,116.105, 1.28070),
    'V':  (10.2971,6.86570, 7.35110,0.438500, 2.07030,26.8938, 2.05710,102.478, 1.21990),
    'Cr': (10.6406,6.10380, 7.35370,0.392000, 3.32400,20.2626, 1.49220,98.7399, 1.18320),
    'Mn': (11.2819,5.34090, 7.35730,0.343200, 3.01930,17.8674, 2.24410,83.7543, 1.08960),
    'Fe': (11.7695,4.76110, 7.35730,0.307200, 3.52220,15.3535, 2.30450,76.8805, 1.03690),
    'Co': (12.2841,4.27910, 7.34090,0.278400, 4.00340,13.5359, 2.34880,71.1692, 1.01180),
    'Ni': (12.8376,3.87850, 7.29200,0.256500, 4.44380,12.1763, 2.38000,66.3421, 1.03410),
    'Cu': (13.3380,3.58280, 7.16760,0.247000, 5.61580,11.3966, 1.67350,64.8126, 1.19100),
    'Zn': (14.0743,3.26550, 7.03180,0.233300, 5.16520,10.3163, 2.41000,58.7097, 1.30410),
    'Zr': (17.8765,1.27618, 10.9480,11.9160, 5.41732,0.117622, 3.65721,87.6627, 2.06929),
    'Nb': (17.6142,1.18865, 12.0144,11.7660, 4.04183,0.204785, 3.53346,69.7957, 3.75591),
    'Mo': (3.70250,0.277200, 17.2356,1.09580, 12.8876,11.0040, 3.74290,61.6584, 4.38750),
    'Ru': (19.2674,0.808520, 12.9182,8.43467, 4.86337,24.7997, 1.56756,94.2928, 5.37874),
    'Pd': (19.3319,0.698655, 15.5017,7.98929, 5.29537,25.2052, 0.605844,76.8986, 5.26593),
    'Ag': (19.2808,0.644600, 16.6885,7.47260, 4.80450,24.6605, 1.04630,99.8156, 5.17900),
    'Sn': (19.1889,5.83030, 19.1005,0.503100, 4.45850,26.8909, 2.46630,83.9571, 4.78210),
    'Hf': (29.1440,1.83260, 15.1726,6.70840, 14.7586,0.321800, 4.30013,25.8449,-6.53900),
    'Ta': (29.2024,1.77330, 15.2293,6.43530, 14.5135,0.295100, 4.76492,23.8132,-6.07300),
    'W':  (29.0818,1.72029, 15.4300,6.59432, 14.4327,0.321703, 5.11982,25.2017,-2.98400),
    'Re': (28.7621,1.67190, 15.7189,6.75390, 14.5564,0.334200, 5.44174,24.9604,-2.30400),
    'Os': (28.1894,1.62903, 16.1550,6.83390, 14.9305,0.344000, 5.67589,25.4571,-1.63600),
    'Pt': (27.0059,1.51293, 17.7639,8.81174, 15.7131,0.424593, 5.78370,38.6103, 11.6883),
    'Au': (16.8819,0.461100, 18.5913,8.62160, 25.5582,1.48260, 5.86000,36.3956, 12.0658),
    'Pb': (31.0617,0.690200, 13.0637,2.35760, 18.4420,8.61800, 5.96960,47.2579, 13.4118),
    'Al': (6.42020,3.03870, 1.90020,0.742600, 1.59360,31.5472, 1.96460,85.0886, 1.11510),
    'B':  (2.05450,23.2185, 1.33260,1.02100, 1.09790,60.3498, 0.706800,0.140300,-0.19320),
}


def atomic_scattering_factor(element, s):
    """
    Compute f0 for a given element at s = sin(θ)/λ (in Å⁻¹).
    Falls back to Z/4 approximation if element not in table.
    """
    el = element.strip().capitalize()
    params = CROMER_MANN.get(el)
    if params is None:
        # Rough fallback: use number of electrons / 4
        # (very rough but better than nothing)
        return 10.0
    a1,b1, a2,b2, a3,b3, a4,b4, c = params
    s2 = s * s
    return (c + a1*math.exp(-b1*s2) + a2*math.exp(-b2*s2)
              + a3*math.exp(-b3*s2) + a4*math.exp(-b4*s2))


# ─────────────────────────────────────────────────────────────────────────────
# MOLAR MASS FROM CHEMICAL FORMULA
# ─────────────────────────────────────────────────────────────────────────────

_ATOMIC_MASS = {
    'H':1.008,'He':4.003,'Li':6.941,'Be':9.012,'B':10.81,'C':12.01,'N':14.01,
    'O':16.00,'F':19.00,'Ne':20.18,'Na':22.99,'Mg':24.31,'Al':26.98,'Si':28.09,
    'P':30.97,'S':32.07,'Cl':35.45,'Ar':39.95,'K':39.10,'Ca':40.08,'Sc':44.96,
    'Ti':47.87,'V':50.94,'Cr':52.00,'Mn':54.94,'Fe':55.85,'Co':58.93,'Ni':58.69,
    'Cu':63.55,'Zn':65.38,'Ga':69.72,'Ge':72.63,'As':74.92,'Se':78.97,'Br':79.90,
    'Rb':85.47,'Sr':87.62,'Y':88.91,'Zr':91.22,'Nb':92.91,'Mo':95.95,'Ru':101.1,
    'Rh':102.9,'Pd':106.4,'Ag':107.9,'Cd':112.4,'In':114.8,'Sn':118.7,'Sb':121.8,
    'Te':127.6,'I':126.9,'Cs':132.9,'Ba':137.3,'La':138.9,'Ce':140.1,'Pr':140.9,
    'Nd':144.2,'Sm':150.4,'Eu':152.0,'Gd':157.3,'Tb':158.9,'Dy':162.5,'Ho':164.9,
    'Er':167.3,'Tm':168.9,'Yb':173.0,'Lu':175.0,'Hf':178.5,'Ta':180.9,'W':183.8,
    'Re':186.2,'Os':190.2,'Ir':192.2,'Pt':195.1,'Au':197.0,'Hg':200.6,'Tl':204.4,
    'Pb':207.2,'Bi':209.0,'Th':232.0,'U':238.0,
}


def molar_mass_from_formula(formula):
    """
    Parse a chemical formula string and return molar mass in g/mol.
    Handles CIF-style formulas like 'Mo2 C1', 'Fe3 O4', 'Si O2'.
    Returns None if formula cannot be parsed.
    """
    if not formula:
        return None
    total = 0.0
    # Match element symbol + optional count (integer or decimal)
    for el, count in re.findall(r'([A-Z][a-z]?)\s*(\d*\.?\d*)', formula):
        mass = _ATOMIC_MASS.get(el)
        if mass is None:
            return None
        n = float(count) if count else 1.0
        total += mass * n
    return total if total > 0 else None


def structure_factor_sq(h, k, l, sites, sin_theta_over_lambda):
    """
    Compute |F(hkl)|² from a list of atom sites.

    sites: list of (element, x, y, z, occupancy)
           where x,y,z are fractional coordinates.
    sin_theta_over_lambda: sin(θ)/λ in Å⁻¹

    Returns |F|² (float).
    """
    F_real = 0.0
    F_imag = 0.0
    s = sin_theta_over_lambda
    for (el, x, y, z, occ) in sites:
        f0 = atomic_scattering_factor(el, s)
        phase = 2.0 * math.pi * (h*x + k*y + l*z)
        F_real += occ * f0 * math.cos(phase)
        F_imag += occ * f0 * math.sin(phase)
    return F_real * F_real + F_imag * F_imag


def structure_factor_sq_dw(h, k, l, sites, sin_theta_over_lambda, B_iso_map=None):
    """
    Compute |F(hkl)|² with isotropic Debye-Waller (thermal) factors.

    sites: list of (element, x, y, z, occupancy)
    sin_theta_over_lambda: sin(θ)/λ in Å⁻¹
    B_iso_map: dict {element: B_iso} in Å². If None, B_iso=0 (no thermal damping).
               Typical values: 0.3–1.0 Å² for metals, 0.5–2.0 Å² for oxides.

    The Debye-Waller factor is: T = exp(-B_iso × (sin θ / λ)²)
    This damps high-angle reflections, which is physically correct —
    thermal vibrations blur the electron density and reduce coherent scattering.

    Returns |F|² (float).
    """
    F_real = 0.0
    F_imag = 0.0
    s = sin_theta_over_lambda
    s2 = s * s
    for (el, x, y, z, occ) in sites:
        f0 = atomic_scattering_factor(el, s)
        # Debye-Waller factor
        B = 0.0
        if B_iso_map is not None:
            B = B_iso_map.get(el, B_iso_map.get('_all', 0.0))
        dw = math.exp(-B * s2)
        phase = 2.0 * math.pi * (h*x + k*y + l*z)
        F_real += occ * f0 * dw * math.cos(phase)
        F_imag += occ * f0 * dw * math.sin(phase)
    return F_real * F_real + F_imag * F_imag


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_reflections(a, b, c, al, be, ga, system, spacegroup_number,
                          wavelength, two_theta_min, two_theta_max, hkl_max=10,
                          sites=None):
    """
    Generate list of [two_theta, d, (h,k,l), intensity_weight] for all
    allowed reflections in the given 2θ range.

    If `sites` is provided (list of (element, x, y, z, occupancy) tuples),
    the intensity weight is:  multiplicity × |F(hkl)|² × LP
    This correctly produces zero-intensity for structure-factor-extinct peaks.

    If `sites` is None (no atom positions available), the intensity weight
    is just the multiplicity (backward-compatible behaviour).

    Multiplicity is determined by counting every symmetry-equivalent
    (h,k,l) in the full sphere that passes the systematic-absence filter
    and maps to the same d-spacing.
    """
    seen_d = {}   # d_key → [two_theta, d, (h,k,l), count, F2_representative]

    for h in range(-hkl_max, hkl_max + 1):
        for k in range(-hkl_max, hkl_max + 1):
            for l in range(-hkl_max, hkl_max + 1):
                if h == 0 and k == 0 and l == 0:
                    continue

                if not is_allowed(h, k, l, spacegroup_number):
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
                    seen_d[d_key][3] += 1        # count this equivalent
                else:
                    # Compute |F|² once per unique d-spacing (representative hkl)
                    F2 = None
                    if sites is not None:
                        s = sin_theta / wavelength
                        F2 = structure_factor_sq(h, k, l, sites, s)
                    seen_d[d_key] = [two_theta, d, (abs(h), abs(k), abs(l)), 1, F2]

    # Build final list with intensity weights
    reflections = []
    for entry in sorted(seen_d.values(), key=lambda x: x[0]):
        two_theta, d, hkl, mult, F2 = entry
        if sites is not None:
            # Skip reflections with negligible structure factor
            if F2 is not None and F2 < 1e-4:
                continue
            # Intensity weight = mult × |F|² (LP applied later in compute_phase_pattern)
            weight = mult * (F2 if F2 is not None else 1.0)
        else:
            weight = mult
        reflections.append([two_theta, d, hkl, weight])

    return reflections  # [(two_theta, d, (h,k,l), intensity_weight), ...]


def generate_reflections_rietveld(a, b, c, al, be, ga, system, spacegroup_number,
                                   wavelength, two_theta_min, two_theta_max,
                                   sites, hkl_max=12):
    """
    Generate reflection list for Rietveld refinement.

    Unlike generate_reflections(), this returns per-reflection components
    that allow recomputing intensities when B_iso changes without
    re-enumerating all hkl:

    Returns list of dicts:
        {'two_theta', 'd', 'hkl', 'mult', 'sin_theta_over_lambda',
         'h_rep', 'k_rep', 'l_rep'}

    The representative (h,k,l) is stored so structure_factor_sq_dw() can
    be called with varying B_iso during refinement.
    """
    seen_d = {}

    for h in range(-hkl_max, hkl_max + 1):
        for k in range(-hkl_max, hkl_max + 1):
            for l in range(-hkl_max, hkl_max + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                if not is_allowed(h, k, l, spacegroup_number):
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
                    seen_d[d_key]['mult'] += 1
                else:
                    # Check if this reflection has nonzero F² (skip extinct peaks)
                    s = sin_theta / wavelength
                    F2_test = structure_factor_sq(h, k, l, sites, s)
                    if F2_test < 1e-4:
                        seen_d[d_key] = {'skip': True, 'mult': 1}
                        continue
                    seen_d[d_key] = {
                        'two_theta': two_theta,
                        'd': d,
                        'hkl': (abs(h), abs(k), abs(l)),
                        'mult': 1,
                        'sin_theta_over_lambda': s,
                        'h_rep': h, 'k_rep': k, 'l_rep': l,
                        'skip': False,
                    }

    refs = [v for v in sorted(seen_d.values(), key=lambda x: x.get('two_theta', 999))
            if not v.get('skip', False)]
    return refs


def compute_rietveld_intensities(refs, sites, B_iso_map=None):
    """
    Compute per-reflection intensities for a Rietveld model:
      I_hkl = mult × |F(hkl)|² × LP

    refs: output of generate_reflections_rietveld()
    sites: atom site list
    B_iso_map: {element: B_iso} or {'_all': B_iso}

    Returns np.array of intensities, same length as refs.
    """
    intensities = np.zeros(len(refs))
    for i, ref in enumerate(refs):
        s = ref['sin_theta_over_lambda']
        h, k, l = ref['h_rep'], ref['k_rep'], ref['l_rep']
        F2 = structure_factor_sq_dw(h, k, l, sites, s, B_iso_map)
        # Lorentz-polarisation factor
        tt = ref['two_theta']
        theta_r = math.radians(tt / 2)
        cos2t = math.cos(math.radians(tt))
        sin_t = math.sin(theta_r)
        cos_t = math.cos(theta_r)
        LP = (1 + cos2t**2) / (sin_t**2 * cos_t) if sin_t > 0 and cos_t > 0 else 1.0
        intensities[i] = ref['mult'] * F2 * LP
    return intensities


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
    Minimal CIF parser — extracts unit cell, space group, formula,
    and atom site coordinates (fractional).

    Returns dict with keys: a, b, c, alpha, beta, gamma, spacegroup_number,
    spacegroup_name, system, formula, cod_id, sites

    sites: list of (element, x, y, z, occupancy) tuples parsed from the
    _atom_site loop. Empty list if no atom sites found.
    """
    result = {
        'a': None, 'b': None, 'c': None,
        'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0,
        'spacegroup_number': 1,
        'spacegroup_name': 'P 1',
        'system': 'triclinic',
        'formula': '',
        'cod_id': '',
        'sites': [],
        'Z': None,
    }

    def parse_val(s):
        """Strip uncertainty e.g. '3.002(5)' → 3.002"""
        s = s.strip().strip("'\"")
        m = re.match(r'^([0-9\.\-\+eE]+)', s)
        return float(m.group(1)) if m else None

    lines = cif_text.splitlines()

    # ── Pass 1: scalar CIF tags ──────────────────────────────────────────
    for line in lines:
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
        elif line.startswith('_cell_formula_units_Z'):
            v = parse_val(line.split()[-1])
            if v: result['Z'] = int(v)

    # ── Pass 2: parse atom_site loop ─────────────────────────────────────
    sites = _parse_atom_site_loop(lines, parse_val)
    if sites:
        result['sites'] = sites

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


def _parse_atom_site_loop(lines, parse_val):
    """
    Extract atom sites from a CIF _atom_site loop block.
    Returns list of (element, x, y, z, occupancy) tuples.

    Handles both _atom_site_type_symbol and _atom_site_label as the
    element identifier, and tolerates varying column orders.
    """
    sites = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        # Find a loop_ that contains _atom_site_ tags
        if line.lower() == 'loop_':
            i += 1
            # Collect column headers
            headers = []
            while i < n:
                hline = lines[i].strip()
                if hline.startswith('_atom_site_'):
                    headers.append(hline.lower())
                    i += 1
                else:
                    break
            if not headers:
                continue

            # Identify column indices for fields we need
            col_type  = None  # _atom_site_type_symbol (preferred)
            col_label = None  # _atom_site_label (fallback for element)
            col_x = col_y = col_z = col_occ = None
            for ci, h in enumerate(headers):
                if h == '_atom_site_type_symbol':     col_type  = ci
                elif h == '_atom_site_label':          col_label = ci
                elif h == '_atom_site_fract_x':        col_x     = ci
                elif h == '_atom_site_fract_y':        col_y     = ci
                elif h == '_atom_site_fract_z':        col_z     = ci
                elif h == '_atom_site_occupancy':      col_occ   = ci

            # Must have at least fractional coordinates and some element id
            el_col = col_type if col_type is not None else col_label
            if el_col is None or col_x is None or col_y is None or col_z is None:
                continue

            # Read data rows
            while i < n:
                dline = lines[i].strip()
                if not dline or dline.startswith('_') or dline.lower() == 'loop_' or dline.startswith('#'):
                    break
                parts = dline.split()
                if len(parts) <= max(el_col, col_x, col_y, col_z):
                    break
                try:
                    el_raw = parts[el_col].strip("'\"")
                    # Extract element symbol: strip trailing digits/charges
                    el = re.match(r'([A-Z][a-z]?)', el_raw)
                    if not el:
                        i += 1
                        continue
                    element = el.group(1)
                    x = parse_val(parts[col_x])
                    y = parse_val(parts[col_y])
                    z = parse_val(parts[col_z])
                    occ = parse_val(parts[col_occ]) if col_occ is not None and col_occ < len(parts) else 1.0
                    if x is not None and y is not None and z is not None:
                        sites.append((element, x, y, z, occ if occ is not None else 1.0))
                except (IndexError, ValueError):
                    pass
                i += 1
            continue
        i += 1

    return sites


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
