"""
modules/xrd/lebail.py
Le Bail profile fitting engine — pymatgen-powered.

Uses pymatgen's XRDCalculator for correct structure factor + LP computation
to seed I_hkl. Falls back to our crystallography.py if pymatgen unavailable.

Outer loop:
  A) Le Bail I_hkl update  (intensity partitioning)
  B) Global scale factor   (linear regression)
  C) Background update     (Chebyshev polynomial)
  D) Cell + profile params (scipy least_squares)
"""

import math, warnings
import numpy as np
from scipy.optimize import least_squares

from .crystallography import (
    generate_reflections, chebyshev_background,
    caglioti_fwhm, scherrer_size, pseudo_voigt,
    compute_fit_statistics, d_spacing
)

# ─────────────────────────────────────────────────────────────────────────────
# PYMATGEN INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def _try_import_pymatgen():
    try:
        from pymatgen.core import Structure
        from pymatgen.analysis.diffraction.xrd import XRDCalculator
        return Structure, XRDCalculator
    except ImportError:
        return None, None


def _cif_to_structure(cif_text):
    """Parse CIF text into a pymatgen Structure. Returns None on failure."""
    Structure, _ = _try_import_pymatgen()
    if Structure is None:
        return None
    import tempfile, os
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cif',
                                         delete=False) as f:
            f.write(cif_text)
            path = f.name
        struct = Structure.from_file(path)
        os.unlink(path)
        return struct
    except Exception:
        return None


def get_pymatgen_intensities(cif_text, wavelength, tt_min, tt_max):
    """
    Use pymatgen XRDCalculator to get per-reflection intensities
    with proper structure factors and LP correction.

    Returns dict: {two_theta_rounded: intensity} or None if unavailable.
    """
    _, XRDCalculator = _try_import_pymatgen()
    if XRDCalculator is None:
        return None

    struct = _cif_to_structure(cif_text)
    if struct is None:
        return None

    try:
        calc    = XRDCalculator(wavelength=wavelength)
        pattern = calc.get_pattern(struct, scaled=False,
                                    two_theta_range=(tt_min, tt_max))
        # pattern.x = 2theta positions, pattern.y = intensities
        intensity_map = {}
        for tt, I in zip(pattern.x, pattern.y):
            intensity_map[round(float(tt), 3)] = float(I)
        return intensity_map
    except Exception:
        return None


def seed_I_hkl_from_pymatgen(refs, intensity_map, tt_r, y_r, bg_est):
    """
    Seed I_hkl values using pymatgen intensities where available,
    falling back to observed peak height for unmatched reflections.
    """
    I_hkl = np.zeros(len(refs))
    pm_max = max(intensity_map.values()) if intensity_map else 1.0

    # Scale pymatgen intensities to match observed data range
    obs_range = max(y_r.max() - bg_est, 1.0)
    pm_scale  = obs_range / pm_max if pm_max > 0 else 1.0

    for k, (tt_peak, d, hkl, mult) in enumerate(refs):
        # Look for matching pymatgen peak within 0.1°
        best_I = None
        best_dist = 0.1
        for pm_tt, pm_I in (intensity_map or {}).items():
            dist = abs(pm_tt - tt_peak)
            if dist < best_dist:
                best_dist = dist
                best_I = pm_I

        if best_I is not None:
            I_hkl[k] = best_I * pm_scale
        else:
            # Fallback: use observed data height near this peak
            near = np.abs(tt_r - tt_peak) < 0.5
            I_hkl[k] = max(y_r[near].max() - bg_est, 1.0) if near.any() else 1.0

    return np.maximum(I_hkl, 0.01)


# ─────────────────────────────────────────────────────────────────────────────
# PROFILE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _get_profiles(tt_arr, refs, U, V, W, eta, zero=0.0):
    """Unit-normalised pseudo-Voigt profiles, one per reflection."""
    profiles = []
    tt_s = tt_arr - zero
    for tt_p, d, hkl, mult in refs:
        fwhm = max(caglioti_fwhm(tt_p, U, V, W), 0.005)
        prof = np.zeros(len(tt_arr))
        msk  = np.abs(tt_s - tt_p) < 20 * fwhm
        if msk.any():
            prof[msk] = pseudo_voigt(tt_s[msk], tt_p, fwhm, eta)
        profiles.append(prof)
    return profiles


# ─────────────────────────────────────────────────────────────────────────────
# CELL PARAMETER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _cell_free(phase):
    """Return (free_values, free_names) for this crystal system."""
    a  = float(phase['a'])
    b  = float(phase.get('b') or a)
    c  = float(phase.get('c') or a)
    be = float(phase.get('beta', 90.0)  or 90.0)
    al = float(phase.get('alpha', 90.0) or 90.0)
    ga = float(phase.get('gamma', 90.0) or 90.0)
    sys_ = (phase.get('system') or 'triclinic').lower()
    if sys_ == 'cubic':
        return [a], ['a']
    elif sys_ == 'tetragonal':
        return [a, c], ['a', 'c']
    elif sys_ in ('hexagonal', 'trigonal'):
        return [a, c], ['a', 'c']
    elif sys_ == 'orthorhombic':
        return [a, b, c], ['a', 'b', 'c']
    elif sys_ == 'monoclinic':
        return [a, b, c, be], ['a', 'b', 'c', 'beta']
    else:
        return [a, b, c, al, be, ga], ['a', 'b', 'c', 'alpha', 'beta', 'gamma']


def _full_cell(free_vals, free_names, phase):
    """Reconstruct (a,b,c,al,be,ga) from free params + symmetry constraints."""
    d = {
        'a': float(phase['a']), 'b': float(phase.get('b') or phase['a']),
        'c': float(phase.get('c') or phase['a']),
        'alpha': float(phase.get('alpha', 90.0) or 90.0),
        'beta':  float(phase.get('beta',  90.0) or 90.0),
        'gamma': float(phase.get('gamma', 90.0) or 90.0),
    }
    for name, val in zip(free_names, free_vals):
        d[name] = val
    sys_ = (phase.get('system') or 'triclinic').lower()
    if sys_ == 'cubic':
        d['b'] = d['a']; d['c'] = d['a']
    elif sys_ in ('hexagonal', 'trigonal', 'tetragonal'):
        d['b'] = d['a']
    return d['a'], d['b'], d['c'], d['alpha'], d['beta'], d['gamma']


# ─────────────────────────────────────────────────────────────────────────────
# MAIN REFINEMENT
# ─────────────────────────────────────────────────────────────────────────────

def run_lebail(tt, y_obs, sigma, phases, wavelength,
               tt_min=None, tt_max=None,
               n_bg_coeffs=6, max_outer=15,
               progress_callback=None):
    """
    Run Le Bail refinement.

    Parameters
    ----------
    tt, y_obs, sigma : np.ndarray — data arrays
    phases   : list of dicts with keys:
                 name, system, spacegroup_number, a, [b, c, alpha, beta, gamma]
                 optional: cif_text (enables pymatgen seeding)
    wavelength : float — Å
    tt_min/max : float — refinement range
    n_bg_coeffs: int
    max_outer  : int

    Returns dict with tt, y_obs, y_calc, y_background, phase_patterns,
                      residuals, statistics, phase_results, zero_shift
    """
    if tt_min is None: tt_min = float(tt.min())
    if tt_max is None: tt_max = float(tt.max())

    mask   = (tt >= tt_min) & (tt <= tt_max)
    tt_r   = tt[mask]
    y_r    = y_obs[mask]
    sig_r  = (sigma[mask] if sigma is not None
              else np.sqrt(np.maximum(y_r, 1.0)))
    weights = 1.0 / np.maximum(sig_r**2, 1e-6)

    if progress_callback: progress_callback('Generating reflections...')

    # ── Per-phase state ───────────────────────────────────────────────────
    phase_state = []
    bg_est = float(np.percentile(y_r, 5))

    for ph in phases:
        free_v, free_n = _cell_free(ph)
        a, b, c, al, be, ga = _full_cell(free_v, free_n, ph)
        refs = generate_reflections(
            a, b, c, al, be, ga,
            ph.get('system', 'triclinic'),
            ph.get('spacegroup_number', 1),
            wavelength, tt_min, tt_max, hkl_max=12
        )

        # Seed I_hkl — use pymatgen if CIF is available
        cif_text = ph.get('cif_text')
        pm_map   = None
        if cif_text:
            if progress_callback:
                progress_callback(f"Computing structure factors for {ph.get('name','?')}...")
            pm_map = get_pymatgen_intensities(cif_text, wavelength, tt_min, tt_max)

        if pm_map:
            I_hkl = seed_I_hkl_from_pymatgen(refs, pm_map, tt_r, y_r, bg_est)
            seeded = 'pymatgen'
        else:
            # Fallback: use observed peak heights
            I_hkl = np.zeros(len(refs))
            for k, (tt_p, d, hkl, mult) in enumerate(refs):
                near = np.abs(tt_r - tt_p) < 0.5
                I_hkl[k] = max(y_r[near].max() - bg_est, 1.0) if near.any() else 1.0
            seeded = 'observed'

        phase_state.append({
            'ph':     ph,
            'refs':   refs,
            'I_hkl':  I_hkl,
            'U':  ph.get('U_init',   0.0),
            'V':  ph.get('V_init',   0.0),
            'W':  ph.get('W_init',   0.37),
            'eta':ph.get('eta_init', 0.5),
            'free_v': free_v,
            'free_n': free_n,
            'seeded': seeded,
        })

    n_bg  = n_bg_coeffs
    bg_c  = np.zeros(n_bg); bg_c[0] = bg_est
    S     = 1.0
    zero  = 0.0

    def total_calc(states, bg_c_, S_, zero_):
        bg_  = np.maximum(chebyshev_background(tt_r, bg_c_, tt_min, tt_max), 0)
        pat_ = np.zeros(len(tt_r))
        for st in states:
            profs = _get_profiles(tt_r, st['refs'],
                                   st['U'], st['V'], st['W'], st['eta'], zero_)
            for k in range(len(st['refs'])):
                pat_ += st['I_hkl'][k] * profs[k]
        return S_ * pat_ + bg_, bg_, S_ * pat_

    def rwp(yo, yc, w):
        d = yo - yc
        return math.sqrt(np.sum(w*d**2) / np.sum(w*yo**2)) * 100

    if progress_callback: progress_callback('Running Le Bail iterations...')

    prev_rwp = 999.0
    for outer in range(max_outer):

        # ── A: Le Bail I_hkl update ───────────────────────────────────────
        # Cache profiles once per outer iteration (major speedup)
        all_profs = [_get_profiles(tt_r, st['refs'],
                                    st['U'], st['V'], st['W'], st['eta'], zero)
                     for st in phase_state]

        for inner in range(30):  # 30 is plenty — early-exit handles the rest
            bg_cur  = np.maximum(chebyshev_background(tt_r, bg_c, tt_min, tt_max), 0)
            pat_all = np.zeros(len(tt_r))
            for i_ph, st in enumerate(phase_state):
                for k in range(len(st['refs'])):
                    pat_all += st['I_hkl'][k] * all_profs[i_ph][k]
            sum_c  = np.maximum(S * pat_all, 1e-8)
            y_nobg = np.maximum(y_r - bg_cur, 0.1)

            max_chg = 0.0
            for i_ph, st in enumerate(phase_state):
                I_new = np.zeros(len(st['refs']))
                for k in range(len(st['refs'])):
                    phi_k = all_profs[i_ph][k]
                    if phi_k.sum() < 1e-10:
                        I_new[k] = st['I_hkl'][k]; continue
                    ratio  = S * st['I_hkl'][k] * phi_k / sum_c
                    numer  = np.sum(weights * y_nobg * ratio)
                    denom  = np.sum(weights * phi_k) * S
                    I_new[k] = max(numer / max(denom, 1e-10), 1e-6)
                I_new = np.clip(I_new, 1e-6, 1e7)
                max_chg = max(max_chg, np.max(np.abs(I_new - st['I_hkl'])))
                st['I_hkl'] = I_new
            if max_chg < 0.1: break  # looser tolerance — outer loop handles refinement

        # ── B: Update global scale ────────────────────────────────────────
        pat_cur = np.zeros(len(tt_r))
        for st in phase_state:
            profs = _get_profiles(tt_r, st['refs'],
                                   st['U'], st['V'], st['W'], st['eta'], zero)
            for k in range(len(st['refs'])):
                pat_cur += st['I_hkl'][k] * profs[k]
        bg_cur = np.maximum(chebyshev_background(tt_r, bg_c, tt_min, tt_max), 0)
        y_nobg2 = y_r - bg_cur
        S_num = np.sum(weights * y_nobg2 * pat_cur)
        S_den = np.sum(weights * pat_cur**2)
        S = max(S_num / max(S_den, 1e-10), 1e-4)

        # ── C: Update background ──────────────────────────────────────────
        yc_noBG = S * pat_cur
        def resid_bg(x):
            bg_ = np.maximum(chebyshev_background(tt_r, x, tt_min, tt_max), 0)
            return (y_r - yc_noBG - bg_) * np.sqrt(weights)
        rb = least_squares(resid_bg, bg_c, method='trf',
                           max_nfev=200, ftol=1e-10, verbose=0)
        bg_c = rb.x

        # ── D: Refine cell + profile per phase ────────────────────────────
        if progress_callback and outer == 0:
            progress_callback('Refining cell parameters and peak shape...')

        for i_ph, st in enumerate(phase_state):
            ph    = st['ph']
            sys_  = (ph.get('system') or 'triclinic').lower()
            free_n = st['free_n']

            def resid_cp(x):
                nf = len(free_n)
                fv  = x[:nf]
                U_, V_, W_, eta_, zero_ = x[nf:nf+5]
                W_ = max(W_, 0.005); eta_ = np.clip(eta_, 0, 1)
                a_,b_,c_,al_,be_,ga_ = _full_cell(fv, free_n, ph)
                try:
                    refs_l = generate_reflections(
                        a_,b_,c_,al_,be_,ga_,sys_,
                        ph.get('spacegroup_number', 1),
                        wavelength, tt_min, tt_max, hkl_max=12)
                except Exception:
                    refs_l = st['refs']
                n = min(len(refs_l), len(st['I_hkl']))
                profs_l = _get_profiles(tt_r, refs_l, U_,V_,W_,eta_,zero_)
                bg_ = np.maximum(chebyshev_background(tt_r, bg_c, tt_min, tt_max), 0)
                pat_ = np.zeros(len(tt_r))
                # Other phases
                for j, st2 in enumerate(phase_state):
                    if j == i_ph: continue
                    p2 = _get_profiles(tt_r, st2['refs'],
                                        st2['U'], st2['V'], st2['W'], st2['eta'], zero)
                    for k in range(len(st2['refs'])):
                        pat_ += st2['I_hkl'][k] * p2[k]
                pat_ += sum(st['I_hkl'][k]*profs_l[k] for k in range(n))
                return (y_r - (S*pat_ + bg_)) * np.sqrt(weights)

            fv0 = list(st['free_v'])
            x0_ = fv0 + [st['U'], st['V'], st['W'], st['eta'], zero]
            lo_ = [v*0.94 for v in fv0] + [0.0, -0.5, 0.005, 0.0, -0.3]
            hi_ = [v*1.06 for v in fv0] + [2.0,  0.5, 2.0,   1.0,  0.3]
            # Angle params: no tight bounds
            for i_p, nm in enumerate(free_n):
                if nm in ('alpha','beta','gamma'):
                    lo_[i_p] = fv0[i_p] - 5
                    hi_[i_p] = fv0[i_p] + 5

            try:
                rs = least_squares(resid_cp, x0_, bounds=(lo_, hi_),
                                   method='trf', max_nfev=600,
                                   ftol=1e-10, xtol=1e-10, verbose=0)
                x_out = rs.x
            except Exception:
                x_out = x0_

            nf = len(free_n)
            fv_new = x_out[:nf]
            U_n, V_n, W_n, eta_n, zero = x_out[nf:nf+5]
            W_n = max(W_n, 0.005); eta_n = np.clip(eta_n, 0, 1)

            a_n,b_n,c_n,al_n,be_n,ga_n = _full_cell(fv_new, free_n, ph)
            try:
                refs_new = generate_reflections(
                    a_n,b_n,c_n,al_n,be_n,ga_n,sys_,
                    ph.get('spacegroup_number',1),
                    wavelength, tt_min, tt_max, hkl_max=12)
            except Exception:
                refs_new = st['refs']

            n_new = len(refs_new)
            I_new = np.ones(n_new)
            n_copy = min(n_new, len(st['I_hkl']))
            I_new[:n_copy] = st['I_hkl'][:n_copy]

            # Update phase dict
            for nm, vl in zip(free_n, fv_new):
                ph[nm] = vl
            if sys_ in ('cubic',):
                ph['b'] = ph['c'] = a_n
            elif sys_ in ('hexagonal','trigonal','tetragonal'):
                ph['b'] = a_n
            ph['c'] = c_n

            st.update({'free_v': fv_new, 'refs': refs_new, 'I_hkl': I_new,
                       'U': U_n, 'V': V_n, 'W': W_n, 'eta': eta_n})

        yc_cur, _, _ = total_calc(phase_state, bg_c, S, zero)
        cur_rwp = rwp(y_r, yc_cur, weights)
        if progress_callback:
            progress_callback(f'Outer iteration {outer+1}: Rwp = {cur_rwp:.2f}%')
        if abs(prev_rwp - cur_rwp) < 0.02:  # stop when Rwp changes less than 0.02%
            break
        prev_rwp = cur_rwp

    # ── Final results ─────────────────────────────────────────────────────
    if progress_callback: progress_callback('Computing statistics...')

    yc_f, bg_f, pat_f_total = total_calc(phase_state, bg_c, S, zero)
    diff_f = y_r - yc_f
    n_params = (sum(len(st['refs']) + len(st['free_n']) + 5
                    for st in phase_state) + n_bg + 1)
    stats = compute_fit_statistics(y_r, yc_f, weights, n_params)

    phase_patterns = []
    phase_results  = []
    total_pat_area = sum(
        sum(st['I_hkl'][k]*_get_profiles(tt_r, st['refs'],
            st['U'], st['V'], st['W'], st['eta'], zero)[k].sum()
            for k in range(len(st['refs'])))
        for st in phase_state
    )

    for i_ph, st in enumerate(phase_state):
        profs = _get_profiles(tt_r, st['refs'],
                               st['U'], st['V'], st['W'], st['eta'], zero)
        pat_ph = S * sum(st['I_hkl'][k]*profs[k] for k in range(len(st['refs'])))
        phase_patterns.append(pat_ph.tolist())

        ph = st['ph']
        a  = float(ph.get('a', 4.0))
        b  = float(ph.get('b') or a)
        c  = float(ph.get('c') or a)
        al = float(ph.get('alpha', 90.0) or 90.0)
        be = float(ph.get('beta',  90.0) or 90.0)
        ga = float(ph.get('gamma', 90.0) or 90.0)

        strongest_tt = (max(st['refs'], key=lambda r: r[3])[0]
                        if st['refs'] else 39.0)
        fwhm_sc = caglioti_fwhm(strongest_tt, st['U'], st['V'], st['W'])
        cryst_A = scherrer_size(fwhm_sc, strongest_tt, wavelength)

        # Weight fraction: proportional to scale * integrated pattern area
        ph_area = sum(st['I_hkl'][k]*profs[k].sum()
                      for k in range(len(st['refs'])))
        wt_frac = (ph_area / max(total_pat_area, 1e-10)) * 100

        phase_results.append({
            'name':              ph.get('name', f'Phase {i_ph+1}'),
            'cod_id':            ph.get('cod_id', ''),
            'formula':           ph.get('formula', ''),
            'a': round(a,5), 'b': round(b,5), 'c': round(c,5),
            'alpha':round(al,3),'beta':round(be,3),'gamma':round(ga,3),
            'system':            (ph.get('system') or 'triclinic').lower(),
            'spacegroup_number': ph.get('spacegroup_number', 1),
            'spacegroup':        ph.get('spacegroup', ''),
            'scale':             round(float(S), 5),
            'U':round(st['U'],5),'V':round(st['V'],5),
            'W':round(st['W'],5),'eta':round(st['eta'],3),
            'fwhm_deg':          round(fwhm_sc, 4),
            'crystallite_size_A':  round(cryst_A, 1) if cryst_A else None,
            'crystallite_size_nm': round(cryst_A/10, 2) if cryst_A else None,
            'weight_fraction_%':   round(wt_frac, 1),
            'n_reflections':       len(st['refs']),
            'tick_positions':      [round(r[0], 3) for r in st['refs']],
            'seeded_by':           st.get('seeded', 'unknown'),
        })

    return {
        'tt':             tt_r.tolist(),
        'y_obs':          y_r.tolist(),
        'y_calc':         yc_f.tolist(),
        'y_background':   bg_f.tolist(),
        'phase_patterns': phase_patterns,
        'residuals':      diff_f.tolist(),
        'statistics':     stats,
        'phase_results':  phase_results,
        'zero_shift':     round(float(zero), 5),
        'n_params':       n_params,
        'wavelength':     wavelength,
        'global_scale':   round(float(S), 5),
        'pymatgen_used':  any(st.get('seeded') == 'pymatgen' for st in phase_state),
    }
