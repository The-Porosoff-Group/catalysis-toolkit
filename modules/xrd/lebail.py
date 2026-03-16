"""
modules/xrd/lebail.py
Le Bail profile fitting engine — validated implementation.

Algorithm:
  Outer loop alternates between:
    A) Le Bail I_hkl update  (intensity partitioning from observed data)
    B) Global scale factor   (linear regression)
    C) Background update     (Chebyshev polynomial least squares)
    D) Cell + profile params (scipy least_squares)

Refines: a,b,c cell params (constrained by system), U,V,W,eta (Caglioti/pseudo-Voigt),
         zero shift, Chebyshev background, global scale.
Does NOT refine atomic positions (Le Bail, not Rietveld).
"""

import math
import numpy as np
from scipy.optimize import least_squares
from .crystallography import (
    generate_reflections, chebyshev_background,
    caglioti_fwhm, scherrer_size, pseudo_voigt,
    compute_fit_statistics
)


def _get_profiles(tt_arr, refs, U, V, W, eta, zero=0.0):
    """Return list of unit-normalised pseudo-Voigt profile arrays, one per reflection."""
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


def _cell_free_params(phase):
    """Return (free_values, free_names) for this crystal system."""
    a  = phase['a']
    b  = phase.get('b', a)
    c  = phase.get('c', a)
    be = phase.get('beta', 90.0)
    al = phase.get('alpha', 90.0)
    ga = phase.get('gamma', 90.0)
    sys_ = phase.get('system', 'triclinic').lower()
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
    """Reconstruct (a,b,c,al,be,ga) from free parameters + symmetry constraints."""
    d = {'a': phase['a'], 'b': phase.get('b', phase['a']),
         'c': phase.get('c', phase['a']),
         'alpha': phase.get('alpha', 90.0),
         'beta':  phase.get('beta',  90.0),
         'gamma': phase.get('gamma', 90.0)}
    for name, val in zip(free_names, free_vals):
        d[name] = val
    sys_ = phase.get('system', 'triclinic').lower()
    if sys_ == 'cubic':
        d['b'] = d['a']; d['c'] = d['a']
    elif sys_ in ('hexagonal', 'trigonal', 'tetragonal'):
        d['b'] = d['a']
    return d['a'], d['b'], d['c'], d['alpha'], d['beta'], d['gamma']


def run_lebail(tt, y_obs, sigma, phases, wavelength,
               tt_min=None, tt_max=None,
               n_bg_coeffs=6, max_outer=15,
               progress_callback=None):
    """
    Run Le Bail refinement on multi-phase XRD data.

    Parameters
    ----------
    tt         : np.ndarray — 2θ values (degrees)
    y_obs      : np.ndarray — observed intensities
    sigma      : np.ndarray — uncertainties; uses sqrt(y) if None
    phases     : list of dicts, each with:
                   name, system, spacegroup_number,
                   a, [b, c, alpha, beta, gamma]
                   optional: U_init,V_init,W_init,eta_init
    wavelength : float — X-ray wavelength Å
    tt_min/max : float — refinement range (defaults to data range)
    n_bg_coeffs: int — number of Chebyshev background terms
    max_outer  : int — max outer loop iterations

    Returns
    -------
    dict with tt, y_obs, y_calc, y_background, phase_patterns,
              residuals, statistics, phase_results, zero_shift, wavelength
    """
    if tt_min is None: tt_min = float(tt.min())
    if tt_max is None: tt_max = float(tt.max())

    mask   = (tt >= tt_min) & (tt <= tt_max)
    tt_r   = tt[mask]
    y_r    = y_obs[mask]
    sig_r  = sigma[mask] if sigma is not None else np.sqrt(np.maximum(y_r, 1.0))
    weights = 1.0 / np.maximum(sig_r**2, 1e-6)

    if progress_callback: progress_callback('Generating reflections...')

    # ── Initialise per-phase state ────────────────────────────────────────
    phase_state = []
    for ph in phases:
        free_v, free_n = _cell_free_params(ph)
        refs = generate_reflections(
            *_full_cell(free_v, free_n, ph),
            ph.get('system', 'triclinic'),
            ph.get('spacegroup_number', 1),
            wavelength, tt_min, tt_max, hkl_max=12
        )
        n = len(refs)
        # Init I_hkl from observed peak heights above background
        bg_est = float(np.percentile(y_r, 5))
        I_hkl  = np.zeros(n)
        for k, (tt_p, d, hkl, mult) in enumerate(refs):
            near = np.abs(tt_r - tt_p) < 0.5
            I_hkl[k] = max(y_r[near].max() - bg_est, 1.0) if near.any() else 1.0

        phase_state.append({
            'ph':      ph,
            'refs':    refs,
            'I_hkl':   I_hkl,
            'U':       ph.get('U_init', 0.0),
            'V':       ph.get('V_init', 0.0),
            'W':       ph.get('W_init', 0.37),
            'eta':     ph.get('eta_init', 0.5),
            'free_v':  free_v,
            'free_n':  free_n,
        })

    n_bg  = n_bg_coeffs
    bg_c  = np.zeros(n_bg); bg_c[0] = float(np.percentile(y_r, 5))
    S     = 1.0
    zero  = 0.0

    def total_calc(states, bg_c_, S_, zero_):
        bg_  = np.maximum(chebyshev_background(tt_r, bg_c_, tt_min, tt_max), 0)
        pat_ = np.zeros(len(tt_r))
        for st in states:
            profs = _get_profiles(tt_r, st['refs'], st['U'], st['V'], st['W'], st['eta'], zero_)
            for k in range(len(st['refs'])):
                pat_ += st['I_hkl'][k] * profs[k]
        return S_ * pat_ + bg_, bg_, S_ * pat_

    def rwp_(yo, yc, w):
        d = yo - yc
        return math.sqrt(np.sum(w*d**2) / np.sum(w*yo**2)) * 100

    if progress_callback: progress_callback('Running Le Bail iterations...')

    prev_rwp = 999.0
    for outer in range(max_outer):

        # ── Step A: Le Bail I_hkl update (all phases simultaneously) ─────
        for inner in range(80):
            all_profs  = []
            all_I      = []
            for st in phase_state:
                profs = _get_profiles(tt_r, st['refs'], st['U'],st['V'],st['W'],st['eta'], zero)
                all_profs.append(profs)
                all_I.append(st['I_hkl'])

            bg_cur = np.maximum(chebyshev_background(tt_r, bg_c, tt_min, tt_max), 0)
            pat_all = np.zeros(len(tt_r))
            for i_ph, (profs, I) in enumerate(zip(all_profs, all_I)):
                for k in range(len(I)):
                    pat_all += I[k] * profs[k]
            sum_c = np.maximum(S * pat_all, 1e-8)
            y_nobg = np.maximum(y_r - bg_cur, 0.1)

            max_change = 0.0
            for i_ph, st in enumerate(phase_state):
                I_new = np.zeros(len(st['refs']))
                for k in range(len(st['refs'])):
                    if all_profs[i_ph][k].sum() < 1e-10:
                        I_new[k] = st['I_hkl'][k]; continue
                    ratio  = S * st['I_hkl'][k] * all_profs[i_ph][k] / sum_c
                    numer  = np.sum(weights * y_nobg * ratio)
                    denom  = np.sum(weights * all_profs[i_ph][k]) * S
                    I_new[k] = max(numer / max(denom, 1e-10), 1e-6)
                I_new = np.clip(I_new, 1e-6, 1e6)
                max_change = max(max_change, np.max(np.abs(I_new - st['I_hkl'])))
                st['I_hkl'] = I_new
            if max_change < 1e-5: break

        # ── Step B: Update global scale ───────────────────────────────────
        pat_cur = np.zeros(len(tt_r))
        for st in phase_state:
            profs = _get_profiles(tt_r, st['refs'], st['U'],st['V'],st['W'],st['eta'],zero)
            for k in range(len(st['refs'])):
                pat_cur += st['I_hkl'][k] * profs[k]
        bg_cur = np.maximum(chebyshev_background(tt_r, bg_c, tt_min, tt_max), 0)
        y_nobg2 = y_r - bg_cur
        S_num = np.sum(weights * y_nobg2 * pat_cur)
        S_den = np.sum(weights * pat_cur**2)
        S = max(S_num / max(S_den, 1e-10), 1e-4)

        # ── Step C: Update background ─────────────────────────────────────
        yc_noBG = S * pat_cur

        def resid_bg(x):
            bg_ = np.maximum(chebyshev_background(tt_r, x, tt_min, tt_max), 0)
            return (y_r - yc_noBG - bg_) * np.sqrt(weights)

        rb = least_squares(resid_bg, bg_c, method='trf', max_nfev=300,
                           ftol=1e-10, verbose=0)
        bg_c = rb.x

        # ── Step D: Refine cell + profile per phase ───────────────────────
        if progress_callback and outer == 0:
            progress_callback('Refining cell parameters and peak shape...')

        for i_ph, st in enumerate(phase_state):
            ph = st['ph']
            free_n = st['free_n']
            sys_   = ph.get('system', 'triclinic')

            def resid_cell_prof(x):
                nf = len(free_n)
                free_v_  = x[:nf]
                U_,V_,W_,eta_,zero_ = x[nf:nf+5]
                W_ = max(W_, 0.005); eta_ = np.clip(eta_, 0, 1)
                a_,b_,c_,al_,be_,ga_ = _full_cell(free_v_, free_n, ph)
                refs_l = generate_reflections(a_,b_,c_,al_,be_,ga_,
                    sys_, ph.get('spacegroup_number',1),
                    wavelength, tt_min, tt_max, hkl_max=12)
                n = min(len(refs_l), len(st['I_hkl']))
                profs_l = _get_profiles(tt_r, refs_l, U_,V_,W_,eta_,zero_)
                bg_ = np.maximum(chebyshev_background(tt_r, bg_c, tt_min, tt_max), 0)
                # Other phases
                pat_ = np.zeros(len(tt_r))
                for j, st2 in enumerate(phase_state):
                    if j == i_ph: continue
                    profs2 = _get_profiles(tt_r,st2['refs'],st2['U'],st2['V'],st2['W'],st2['eta'],zero)
                    for k in range(len(st2['refs'])):
                        pat_ += st2['I_hkl'][k]*profs2[k]
                pat_ += sum(st['I_hkl'][k]*profs_l[k] for k in range(n))
                yc = S*pat_ + bg_
                return (y_r - yc)*np.sqrt(weights)

            x0_ = list(st['free_v']) + [st['U'],st['V'],st['W'],st['eta'],zero]
            nf  = len(free_n)
            # Cell bounds ±4%
            lo_ = [v*0.96 for v in st['free_v']] + [0.0,-0.5,0.005,0.0,-0.3]
            hi_ = [v*1.04 for v in st['free_v']] + [2.0, 0.5,2.0, 1.0, 0.3]
            rs = least_squares(resid_cell_prof, x0_, bounds=(lo_,hi_),
                               method='trf', max_nfev=800, ftol=1e-10,xtol=1e-10,verbose=0)
            free_v_new = rs.x[:nf]
            U_n,V_n,W_n,eta_n,zero = rs.x[nf:nf+5]
            W_n = max(W_n,0.005); eta_n=np.clip(eta_n,0,1)

            a_n,b_n,c_n,al_n,be_n,ga_n = _full_cell(free_v_new, free_n, ph)
            refs_new = generate_reflections(a_n,b_n,c_n,al_n,be_n,ga_n,
                sys_, ph.get('spacegroup_number',1),
                wavelength, tt_min, tt_max, hkl_max=12)
            n_new = len(refs_new)
            I_new = np.ones(n_new)
            n_copy = min(n_new, len(st['I_hkl']))
            I_new[:n_copy] = st['I_hkl'][:n_copy]

            st.update({'free_v':free_v_new,'refs':refs_new,'I_hkl':I_new,
                       'U':U_n,'V':V_n,'W':W_n,'eta':eta_n})
            ph.update(dict(zip(free_n, free_v_new)))
            if 'b' not in free_n and sys_ in ('cubic','hexagonal','trigonal','tetragonal'):
                ph['b'] = a_n
            ph['c'] = c_n

        yc_cur,_,_ = total_calc(phase_state, bg_c, S, zero)
        cur_rwp = rwp_(y_r, yc_cur, weights)
        if progress_callback:
            progress_callback(f'Iteration {outer+1}: Rwp = {cur_rwp:.2f}%')
        if abs(prev_rwp - cur_rwp) < 0.005: break
        prev_rwp = cur_rwp

    # ── Final results ─────────────────────────────────────────────────────
    if progress_callback: progress_callback('Computing final statistics...')

    yc_f, bg_f, pat_f_total = total_calc(phase_state, bg_c, S, zero)
    diff_f = y_r - yc_f
    n_params_total = sum(len(st['refs']) + len(st['free_n']) + 5 for st in phase_state) + n_bg + 1
    stats = compute_fit_statistics(y_r, yc_f, weights, n_params_total)

    # Per-phase patterns and results
    phase_patterns = []
    phase_results  = []
    total_scale    = sum(abs(S) for _ in phase_state)

    for i_ph, st in enumerate(phase_state):
        profs = _get_profiles(tt_r, st['refs'], st['U'],st['V'],st['W'],st['eta'],zero)
        pat_ph = S * sum(st['I_hkl'][k]*profs[k] for k in range(len(st['refs'])))
        phase_patterns.append(pat_ph.tolist())

        ph = st['ph']
        a  = ph['a']; b=ph.get('b',a); c=ph.get('c',a)
        al = ph.get('alpha',90.); be=ph.get('beta',90.); ga=ph.get('gamma',90.)

        # Strongest reflection for Scherrer
        strongest_tt = max(st['refs'], key=lambda r: r[3])[0] if st['refs'] else 39.0
        fwhm_sc = caglioti_fwhm(strongest_tt, st['U'],st['V'],st['W'])
        cryst_A  = scherrer_size(fwhm_sc, strongest_tt, wavelength)

        phase_results.append({
            'name':              ph.get('name', f'Phase {i_ph+1}'),
            'a': round(a,5), 'b': round(b,5), 'c': round(c,5),
            'alpha':round(al,3),'beta':round(be,3),'gamma':round(ga,3),
            'system':            ph.get('system','triclinic'),
            'spacegroup_number': ph.get('spacegroup_number',1),
            'scale':             round(S,5),
            'U':round(st['U'],5),'V':round(st['V'],5),
            'W':round(st['W'],5),'eta':round(st['eta'],3),
            'fwhm_deg':          round(fwhm_sc,4),
            'crystallite_size_A':  round(cryst_A,1) if cryst_A else None,
            'crystallite_size_nm': round(cryst_A/10,2) if cryst_A else None,
            'weight_fraction_%': round(100.0/len(phase_state),1),  # equal if single phase
            'n_reflections':     len(st['refs']),
            'tick_positions':    [round(r[0],3) for r in st['refs']],
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
        'zero_shift':     round(zero, 5),
        'n_params':       n_params_total,
        'wavelength':     wavelength,
        'global_scale':   round(S, 5),
    }
