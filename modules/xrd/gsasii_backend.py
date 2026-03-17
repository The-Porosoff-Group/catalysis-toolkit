"""
modules/xrd/gsasii_backend.py
GSAS-II integration for Rietveld/Le Bail refinement via GSASIIscriptable.

Requires GSAS-II installed in the Python environment (conda install gsas2full -c briantoby).
This module wraps GSASIIscriptable to provide a refinement backend compatible
with the toolkit's result format (same keys as run_lebail / run_rietveld).
"""

import math, os, tempfile, warnings
import numpy as np

# ── GSAS-II availability check ──────────────────────────────────────────────

_GSASII_AVAILABLE = False
_GSASII_IMPORT_ERROR = None

try:
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
    """Return True if GSASIIscriptable is importable."""
    return _GSASII_AVAILABLE


def import_error():
    """Return the import error message, or None if available."""
    return _GSASII_IMPORT_ERROR


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _write_xye(path, tt, y_obs, sigma):
    """Write a .xye file (2theta  intensity  sigma) for GSAS-II import."""
    with open(path, 'w') as f:
        for i in range(len(tt)):
            f.write(f"{tt[i]:.6f}  {y_obs[i]:.4f}  {sigma[i]:.4f}\n")


def _write_temp_cif(cif_text, phase_name='phase'):
    """Write CIF text to a temporary file. Returns path."""
    fd, path = tempfile.mkstemp(suffix='.cif', prefix=f'gsas_{phase_name}_')
    with os.fdopen(fd, 'w') as f:
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
            f"Install with: conda install gsas2full -c briantoby\n"
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
    work_dir = tempfile.mkdtemp(prefix='gsas2_')
    gpx_path = os.path.join(work_dir, 'refine.gpx')
    data_path = os.path.join(work_dir, 'data.xye')
    _write_xye(data_path, tt_r, y_r, sig_r)

    cif_paths = []
    for ph in phases:
        cif_path = _write_temp_cif(ph['cif_text'], ph.get('name', 'phase'))
        cif_paths.append(cif_path)

    try:
        # ── Build GSAS-II project ────────────────────────────────────────
        gpx = G2sc.G2Project(newgpx=gpx_path)

        # Add histogram (powder data)
        histogram = gpx.add_powder_histogram(
            data_path, 'dummy',  # iparams handled below
            databank=None,
        )

        # Set instrument parameters manually
        inst_params = histogram.data['Instrument Parameters'][0]
        inst_params['Type'] = ['PXC', 'PXC']  # Powder X-ray, CW
        inst_params['Lam'] = [wavelength, wavelength]
        inst_params['Zero'] = [0.0, 0.0]

        # Initial profile parameters (will be refined)
        inst_params['U'] = [0.01, 0.01]
        inst_params['V'] = [-0.01, -0.01]
        inst_params['W'] = [0.15, 0.15]
        inst_params['X'] = [0.0, 0.0]
        inst_params['Y'] = [0.1, 0.1]
        inst_params['SH/L'] = [0.002, 0.002]  # asymmetry

        # Set data range
        histogram.data['Limits'] = [[tt_min, tt_max], [tt_min, tt_max]]

        # Set background
        bkg_data = histogram.data['Background']
        bkg_data[0] = ['chebyschev-1', True, n_bg_coeffs,
                        1.0] + [0.0] * (n_bg_coeffs - 1)

        # Add phases from CIF files
        gsas_phases = []
        for i, (ph, cif_path) in enumerate(zip(phases, cif_paths)):
            phase_obj = gpx.add_phase(
                cif_path,
                phasename=ph.get('name', f'Phase_{i+1}'),
                histograms=[histogram],
            )
            gsas_phases.append(phase_obj)

        if progress_callback:
            progress_callback('GSAS-II: stage 1 — refining background + scale...')

        # ── Stage 1: Background + scale ──────────────────────────────────
        for phase_obj in gsas_phases:
            hapData = list(phase_obj.data['Histograms'].values())[0]
            hapData['Scale'] = [1.0, True]  # refine scale

        gpx.set_refinement({
            'set': {
                'Background': {'type': 'chebyschev-1', 'refine': True,
                                'no. coeffs': n_bg_coeffs},
            }
        })
        gpx.do_refinements([{'set': {}, 'cycles': min(max_cycles, 5)}])

        if progress_callback:
            progress_callback('GSAS-II: stage 2 — refining profile parameters...')

        # ── Stage 2: Profile parameters ──────────────────────────────────
        gpx.set_refinement({
            'set': {
                'Instrument Parameters': ['U', 'V', 'W', 'X', 'Y'],
            }
        })
        gpx.do_refinements([{'set': {}, 'cycles': min(max_cycles, 5)}])

        if progress_callback:
            progress_callback('GSAS-II: stage 3 — refining cell + atoms...')

        # ── Stage 3: Cell parameters + atomic displacement ───────────────
        for phase_obj in gsas_phases:
            phase_obj.set_refinements({
                'Cell': True,
                'Atoms': {'U': True},  # isotropic displacement
            })

        gpx.do_refinements([{'set': {}, 'cycles': max_cycles}])

        if progress_callback:
            progress_callback('GSAS-II: extracting results...')

        # ── Extract results ──────────────────────────────────────────────
        # Get calculated pattern
        y_calc_full = np.array(histogram.getdata('ycalc'))
        y_obs_full = np.array(histogram.getdata('yobs'))
        y_bg_full = np.array(histogram.getdata('yback'))
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

        # Get weight fractions from GSAS-II
        wt_fracs = {}
        try:
            # GSAS-II stores weight fractions in the histogram phase data
            for phase_obj in gsas_phases:
                hapData = list(phase_obj.data['Histograms'].values())[0]
                wt_fracs[phase_obj.name] = hapData.get('Scale', [1.0])[0]
        except Exception:
            pass

        # Normalise scale factors to weight fractions
        total_scale = sum(wt_fracs.values()) or 1.0

        for i, (ph, phase_obj) in enumerate(zip(phases, gsas_phases)):
            # Cell parameters
            cell = phase_obj.get_cell()
            a, b, c = cell['a'], cell['b'], cell['c']
            alpha, beta, gamma = cell['alpha'], cell['beta'], cell['gamma']
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

            # Weight fraction
            scale_val = wt_fracs.get(phase_obj.name, 1.0)
            wt_pct = (scale_val / total_scale) * 100 if total_scale > 0 else 0

            # Tick positions (reflection list)
            tick_positions = []
            try:
                refList = list(phase_obj.data['Histograms'].values())[0].get(
                    'Reflection Lists', {})
                if refList:
                    first_key = list(refList.keys())[0]
                    for ref in refList[first_key].get('RefList', []):
                        tt_ref = ref[5]  # 2theta position
                        if tt_min <= tt_ref <= tt_max:
                            tick_positions.append(round(tt_ref, 3))
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
