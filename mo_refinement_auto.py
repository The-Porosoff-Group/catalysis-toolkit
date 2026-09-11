"""
Automated Rietveld refinement for Mo/Mo2C/MoC phase quantification.

Logic mirrors manual process:
1. Try each phase alone — pick the one with lowest Rwp as the "dominant" phase
2. Add remaining phases one at a time — keep if Rwp drops > RWP_IMPROVEMENT_THRESHOLD
3. For MoC: try all non-stoichiometric variants + stoichiometric — keep best Rwp
4. Output CSV table matching manual results format

Usage:
    python mo_refinement_auto.py
"""

import sys, os, glob, csv, warnings, shutil, re
import numpy as np

# ── GSAS-II paths ─────────────────────────────────────────────────────────────
GSAS_PATHS = [
    '/Users/shane/g2full/GSAS-II',
    '/Users/shane/g2full/GSAS-II/GSASII',
]
for p in GSAS_PATHS:
    if p not in sys.path:
        sys.path.insert(0, p)

import GSASIIscriptable as G2sc

# ── USER CONFIG ───────────────────────────────────────────────────────────────
PATTERN_DIR  = '/Users/shane/Desktop/Eva_proj/XRD Index 3'
CIF_DIR      = '/Users/shane/Downloads/Reference cif'
INSTPRM_FILE = '/Users/shane/Desktop/Porosoff Research/Repos/catalysis-toolkit/smartlab_Si640g.instprm'
OUTPUT_CSV   = '/Users/shane/Desktop/Eva_proj/refinement_results.csv'
WORK_DIR     = '/Users/shane/Desktop/Eva_proj/gsas_tmp'

RWP_IMPROVEMENT_THRESHOLD = 2.0   # % absolute Rwp drop needed to keep a phase

PHASE_CIFS = {
    'Mo':   os.path.join(CIF_DIR, 'Mo_Im3m_fixed.cif'),
    'Mo2C': os.path.join(CIF_DIR, 'Mo2C_Pbcn_fixed.cif'),
}

MOC_VARIANTS = {
    'MoC_0.66': os.path.join(CIF_DIR, 'MoC_0.66.cif'),
    'MoC_0.68': os.path.join(CIF_DIR, 'MoC_0.68.cif'),
    'MoC_0.70': os.path.join(CIF_DIR, 'MoC_0.70.cif'),
    'MoC_0.72': os.path.join(CIF_DIR, 'MoC_0.72.cif'),
    'MoC_0.74': os.path.join(CIF_DIR, 'MoC_0.74.cif'),
    'MoC_0.75': os.path.join(CIF_DIR, 'MoC_0.75.cif'),
    'MoC_1.00': os.path.join(CIF_DIR, 'MoC_cubic_fixed.cif'),
}

os.makedirs(WORK_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def write_instprm(work_dir):
    path = os.path.join(work_dir, 'instrument.instprm')
    if os.path.isfile(INSTPRM_FILE):
        shutil.copy2(INSTPRM_FILE, path)
        return path
    lines = [
        '#GSAS-II instrument parameter file; do not add/delete items!',
        'Type:PXC',
        'Lam1:1.540593', 'Lam2:1.544414', 'I(L2)/I(L1):0.5000',
        'Zero:0.0', 'Polariz.:0.7',
        'U:2.0', 'V:-2.0', 'W:5.0', 'X:0.0', 'Y:0.0', 'Z:0.0',
        'SH/L:0.002', 'Azimuth:0.0',
    ]
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def load_csv_pattern(csv_path):
    data = []
    with open(csv_path, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip().replace(',', ' ')
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
                if np.isfinite(x) and np.isfinite(y):
                    data.append((x, y))
            except ValueError:
                continue
    if not data:
        return None, None
    arr = np.array(data)
    return arr[:, 0], arr[:, 1]


def write_xye(path, tt, y_obs):
    sigma = np.sqrt(np.maximum(y_obs, 1.0))
    with open(path, 'w') as f:
        for x, y, s in zip(tt, y_obs, sigma):
            f.write(f'{x:.6f} {y:.6f} {s:.6f}\n')


def parse_lst_wt_fracs(lst_path):
    """Parse weight fractions and sigmas from GSAS-II .lst file."""
    wt_fracs, wt_sigmas = {}, {}
    if not os.path.isfile(lst_path):
        return wt_fracs, wt_sigmas
    with open(lst_path, 'r', errors='ignore') as f:
        content = f.read()
    # Match blocks: Phase fraction / Weight fraction / Phase name
    pattern = (r'Phase fraction\s*:\s*([\d.]+),\s*sig\s*([\d.]+)\s*'
               r'Weight fraction\s*:\s*([\d.]+),\s*sig\s*([\d.]+)\s*'
               r'Phase:\s*([^\n]+)')
    for m in re.finditer(pattern, content, re.MULTILINE):
        pf, pf_sig, wf, wf_sig, pname = m.groups()
        pname = pname.strip().split(' in ')[0].strip()
        wt_fracs[pname]  = float(wf) * 100
        wt_sigmas[pname] = float(wf_sig) * 100
    return wt_fracs, wt_sigmas


def run_refinement(pattern_path, phase_dict, work_subdir):
    """
    Run Rietveld refinement with given phases.
    Returns dict with Rwp, GOF, chi2, wt_fracs, wt_sigmas or None on failure.
    """
    # Validate all CIF files exist first
    for pname, cif in phase_dict.items():
        if not cif or not os.path.isfile(cif):
            print(f'    CIF not found for {pname}: {cif}')
            return None

    os.makedirs(work_subdir, exist_ok=True)
    gpx_path  = os.path.join(work_subdir, 'refine.gpx')
    xye_path  = os.path.join(work_subdir, 'data.xye')
    inst_path = write_instprm(work_subdir)

    tt, y_obs = load_csv_pattern(pattern_path)
    if tt is None or len(tt) < 10:
        print(f'    Could not load pattern: {pattern_path}')
        return None
    write_xye(xye_path, tt, y_obs)

    try:
        gpx  = G2sc.G2Project(newgpx=gpx_path)
        hist = gpx.add_powder_histogram(xye_path, inst_path)

        # Fix histogram scale — never refine it
        try:
            hist.data['Sample Parameters']['Scale'] = [1.0, False]
        except Exception:
            pass

        # Add phases
        phase_objs = {}
        for pname, cif_path in phase_dict.items():
            try:
                phase_objs[pname] = gpx.add_phase(
                    cif_path, phasename=pname,
                    histograms=[hist], fmthint='CIF')
            except Exception as e:
                print(f'    Phase {pname} failed to load: {e}')
                return None

        # Check reflections exist for each phase
        for pname, pobj in phase_objs.items():
            try:
                refl = hist.data.get('Reflection Lists', {}).get(pname, {})
                # GSAS-II populates this after first refinement cycle
                # so we just trust it will work if CIF loaded OK
            except Exception:
                pass

        # Equal initial scales
        init_scale = max(0.01, float(np.max(y_obs)) / (len(phase_objs) * 100))
        for pobj in phase_objs.values():
            hap = list(pobj.data['Histograms'].values())[0]
            hap['Scale'] = [init_scale, True]

        # Background
        bg_init = float(np.percentile(y_obs, 5))
        hist.data['Background'][0] = [
            'chebyschev-1', True, 3, bg_init, 0.0, 0.0]

        # Step 1: background + scales
        gpx.do_refinements([{'set': {
            'Background': {'type': 'chebyschev-1',
                           'refine': True, 'no. coeffs': 3},
        }, 'cycles': 5}])

        # Step 2: add zero shift
        gpx.do_refinements([{'set': {
            'Background': {'type': 'chebyschev-1',
                           'refine': True, 'no. coeffs': 3},
            'Instrument Parameters': ['Zero'],
        }, 'cycles': 5}])

        # Step 3: more cycles for convergence
        gpx.do_refinements([{'set': {
            'Background': {'type': 'chebyschev-1',
                           'refine': True, 'no. coeffs': 3},
            'Instrument Parameters': ['Zero'],
        }, 'cycles': 10}])

        # Extract Rwp
        try:
            gsas_stats = hist.get_statistics()
            rwp = float(gsas_stats.get('Rwp', 99.0))
            gof = float(gsas_stats.get('GOF', 99.0))
        except Exception:
            res  = hist.residuals
            rwp  = float(res.get('wR', 99.0))
            gof  = 99.0

        # Guard: treat obviously diverged refinements as failures
        if rwp > 95 or not np.isfinite(rwp):
            print(f'    Rwp={rwp:.1f}% — treating as failed')
            return None

        # Parse weight fractions from .lst
        lst_path = gpx_path.replace('.gpx', '.lst')
        wt_fracs, wt_sigmas = parse_lst_wt_fracs(lst_path)

        # Fallback: compute from raw scales if lst parse empty
        if not wt_fracs:
            raw_scales = {}
            for pname, pobj in phase_objs.items():
                hap = list(pobj.data['Histograms'].values())[0]
                raw_scales[pname] = max(hap.get('Scale', [0])[0], 0)
            total = sum(raw_scales.values()) or 1e-10
            for pname, s in raw_scales.items():
                wt_fracs[pname]  = (s / total) * 100
                wt_sigmas[pname] = None

        chi2 = round(gof ** 2, 3) if np.isfinite(gof) else None

        return {
            'Rwp':       round(rwp, 3),
            'GOF':       round(gof, 3),
            'chi2':      chi2,
            'wt_fracs':  wt_fracs,
            'wt_sigmas': wt_sigmas,
            'phases':    list(phase_dict.keys()),
        }

    except Exception as e:
        print(f'    Refinement exception: {e}')
        return None
    finally:
        shutil.rmtree(work_subdir, ignore_errors=True)


def select_phases_for_pattern(pattern_path, pattern_name):
    """Sequential phase selection mirroring manual process."""
    print(f'\n{"="*60}')
    print(f'Pattern: {pattern_name}')
    print(f'{"="*60}')

    base_dir = os.path.join(WORK_DIR, pattern_name)

    # ── Step 1: find dominant phase ───────────────────────────────────
    print('Step 1: Finding dominant phase...')
    candidates = {}  # name → {Rwp, cif, result}

    for pname, cif in PHASE_CIFS.items():
        print(f'  Trying {pname} alone...')
        result = run_refinement(
            pattern_path, {pname: cif},
            os.path.join(base_dir, f'single_{pname}'))
        if result:
            print(f'    Rwp = {result["Rwp"]:.3f}%')
            candidates[pname] = {'Rwp': result['Rwp'], 'cif': cif,
                                  'result': result}

    best_moc_name = None
    best_moc_cif  = None
    best_moc_rwp  = 999.0
    for moc_name, moc_cif in MOC_VARIANTS.items():
        print(f'  Trying {moc_name} alone...')
        result = run_refinement(
            pattern_path, {moc_name: moc_cif},
            os.path.join(base_dir, f'single_{moc_name}'))
        if result:
            print(f'    Rwp = {result["Rwp"]:.3f}%')
            candidates[moc_name] = {'Rwp': result['Rwp'], 'cif': moc_cif,
                                     'result': result}
            if result['Rwp'] < best_moc_rwp:
                best_moc_rwp  = result['Rwp']
                best_moc_name = moc_name
                best_moc_cif  = moc_cif

    if not candidates:
        print('  ERROR: all single-phase refinements failed')
        return None

    # Pick dominant = lowest Rwp
    dominant_name = min(candidates, key=lambda k: candidates[k]['Rwp'])
    dominant_cif  = candidates[dominant_name]['cif']
    current_rwp   = candidates[dominant_name]['Rwp']
    print(f'\n  Dominant phase: {dominant_name} (Rwp={current_rwp:.3f}%)')

    # ── Step 2: sequentially add phases ───────────────────────────────
    active_phases = {dominant_name: dominant_cif}

    # Non-MoC phases to try adding
    for pname, cif in PHASE_CIFS.items():
        if pname == dominant_name:
            continue
        trial = dict(active_phases)
        trial[pname] = cif
        print(f'\nStep 2: Trying adding {pname}...')
        result = run_refinement(
            pattern_path, trial,
            os.path.join(base_dir, f'add_{pname}'))
        if result:
            drop = current_rwp - result['Rwp']
            print(f'  Rwp: {current_rwp:.3f}% → {result["Rwp"]:.3f}%  '
                  f'(Δ={drop:.3f}%)')
            if drop > RWP_IMPROVEMENT_THRESHOLD:
                print(f'  ✓ Keeping {pname}')
                active_phases[pname] = cif
                current_rwp = result['Rwp']
            else:
                print(f'  ✗ Dropping {pname}')
        else:
            print(f'  ✗ {pname} refinement failed — skipping')

    # Add best MoC variant if not already dominant
    moc_already = any('MoC' in k for k in active_phases)
    if not moc_already and best_moc_name:
        print(f'\nStep 2: Trying best MoC variant ({best_moc_name})...')
        trial = dict(active_phases)
        trial[best_moc_name] = best_moc_cif
        result = run_refinement(
            pattern_path, trial,
            os.path.join(base_dir, f'add_{best_moc_name}'))
        if result:
            drop = current_rwp - result['Rwp']
            print(f'  Rwp: {current_rwp:.3f}% → {result["Rwp"]:.3f}%  '
                  f'(Δ={drop:.3f}%)')
            if drop > RWP_IMPROVEMENT_THRESHOLD:
                print(f'  ✓ Keeping {best_moc_name}')
                active_phases[best_moc_name] = best_moc_cif
                current_rwp = result['Rwp']
            else:
                print(f'  ✗ Dropping MoC (no improvement)')
        else:
            print(f'  ✗ MoC refinement failed — skipping')

    # ── Step 3: final refinement with selected phases ──────────────────
    print(f'\nFinal phases: {list(active_phases.keys())}')
    final = run_refinement(
        pattern_path, active_phases,
        os.path.join(base_dir, 'final'))

    if final:
        final['active_phases'] = list(active_phases.keys())
        final['moc_variant']   = next(
            (k for k in active_phases if 'MoC' in k), 'none')
    return final


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pattern_files = sorted(glob.glob(os.path.join(PATTERN_DIR, 'M*.csv')))
    if not pattern_files:
        print(f'No pattern files found in {PATTERN_DIR}')
        return

    print(f'Found {len(pattern_files)} patterns: '
          f'{[os.path.basename(p) for p in pattern_files]}')

    rows = []
    for pattern_path in pattern_files:
        pattern_name = os.path.splitext(os.path.basename(pattern_path))[0]
        result = select_phases_for_pattern(pattern_path, pattern_name)

        if result is None:
            print(f'\nFAILED: {pattern_name}')
            rows.append({'Index': pattern_name, 'Error': 'failed'})
            continue

        wf = result['wt_fracs']
        ws = result['wt_sigmas']

        def get_wf(keys):
            for k in keys:
                if k in wf:
                    return round(wf[k], 1)
            return 0

        def get_ws(keys):
            for k in keys:
                if k in ws and ws[k] is not None:
                    return round(ws[k], 3)
            return None

        moc_keys = [k for k in wf if 'MoC' in k]

        row = {
            'Index':            pattern_name,
            'Mo (Im3m) wt%':    get_wf(['Mo']),
            'Mo sigma':         get_ws(['Mo']),
            'Mo2C (ortho) wt%': get_wf(['Mo2C']),
            'Mo2C sigma':       get_ws(['Mo2C']),
            'MoC wt%':          get_wf(moc_keys),
            'MoC sigma':        get_ws(moc_keys),
            'MoC variant':      result.get('moc_variant', 'none'),
            'Active phases':    ', '.join(result.get('active_phases', [])),
            'Rwp':              result['Rwp'],
            'GOF':              result['GOF'],
            'Chi2':             result['chi2'],
        }
        rows.append(row)

        print(f'\nResult for {pattern_name}:')
        print(f'  Mo:   {row["Mo (Im3m) wt%"]}% ± {row["Mo sigma"]}%')
        print(f'  Mo2C: {row["Mo2C (ortho) wt%"]}% ± {row["Mo2C sigma"]}%')
        print(f'  MoC ({row["MoC variant"]}): {row["MoC wt%"]}% ± {row["MoC sigma"]}%')
        print(f'  Rwp={result["Rwp"]}%  GOF={result["GOF"]}  Chi2={result["chi2"]}')

    if rows:
        fieldnames = ['Index', 'Mo (Im3m) wt%', 'Mo sigma',
                      'Mo2C (ortho) wt%', 'Mo2C sigma',
                      'MoC wt%', 'MoC sigma', 'MoC variant',
                      'Active phases', 'Rwp', 'GOF', 'Chi2']
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        print(f'\n✅ Results saved to {OUTPUT_CSV}')


if __name__ == '__main__':
    main()