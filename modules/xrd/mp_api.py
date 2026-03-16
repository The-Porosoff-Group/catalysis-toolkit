"""
modules/xrd/mp_api.py
Materials Project database interface.
https://materialsproject.org

Provides ~154k DFT-computed structures — complete coverage of all elements,
carbides, nitrides, oxides, and intermetallics. Complements COD which is
experimental-only and has gaps for common phases (pure metals, metastable carbides).

Requires a free API key from https://materialsproject.org/api
(register → dashboard → copy API key → paste into config.yaml)
"""

import re, math, requests
from .crystallography import parse_cif
from .cod_api import infer_system, _sf

MP_BASE    = "https://api.materialsproject.org"
MP_SUMMARY = f"{MP_BASE}/materials/summary/"
MP_STRUCT  = f"{MP_BASE}/materials/{{mp_id}}/cif"
TIMEOUT    = 15

# Map MP stability descriptor to human-readable label
STABILITY_LABELS = {
    0.0:  "stable",
    0.05: "near-stable",
    0.1:  "metastable",
}


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def search_by_elements(elements, api_key, strict=True, max_results=50,
                        sort_by="formula"):
    """
    Search Materials Project for structures containing the given elements.

    strict=True  → only exact chemical systems (e.g. W-C gives only W/C/WC/W2C etc.)
    strict=False → all structures containing those elements (may include others)
    """
    if not api_key:
        return {'error': 'No Materials Project API key configured. '
                         'Add it to config.yaml.'}

    elements = [e.strip().capitalize() for e in elements if e.strip()]
    if not elements:
        return {'error': 'No elements provided.'}

    headers = {"X-API-KEY": api_key}
    params  = {
        "fields":     "material_id,formula_pretty,symmetry,structure,energy_above_hull,"
                      "theoretical,nelements,elements",
        "deprecated":  "false",
        "_limit":      max_results,
    }

    if strict:
        # Exact chemical system: W-C returns only W,C,WC,W2C,WC2 etc.
        params["chemsys"] = "-".join(sorted(elements))
    else:
        # All structures containing these elements (may have others too)
        params["elements"] = ",".join(elements)

    try:
        resp = requests.get(MP_SUMMARY, headers=headers, params=params,
                             timeout=TIMEOUT)
        if resp.status_code == 403:
            return {'error': 'Materials Project API key invalid or expired. '
                             'Check config.yaml.'}
        resp.raise_for_status()
        data = resp.json()
        return _parse_mp_results(data.get("data", []), sort_by)
    except requests.exceptions.ConnectionError:
        return {'error': 'Cannot reach Materials Project. Check internet connection.'}
    except requests.exceptions.Timeout:
        return {'error': 'Materials Project search timed out.'}
    except Exception as e:
        return {'error': f'Materials Project search error: {e}'}


def search_by_formula(formula, api_key, max_results=30, sort_by="formula"):
    """Search by exact reduced formula e.g. 'W2C', 'WC', 'W'."""
    if not api_key:
        return {'error': 'No Materials Project API key configured.'}

    headers = {"X-API-KEY": api_key}
    params  = {
        "fields":     "material_id,formula_pretty,symmetry,structure,energy_above_hull,"
                      "theoretical,nelements,elements",
        "formula":     formula.strip(),
        "deprecated":  "false",
        "_limit":      max_results,
    }
    try:
        resp = requests.get(MP_SUMMARY, headers=headers, params=params,
                             timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return _parse_mp_results(data.get("data", []), sort_by)
    except requests.exceptions.ConnectionError:
        return {'error': 'Cannot reach Materials Project. Check internet connection.'}
    except Exception as e:
        return {'error': f'Materials Project search error: {e}'}


def search_by_name(name, api_key, max_results=30, sort_by="formula"):
    """
    Search by formula string (MP doesn't have a free-text mineral name search,
    so we try it as a formula first, then as a chemsys if that fails).
    """
    if not api_key:
        return {'error': 'No Materials Project API key configured.'}

    # Try formula search first
    results = search_by_formula(name, api_key, max_results, sort_by)
    if not isinstance(results, dict):
        return results

    # If empty, try parsing as element list
    elements = re.findall(r'[A-Z][a-z]?', name)
    if elements:
        return search_by_elements(elements, api_key, strict=False,
                                   max_results=max_results, sort_by=sort_by)
    return results


def _parse_mp_results(data, sort_by="formula"):
    """Parse MP summary API response into standard phase candidate dicts."""
    results = []
    for entry in data:
        try:
            mp_id   = entry.get("material_id", "")
            formula = entry.get("formula_pretty", "")
            sym     = entry.get("symmetry", {}) or {}
            sg_sym  = sym.get("symbol", "")
            sg_num  = int(sym.get("number", 1) or 1)
            system  = (sym.get("crystal_system", "") or "").lower() or \
                       infer_system(sg_num)
            e_hull  = entry.get("energy_above_hull", 0.0) or 0.0
            theoretical = entry.get("theoretical", True)

            # Extract cell from structure if available
            struct = entry.get("structure") or {}
            lattice = struct.get("lattice") or {}
            a  = _sf(lattice.get("a"))
            b  = _sf(lattice.get("b"))
            c  = _sf(lattice.get("c"))
            al = _sf(lattice.get("alpha"), 90.0)
            be = _sf(lattice.get("beta"),  90.0)
            ga = _sf(lattice.get("gamma"), 90.0)

            # Stability label
            if e_hull < 0.001:
                stability = "stable (on hull)"
            elif e_hull < 0.05:
                stability = f"near-stable (+{e_hull*1000:.0f} meV/atom)"
            elif e_hull < 0.15:
                stability = f"metastable (+{e_hull*1000:.0f} meV/atom)"
            else:
                stability = f"unstable (+{e_hull*1000:.0f} meV/atom)"

            results.append({
                'mp_id':             mp_id,
                'cod_id':            mp_id,   # use mp_id as the identifier
                'formula':           formula,
                'name':              formula,
                'spacegroup':        sg_sym,
                'spacegroup_number': sg_num,
                'system':            system or 'triclinic',
                'a': a, 'b': b or a, 'c': c or a,
                'alpha': al, 'beta': be, 'gamma': ga,
                'stability':         stability,
                'e_above_hull':      round(e_hull, 4),
                'theoretical':       theoretical,
                'year':              'DFT',
                'authors':           'Materials Project',
                'journal':           'Comp.',
                'source':            'mp',
            })
        except Exception:
            continue

    # Sort
    if sort_by == "year_desc":
        results.sort(key=lambda r: r.get('e_above_hull', 99))  # stable first
    elif sort_by == "cell_a":
        results.sort(key=lambda r: r.get('a') or 999)
    elif sort_by == "spacegroup":
        results.sort(key=lambda r: r.get('spacegroup_number', 999))
    else:
        results.sort(key=lambda r: r.get('formula', ''))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CIF DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def fetch_cif(mp_id, api_key):
    """
    Fetch CIF for a Materials Project entry.
    Returns parsed structure dict with 'cif_text' key.
    """
    if not api_key:
        raise ValueError('No Materials Project API key configured.')

    headers = {"X-API-KEY": api_key}
    url     = f"{MP_BASE}/materials/{mp_id}/cif"
    try:
        resp = requests.get(url, headers=headers,
                             params={"fmt": "conventional_standard"},
                             timeout=TIMEOUT)
        if resp.status_code == 403:
            raise PermissionError('API key invalid or expired.')
        resp.raise_for_status()

        # MP returns CIF text directly
        cif_text = resp.text
        parsed   = parse_cif(cif_text)
        parsed['mp_id']    = mp_id
        parsed['cod_id']   = mp_id
        parsed['cif_text'] = cif_text
        parsed['source']   = 'mp'
        return parsed

    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        raise ConnectionError(f'Cannot reach Materials Project: {e}')
    except Exception as e:
        raise RuntimeError(f'CIF download error for {mp_id}: {e}')


# ─────────────────────────────────────────────────────────────────────────────
# API KEY VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_api_key(api_key):
    """Quick check that the API key works. Returns (valid, message)."""
    if not api_key or len(api_key) < 10:
        return False, "API key too short or missing."
    try:
        headers = {"X-API-KEY": api_key}
        params  = {"formula": "W", "fields": "material_id", "_limit": 1}
        resp    = requests.get(MP_SUMMARY, headers=headers, params=params,
                                timeout=8)
        if resp.status_code == 403:
            return False, "API key rejected by Materials Project."
        resp.raise_for_status()
        return True, "API key valid."
    except requests.exceptions.ConnectionError:
        return False, "Cannot reach Materials Project (no internet?)."
    except Exception as e:
        return False, f"Validation error: {e}"
