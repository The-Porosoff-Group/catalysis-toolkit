"""
modules/xrd/cod_api.py
Interface to the Crystallography Open Database (COD) REST API.
https://www.crystallography.net/cod/

Live queries — no local database needed.
"""

import requests, math, re
from .crystallography import parse_cif, d_spacing, generate_reflections

COD_BASE     = "https://www.crystallography.net/cod"
COD_SEARCH   = f"{COD_BASE}/result.php"
COD_CIF_URL  = f"{COD_BASE}/cif/{{cod_id}}.cif"
TIMEOUT      = 12  # seconds


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def search_by_elements(elements, max_results=30):
    """
    Search COD for structures containing exactly the given elements.
    elements: list of strings e.g. ['Mo', 'C']
    Returns list of dicts: {cod_id, formula, spacegroup, a, b, c, ...}
    """
    # Build element string for COD query
    elem_str = ' '.join(sorted(elements))
    params = {
        'el1': elements[0] if len(elements) > 0 else '',
        'strictmax': 'on',   # only structures with EXACTLY these elements
        'format': 'json',
        'limit': max_results,
    }
    for i, el in enumerate(elements[1:], 2):
        params[f'el{i}'] = el

    try:
        resp = requests.get(COD_SEARCH, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return _parse_search_results(data)
    except requests.exceptions.ConnectionError:
        return {'error': 'Cannot reach COD — check internet connection.'}
    except requests.exceptions.Timeout:
        return {'error': 'COD query timed out. Try again.'}
    except Exception as e:
        return {'error': f'COD search error: {str(e)}'}


def search_by_name(name, max_results=20):
    """
    Search COD by mineral/compound name.
    """
    params = {
        'text': name,
        'format': 'json',
        'limit': max_results,
    }
    try:
        resp = requests.get(COD_SEARCH, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return _parse_search_results(data)
    except requests.exceptions.ConnectionError:
        return {'error': 'Cannot reach COD — check internet connection.'}
    except Exception as e:
        return {'error': f'COD search error: {str(e)}'}


def _parse_search_results(data):
    """Parse COD JSON search response into clean list of phase candidates."""
    if not isinstance(data, list):
        return []
    results = []
    for entry in data:
        try:
            cod_id = str(entry.get('file', '')).zfill(7)
            results.append({
                'cod_id':      cod_id,
                'formula':     entry.get('formula', '').strip("'\" "),
                'spacegroup':  entry.get('sg', ''),
                'a':           _safe_float(entry.get('a')),
                'b':           _safe_float(entry.get('b')),
                'c':           _safe_float(entry.get('c')),
                'alpha':       _safe_float(entry.get('alpha'), 90.0),
                'beta':        _safe_float(entry.get('beta'),  90.0),
                'gamma':       _safe_float(entry.get('gamma'), 90.0),
                'year':        entry.get('year', ''),
                'authors':     entry.get('authors', ''),
                'journal':     entry.get('journal', ''),
                'mineral':     entry.get('mineral', ''),
            })
        except Exception:
            continue
    return results


def _safe_float(val, default=None):
    if val is None: return default
    try:
        s = str(val).strip()
        m = re.match(r'^([0-9\.\-\+eE]+)', s)
        return float(m.group(1)) if m else default
    except Exception:
        return default


# ─────────────────────────────────────────────────────────────────────────────
# CIF DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def fetch_cif(cod_id):
    """
    Download CIF file for given COD ID.
    Returns (cif_text, parsed_structure_dict) or raises on error.
    """
    url = COD_CIF_URL.format(cod_id=str(cod_id).zfill(7))
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        cif_text = resp.text
        parsed   = parse_cif(cif_text)
        parsed['cod_id']   = str(cod_id)
        parsed['cif_text'] = cif_text
        return parsed
    except requests.exceptions.ConnectionError:
        raise ConnectionError('Cannot reach COD. Check internet connection.')
    except requests.exceptions.Timeout:
        raise TimeoutError('CIF download timed out.')
    except Exception as e:
        raise RuntimeError(f'CIF download error: {e}')


# ─────────────────────────────────────────────────────────────────────────────
# QUICK STICK PATTERN (for preview / phase matching overlay)
# ─────────────────────────────────────────────────────────────────────────────

def get_stick_pattern(structure, wavelength, tt_min=5.0, tt_max=90.0):
    """
    Generate a stick pattern (list of 2θ positions) for a phase candidate.
    Used for quick overlay before full refinement.
    structure: dict from parse_cif() or search result
    Returns list of (two_theta, relative_intensity, hkl_label)
    """
    a     = structure.get('a') or 4.0
    b     = structure.get('b') or a
    c     = structure.get('c') or a
    al    = structure.get('alpha', 90.0)
    be    = structure.get('beta',  90.0)
    ga    = structure.get('gamma', 90.0)
    sys_  = structure.get('system', 'triclinic')
    sg    = structure.get('spacegroup_number', 1)

    try:
        refs = generate_reflections(a, b, c, al, be, ga, sys_, sg,
                                     wavelength, tt_min, tt_max, hkl_max=8)
    except Exception:
        return []

    if not refs:
        return []

    # Normalise multiplicity as rough intensity proxy
    mults = [r[3] for r in refs]
    max_m = max(mults) if mults else 1
    sticks = []
    for tt, d, hkl, m in refs:
        label = f"({hkl[0]}{hkl[1]}{hkl[2]})"
        sticks.append({
            'two_theta': round(tt, 3),
            'd':         round(d,  4),
            'hkl':       label,
            'rel_int':   round(m / max_m, 3),
        })
    return sticks
