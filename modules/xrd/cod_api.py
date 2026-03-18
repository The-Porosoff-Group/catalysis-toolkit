"""
modules/xrd/cod_api.py
Interface to the Crystallography Open Database (COD) REST API.
https://wiki.crystallography.net/RESTful_API/

Uses the CSV endpoint which reliably returns full cell parameters,
unlike the JSON endpoint which often has missing values.
"""

import io, csv, re, math, requests, tempfile, os
from .crystallography import parse_cif

COD_SEARCH = "https://www.crystallography.net/cod/result.php"
COD_CIF    = "https://www.crystallography.net/cod/cif/{cod_id}.cif"
TIMEOUT    = 15

SORT_OPTIONS = {
    "formula":      ("Chemical formula", "formula", "asc"),
    "spacegroup":   ("Space group",      "sg",      "asc"),
    "year_desc":    ("Year (newest)",    "year",    "desc"),
    "year_asc":     ("Year (oldest)",    "year",    "asc"),
    "cell_a":       ("Cell param a",     "a",       "asc"),
}


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def search_by_elements(elements, strict=True, max_results=100,
                        sort_by="formula", extra_elements_ok=False):
    """
    Search COD for structures containing the given elements.

    strict=True  → only structures with EXACTLY these elements (no others)
    strict=False → structures containing AT LEAST these elements

    Returns list of dicts, or {'error': str} on failure.
    """
    elements = [e.strip().capitalize() for e in elements if e.strip()]
    if not elements:
        return {'error': 'No elements provided.'}

    params = {'format': 'csv', 'limit': max_results}

    for i, el in enumerate(elements, 1):
        params[f'el{i}'] = el

    if strict:
        params['strictmin'] = len(elements)
        params['strictmax'] = len(elements)

    _apply_sort(params, sort_by)

    try:
        resp = requests.get(COD_SEARCH, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        return _parse_csv(resp.text)
    except requests.exceptions.ConnectionError:
        return {'error': 'Cannot reach COD. Check internet connection.'}
    except requests.exceptions.Timeout:
        return {'error': 'COD search timed out. Try again.'}
    except Exception as e:
        return {'error': f'Search error: {e}'}


def search_by_name(name, max_results=100, sort_by="formula"):
    """Search COD by mineral/compound name or keyword."""
    params = {'format': 'csv', 'limit': max_results, 'text': name.strip()}
    _apply_sort(params, sort_by)
    try:
        resp = requests.get(COD_SEARCH, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        return _parse_csv(resp.text)
    except requests.exceptions.ConnectionError:
        return {'error': 'Cannot reach COD. Check internet connection.'}
    except Exception as e:
        return {'error': f'Search error: {e}'}


def search_by_formula(formula, max_results=100, sort_by="formula"):
    """
    Search COD by exact empirical formula (Hill notation, space-separated).
    e.g. 'C1 Mo2' or 'Mo2C'
    """
    # Normalise: ensure Hill notation with spaces
    formula_hill = _to_hill(formula)
    params = {'format': 'csv', 'limit': max_results, 'formula': formula_hill}
    _apply_sort(params, sort_by)
    try:
        resp = requests.get(COD_SEARCH, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        return _parse_csv(resp.text)
    except requests.exceptions.ConnectionError:
        return {'error': 'Cannot reach COD. Check internet connection.'}
    except Exception as e:
        return {'error': f'Search error: {e}'}


def _apply_sort(params, sort_by):
    if sort_by in SORT_OPTIONS:
        _, field, order = SORT_OPTIONS[sort_by]
        params['order_by'] = field
        params['order']    = order


def _to_hill(formula):
    """Convert formula string like 'Mo2C' to Hill notation 'C1 Mo2'."""
    # Parse element/count pairs
    pairs = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    counts = {}
    for el, n in pairs:
        if el:
            counts[el] = counts.get(el, 0) + (int(n) if n else 1)
    # Hill order: C first, H second, then alphabetical
    ordered = []
    for el in ['C', 'H']:
        if el in counts:
            ordered.append((el, counts.pop(el)))
    for el in sorted(counts):
        ordered.append((el, counts[el]))
    return ' '.join(f'{el}{n}' for el, n in ordered)


# ─────────────────────────────────────────────────────────────────────────────
# CSV PARSER
# ─────────────────────────────────────────────────────────────────────────────

def _parse_csv(text):
    """
    Parse COD CSV response into list of phase candidate dicts.
    COD CSV columns (not all always present):
      file, a, b, c, alpha, beta, gamma, sg, mineral, formula, authors, year,
      journal, volume, pages, doi, ...
    """
    # COD sometimes prepends comment lines starting with '#'
    lines = [l for l in text.splitlines() if not l.startswith('#')]
    if not lines:
        return []

    results = []
    reader = csv.DictReader(io.StringIO('\n'.join(lines)))

    for row in reader:
        cod_id = row.get('file', '').strip().zfill(7)
        if not cod_id or cod_id == '0000000':
            continue

        a  = _sf(row.get('a'))
        b  = _sf(row.get('b'))
        c  = _sf(row.get('c'))
        al = _sf(row.get('alpha'), 90.0)
        be = _sf(row.get('beta'),  90.0)
        ga = _sf(row.get('gamma'), 90.0)

        # Infer system from cell angles
        sg_num = _parse_sg_number(row.get('sg', ''))
        system = infer_system(sg_num, al, be, ga)

        formula = row.get('formula', '').strip().strip("'\"")
        mineral = row.get('mineral', '').strip()
        display_name = mineral if mineral else formula

        results.append({
            'cod_id':            cod_id,
            'formula':           formula,
            'name':              display_name,
            'spacegroup':        row.get('sg', '').strip().strip("'\""),
            'spacegroup_number': sg_num,
            'system':            system,
            'a': a, 'b': b or a, 'c': c or a,
            'alpha': al, 'beta': be, 'gamma': ga,
            'year':    row.get('year', '').strip(),
            'authors': row.get('authors', '').strip()[:60],
            'journal': row.get('journal', '').strip(),
            'doi':     row.get('doi', '').strip(),
            'mineral': mineral,
        })

    return results


def _sf(val, default=None):
    """Safe float parse."""
    if val is None: return default
    try:
        s = str(val).strip().strip("'\"")
        m = re.match(r'^-?[\d\.]+', s)
        return float(m.group()) if m else default
    except Exception:
        return default


def _parse_sg_number(sg_str):
    """Extract space group number from string like 'P 63/m m c' or '194'."""
    sg_str = str(sg_str).strip().strip("'\"")
    # Try parsing as integer directly
    try:
        return int(sg_str)
    except ValueError:
        pass
    # Look for trailing number
    m = re.search(r'(\d+)\s*$', sg_str)
    if m:
        return int(m.group(1))
    return 1


def infer_system(sg_num, al=90, be=90, ga=90):
    """Infer crystal system from space group number."""
    if   1   <= sg_num <= 2:   return 'triclinic'
    elif 3   <= sg_num <= 15:  return 'monoclinic'
    elif 16  <= sg_num <= 74:  return 'orthorhombic'
    elif 75  <= sg_num <= 142: return 'tetragonal'
    elif 143 <= sg_num <= 167: return 'trigonal'
    elif 168 <= sg_num <= 194: return 'hexagonal'
    elif 195 <= sg_num <= 230: return 'cubic'
    # Fallback: infer from angles
    if abs(ga - 120) < 1 and abs(al - 90) < 1 and abs(be - 90) < 1:
        return 'hexagonal'
    if al == be == ga == 90:
        return 'orthorhombic'
    return 'triclinic'


# ─────────────────────────────────────────────────────────────────────────────
# CIF DOWNLOAD AND STRUCTURE BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_cif(cod_id):
    """
    Download CIF for given COD ID.
    Returns dict with all cell params, spacegroup, formula, and 'cif_text'.
    """
    url = COD_CIF.format(cod_id=str(cod_id).zfill(7))
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


def save_cif_temp(cif_text, cod_id='manual'):
    """Save CIF text to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.cif',
        prefix=f'cod_{cod_id}_',
        delete=False
    )
    tmp.write(cif_text)
    tmp.close()
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# STICK PATTERN PREVIEW  (no pymatgen needed — uses our crystallography engine)
# ─────────────────────────────────────────────────────────────────────────────

def get_stick_pattern(structure, wavelength, tt_min=5.0, tt_max=90.0):
    """
    Generate a quick stick pattern for preview overlay.

    If the structure dict contains 'sites' (from CIF parsing) or 'cif_text'
    (from which sites can be parsed), the pattern uses actual structure
    factors — correctly zeroing out structure-factor-extinct reflections.

    Otherwise falls back to multiplicity-only weights.
    """
    from .crystallography import generate_reflections, parse_cif, expand_sites_from_cif
    a  = structure.get('a') or 4.0
    b  = structure.get('b') or a
    c  = structure.get('c') or a
    al = structure.get('alpha', 90.0) or 90.0
    be = structure.get('beta',  90.0) or 90.0
    ga = structure.get('gamma', 90.0) or 90.0
    sys_  = structure.get('system', 'triclinic') or 'triclinic'
    sg    = structure.get('spacegroup_number', 1) or 1

    # Try to get atom sites for structure factor calculation.
    # Use pymatgen expansion for correct F² (asymmetric unit → full cell),
    # then fall back to raw parse_cif.
    sites = structure.get('sites')
    if not sites and structure.get('cif_text'):
        sites = expand_sites_from_cif(structure['cif_text'])
        if not sites:
            try:
                parsed = parse_cif(structure['cif_text'])
                sites = parsed.get('sites')
            except Exception:
                sites = None
    # sites=None is fine — generate_reflections will use multiplicity-only

    try:
        refs = generate_reflections(a, b, c, al, be, ga, sys_, sg,
                                     wavelength, tt_min, tt_max, hkl_max=8,
                                     sites=sites or None)
    except Exception:
        return []

    if not refs:
        return []

    max_w = max((r[3] for r in refs), default=1)
    return [{'two_theta': round(r[0], 3),
             'd':         round(r[1], 4),
             'hkl':       f'({r[2][0]}{r[2][1]}{r[2][2]})',
             'rel_int':   round(r[3] / max_w, 3)} for r in refs]
