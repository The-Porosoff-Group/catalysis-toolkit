"""
app.py  —  Catalysis Data Toolkit
Local web server. Run with:  python app.py
Then open:  http://localhost:5000
"""

import os, sys, re, json, base64, webbrowser
from datetime import datetime
from threading import Timer

import yaml
from flask import Flask, render_template, request, jsonify, send_file

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.join(BASE_DIR, 'modules')
UPLOAD_DIR  = os.path.join(BASE_DIR, 'uploads')
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')
os.makedirs(UPLOAD_DIR, exist_ok=True)
sys.path.insert(0, MODULES_DIR)

# ── Load config ───────────────────────────────────────────────────────────────
def load_config():
    defaults = {
        'materials_project': {'api_key': ''},
        'cache': {'directory': '~/.catalysis_toolkit_cache', 'max_size_mb': 500},
        'performance': {'max_outer_iterations': 10, 'preload_pymatgen': True},
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                user = yaml.safe_load(f) or {}
            # Deep merge
            for section, vals in user.items():
                if isinstance(vals, dict):
                    defaults.setdefault(section, {}).update(vals)
                else:
                    defaults[section] = vals
        except Exception as e:
            print(f"  Warning: could not read config.yaml: {e}")
    return defaults

CONFIG = load_config()
MP_API_KEY   = CONFIG['materials_project'].get('api_key', '').strip()
CACHE_DIR    = CONFIG['cache'].get('directory', '~/.catalysis_toolkit_cache')
CACHE_MAX_MB = CONFIG['cache'].get('max_size_mb', 500)
MAX_OUTER    = CONFIG['performance'].get('max_outer_iterations', 10)

# ── Import modules ────────────────────────────────────────────────────────────
import gc_processor
import tga_processor
import bet_processor
import xrd_processor
from xrd.cif_cache import get_cache, cached_fetch_cod, cached_fetch_mp
from xrd.mp_api    import (search_by_elements  as mp_search_elements,
                            search_by_formula   as mp_search_formula,
                            search_by_name      as mp_search_name,
                            fetch_cif           as mp_fetch_cif,
                            validate_api_key    as mp_validate_key)
from xrd.cod_api   import (search_by_elements  as cod_search_elements,
                            search_by_formula   as cod_search_formula,
                            search_by_name      as cod_search_name,
                            fetch_cif           as cod_fetch_cif,
                            get_stick_pattern,  SORT_OPTIONS)

# Initialise cache with config settings
_cache = get_cache(cache_dir=CACHE_DIR, max_size_mb=CACHE_MAX_MB)

# ── Preload pymatgen ──────────────────────────────────────────────────────────
_pymatgen_ready = False
if CONFIG['performance'].get('preload_pymatgen', True):
    try:
        print("  Pre-loading pymatgen...", end=' ', flush=True)
        from pymatgen.core import Structure
        from pymatgen.analysis.diffraction.xrd import XRDCalculator
        _dummy = XRDCalculator(wavelength=1.54056)
        _pymatgen_ready = True
        print("done.")
    except ImportError:
        print("not installed (run: pip install pymatgen)")
    except Exception as e:
        print(f"failed ({e})")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

MODULES = [
    {'id': 'gc',  'name': 'GC Analysis',       'icon': '⚗️',  'status': 'active'},
    {'id': 'tga', 'name': tga_processor.MODULE_INFO['name'],
                  'icon': tga_processor.MODULE_INFO['icon'],
                  'status': tga_processor.MODULE_INFO['status']},
    {'id': 'bet', 'name': bet_processor.MODULE_INFO['name'],
                  'icon': bet_processor.MODULE_INFO['icon'],
                  'status': bet_processor.MODULE_INFO['status']},
    {'id': 'xrd', 'name': xrd_processor.MODULE_INFO['name'],
                  'icon': xrd_processor.MODULE_INFO['icon'],
                  'status': 'active'},
]

# ── Routes — General ──────────────────────────────────────────────────────────

@app.route('/')
def index():
    reaction_configs = gc_processor.list_reaction_configs(
        os.path.join(MODULES_DIR, 'reaction_configs'))
    return render_template('index.html',
                           modules=MODULES,
                           reaction_configs=reaction_configs,
                           mp_key_set=bool(MP_API_KEY),
                           pymatgen_ready=_pymatgen_ready)

@app.route('/api/reaction_configs')
def get_reaction_configs():
    return jsonify(gc_processor.list_reaction_configs(
        os.path.join(MODULES_DIR, 'reaction_configs')))

@app.route('/api/status')
def api_status():
    cache_stats = _cache.stats()
    return jsonify({
        'pymatgen_ready': _pymatgen_ready,
        'mp_key_set':     bool(MP_API_KEY),
        'cache':          cache_stats,
    })

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    _cache.clear()
    return jsonify({'ok': True, 'message': 'Cache cleared.'})

# ── Routes — GC ───────────────────────────────────────────────────────────────

@app.route('/api/process_gc', methods=['POST'])
def process_gc():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded.'}), 400
        f = request.files['file']
        if not f.filename.endswith('.xlsx'):
            return jsonify({'error': 'Please upload an .xlsx file.'}), 400
        safe_name   = re.sub(r'[^\w\-.]', '_', f.filename)
        upload_path = os.path.join(UPLOAD_DIR, safe_name)
        f.save(upload_path)
        form        = request.form
        config_file = form.get('reaction_config')
        config_path = os.path.join(MODULES_DIR, 'reaction_configs', config_file)
        if not os.path.exists(config_path):
            return jsonify({'error': f'Reaction config not found: {config_file}'}), 400
        reaction_config = gc_processor.load_reaction_config(config_path)
        metadata = {
            'catalyst_id': form.get('catalyst_id', 'Unknown'),
            'reactant':    reaction_config['reactant'],
            'temperature': form.get('temperature', ''),
            'pressure':    form.get('pressure', ''),
            'ghsv':        form.get('ghsv', ''),
            'notes':       form.get('notes', ''),
            'source_file': f.filename,
            'reaction':    reaction_config['name'],
        }
        inlet_flows = {}
        try:
            inlet_flows = {k: float(v)
                           for k, v in json.loads(form.get('inlet_flows', '{}')).items()
                           if v}
        except (json.JSONDecodeError, ValueError):
            pass
        ss_start = int(form.get('ss_start', 1))
        ss_end   = int(form.get('ss_end',   999))
        output_base = form.get('output_dir', '').strip()
        if not output_base or not os.path.isdir(output_base):
            output_base = os.path.join(BASE_DIR, 'results')
        ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_id = re.sub(r'[^\w\-]', '_', metadata['catalyst_id'])
        output_dir = os.path.join(output_base, f'{safe_id}_{ts}')
        result = gc_processor.run(
            filepath=upload_path, output_dir=output_dir,
            reaction_config=reaction_config, metadata=metadata,
            inlet_flows=inlet_flows, ss_start=ss_start, ss_end=ss_end)
        with open(result['plot_path'], 'rb') as img:
            plot_b64 = base64.b64encode(img.read()).decode('utf-8')
        clean = {k: v for k, v in result.items() if k != 'plot_path'}
        clean['plot_b64'] = plot_b64
        return jsonify(clean)
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/api/download')
def download_file():
    path = request.args.get('path', '')
    if os.path.isfile(path):
        return send_file(path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

@app.route('/api/open_folder')
def open_folder():
    path = request.args.get('path', '')
    if os.path.isdir(path):
        os.startfile(path)
        return jsonify({'ok': True})
    return jsonify({'error': 'Folder not found'}), 404

# ── Routes — XRD search ───────────────────────────────────────────────────────

@app.route('/api/xrd/search', methods=['POST'])
def xrd_search():
    try:
        data       = request.get_json()
        elements   = [e.strip() for e in data.get('elements', []) if e.strip()]
        name       = data.get('name', '').strip()
        formula    = data.get('formula', '').strip()
        wavelength = float(data.get('wavelength', 1.54056))
        sort_by    = data.get('sort_by', 'formula')
        strict     = bool(data.get('strict', True))
        limit      = min(int(data.get('limit', 100)), 200)
        source     = data.get('source', 'cod')  # 'cod' or 'mp' or 'both'

        cod_results = []
        mp_results  = []

        # ── COD search ────────────────────────────────────────────────────────
        if source in ('cod', 'both'):
            if formula:
                r = cod_search_formula(formula, max_results=limit, sort_by=sort_by)
            elif name:
                r = cod_search_name(name, max_results=limit, sort_by=sort_by)
            elif elements:
                r = cod_search_elements(elements, strict=strict,
                                         max_results=limit, sort_by=sort_by)
            else:
                r = {'error': 'Provide elements, formula, or name.'}
            if isinstance(r, list):
                for entry in r:
                    entry['source'] = 'cod'
                cod_results = r
            elif isinstance(r, dict) and 'error' in r and source == 'cod':
                return jsonify(r), 500

        # ── Materials Project search ──────────────────────────────────────────
        if source in ('mp', 'both') and MP_API_KEY:
            if formula:
                r = mp_search_formula(formula, MP_API_KEY,
                                       max_results=limit, sort_by=sort_by)
            elif elements:
                r = mp_search_elements(elements, MP_API_KEY, strict=strict,
                                        max_results=limit, sort_by=sort_by)
            elif name:
                # MP has no free-text search — use our smart name→elements fallback
                r = mp_search_name(name, MP_API_KEY,
                                    max_results=limit, sort_by=sort_by)
            else:
                r = []
            if isinstance(r, list):
                mp_results = r
            elif isinstance(r, dict) and 'error' in r and source == 'mp':
                return jsonify(r), 500

        # Combine: MP first (computed, more complete), then COD
        combined = mp_results + cod_results

        # Add stick patterns (first 20 only — performance)
        for entry in combined[:20]:
            try:
                entry['stick_pattern'] = get_stick_pattern(entry, wavelength)
            except Exception:
                entry['stick_pattern'] = []

        # Cache any CIF text that came back inline with MP results
        for entry in combined:
            cif_text = entry.pop('_cif_text', '')
            if cif_text and '_cell_length_a' in cif_text:
                mp_id     = entry.get('mp_id', entry.get('cod_id', ''))
                cache_key = f"mp:{mp_id}"
                if mp_id and not _cache.has(cache_key):
                    _cache.put(cache_key, cif_text)

        return jsonify({
            'results':     combined,
            'total':       len(combined),
            'cod_count':   len(cod_results),
            'mp_count':    len(mp_results),
            'mp_available': bool(MP_API_KEY),
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/xrd/fetch_cif', methods=['POST'])
def xrd_fetch_cif():
    """Fetch and cache a CIF from COD or Materials Project."""
    try:
        data       = request.get_json()
        cod_id     = str(data.get('cod_id', ''))
        source     = data.get('source', 'cod')
        wavelength = float(data.get('wavelength', 1.54056))

        if source == 'mp' or cod_id.startswith('mp-'):
            result = cached_fetch_mp(cod_id, MP_API_KEY, mp_fetch_cif)
        else:
            result = cached_fetch_cod(cod_id, cod_fetch_cif)

        # Store CIF text server-side; don't send over wire
        cif_text  = result.pop('cif_text', '')
        cache_key = f"{'mp' if source=='mp' else 'cod'}:{cod_id}"
        _cache.put(cache_key, cif_text)

        # Compute accurate stick pattern server-side using crystallography engine
        # (applies proper extinction rules — much more accurate than client-side)
        try:
            sticks = get_stick_pattern(result, wavelength, tt_min=5.0, tt_max=100.0)
            result['stick_pattern'] = sticks
        except Exception:
            result['stick_pattern'] = []

        result['cached'] = result.get('cached', False)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/xrd/mp_debug_cif', methods=['GET'])
def mp_debug_cif():
    """Show structure data from summary endpoint for a given mp-id."""
    import requests as req
    if not MP_API_KEY:
        return jsonify({'error': 'No MP API key configured'})
    try:
        mp_id   = request.args.get('id', 'mp-91')
        headers = {'X-API-KEY': MP_API_KEY, 'Accept': 'application/json'}
        resp    = req.get(
            'https://api.materialsproject.org/materials/summary/',
            headers=headers,
            params={'material_ids': mp_id, '_fields': 'material_id,formula_pretty,structure',
                    'deprecated': 'false', '_limit': 1},
            timeout=15,
        )
        data = resp.json().get('data', [])
        entry = data[0] if data else {}
        struct = entry.get('structure', {})
        lattice = struct.get('lattice', {}) if struct else {}
        return jsonify({
            'status':          resp.status_code,
            'formula':         entry.get('formula_pretty'),
            'has_structure':   bool(struct),
            'lattice':         lattice,
            'n_sites':         len(struct.get('sites', [])) if struct else 0,
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/xrd/mp_debug', methods=['GET'])
def mp_debug():
    """Dump raw MP API response for debugging field names."""
    import requests as req
    if not MP_API_KEY:
        return jsonify({'error': 'No MP API key configured'})
    try:
        headers = {'X-API-KEY': MP_API_KEY, 'Accept': 'application/json'}
        resp = req.get('https://api.materialsproject.org/materials/summary/',
                        headers=headers,
                        params={'chemsys': 'W', 'deprecated': 'false', '_limit': 1,
                                '_fields': 'material_id,formula_pretty,symmetry,'
                                           'energy_above_hull,theoretical,structure'},
                        timeout=15)
        return jsonify({
            'status':  resp.status_code,
            'raw':     resp.json(),
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/xrd/validate_mp_key', methods=['POST'])
def validate_mp_key():
    data = request.get_json() or {}
    key  = data.get('api_key', MP_API_KEY).strip()
    valid, msg = mp_validate_key(key)
    return jsonify({'valid': valid, 'message': msg})


@app.route('/api/process_xrd', methods=['POST'])
def process_xrd():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        f = request.files['file']
        safe_name   = re.sub(r'[^\w\-.]', '_', f.filename)
        upload_path = os.path.join(UPLOAD_DIR, safe_name)
        f.save(upload_path)

        form       = request.form
        wavelength = float(form.get('wavelength', 1.54056))
        tt_min     = float(form.get('tt_min', 5.0))
        tt_max     = float(form.get('tt_max', 90.0))
        sample_id  = form.get('sample_id', 'Sample')
        notes      = form.get('notes', '')
        wl_label   = form.get('wavelength_label', f'λ={wavelength:.5f} Å')

        phases = json.loads(form.get('phases', '[]'))
        if not phases:
            return jsonify({'error': 'No phases selected for refinement.'}), 400

        # Re-attach cached CIF text for pymatgen seeding
        for ph in phases:
            source = ph.get('source', 'cod')
            cid    = str(ph.get('cod_id', ph.get('mp_id', '')))
            cache_key = f"{source}:{cid}"
            text = _cache.get(cache_key)
            if not text:
                # Try alternate key format
                text = _cache.get(f"cod:{cid}") or _cache.get(f"mp:{cid}")
            if text:
                ph['cif_text'] = text

        output_base = form.get('output_dir', '').strip()
        if not output_base or not os.path.isdir(output_base):
            output_base = os.path.join(BASE_DIR, 'results')
        ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_id = re.sub(r'[^\w\-]', '_', sample_id)
        out_dir = os.path.join(output_base, f'XRD_{safe_id}_{ts}')

        result = xrd_processor.run(
            filepath   = upload_path,
            output_dir = out_dir,
            metadata   = {'sample_id': sample_id, 'notes': notes},
            params     = {
                'phases':           phases,
                'wavelength':       wavelength,
                'wavelength_label': wl_label,
                'tt_min':           tt_min,
                'tt_max':           tt_max,
                'n_bg_coeffs':      6,
                'max_outer':        MAX_OUTER,
            }
        )

        with open(result['plot_path'], 'rb') as img:
            plot_b64 = base64.b64encode(img.read()).decode()

        return jsonify({
            'plot_b64':      plot_b64,
            'statistics':    result['statistics'],
            'phase_results': result['phase_results'],
            'zero_shift':    result['zero_shift'],
            'pymatgen_used': result.get('pymatgen_used', False),
            'summary_path':  result['summary_path'],
            'output_dir':    out_dir,
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

# ── Launch ────────────────────────────────────────────────────────────────────

def open_browser():
    webbrowser.open('http://localhost:5000')

@app.route('/api/xrd/preview', methods=['POST'])
def xrd_preview():
    """
    Parse an uploaded XRD file and return data for live preview plot.
    Downsampled to ≤2000 points for fast rendering.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file'}), 400
        f = request.files['file']
        safe = re.sub(r'[^\w\-.]', '_', f.filename)
        path = os.path.join(UPLOAD_DIR, safe)
        f.save(path)

        import xrd_processor
        data = xrd_processor.parse_xrd_file(path)
        tt   = data['tt'].tolist()
        y    = data['intensity'].tolist()

        MAX_PTS = 2000
        if len(tt) > MAX_PTS:
            step = len(tt) // MAX_PTS
            tt = tt[::step]
            y  = y[::step]

        return jsonify({
            'tt':        tt,
            'intensity': y,
            'tt_min':    round(min(tt), 2),
            'tt_max':    round(max(tt), 2),
            'y_max':     round(max(y), 1),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "━"*50)
    print("  Catalysis Data Toolkit")
    print(f"  pymatgen:          {'ready' if _pymatgen_ready else 'not installed'}")
    print(f"  Materials Project: {'configured' if MP_API_KEY else 'no API key'}")
    print(f"  CIF cache:         {_cache.stats()['entries']} entries "
          f"({_cache.stats()['size_mb']} MB)")
    print("  http://localhost:5000")
    print("━"*50 + "\n")

    # Only open browser once — not in the reloader child process
    import os
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        Timer(1.2, open_browser).start()

    app.run(debug=True, port=5000, use_reloader=True)

