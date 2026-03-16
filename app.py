"""
app.py  —  Catalysis Data Toolkit
Local web server. Run with:  python app.py
Then open:  http://localhost:5000
"""

import os, sys, re, shutil, json, webbrowser
from datetime import datetime
from threading import Timer
from flask import Flask, render_template, request, jsonify, send_file

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.join(BASE_DIR, 'modules')
CONFIG_DIR  = os.path.join(MODULES_DIR, 'reaction_configs')
UPLOAD_DIR  = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)
sys.path.insert(0, MODULES_DIR)

import gc_processor
import tga_processor
import bet_processor
import xrd_processor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max upload


# ── Helper: build module list for sidebar ─────────────────────────────────────
MODULES = [
    {'id': 'gc',  'name': 'GC Analysis',     'icon': '⚗️',  'status': 'active'},
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


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    reaction_configs = gc_processor.list_reaction_configs(CONFIG_DIR)
    return render_template('index.html',
                           modules=MODULES,
                           reaction_configs=reaction_configs)

@app.route('/api/reaction_configs')
def get_reaction_configs():
    return jsonify(gc_processor.list_reaction_configs(CONFIG_DIR))

@app.route('/api/process_gc', methods=['POST'])
def process_gc():
    try:
        # ── Receive file ──────────────────────────────────────────────────────
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded.'}), 400
        f = request.files['file']
        if not f.filename.endswith('.xlsx'):
            return jsonify({'error': 'Please upload an .xlsx file.'}), 400

        safe_name  = re.sub(r'[^\w\-.]', '_', f.filename)
        upload_path = os.path.join(UPLOAD_DIR, safe_name)
        f.save(upload_path)

        # ── Parse form data ───────────────────────────────────────────────────
        form        = request.form
        config_file = form.get('reaction_config')
        config_path = os.path.join(CONFIG_DIR, config_file)
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

        # Inlet flows
        inlet_flows = {}
        flows_raw = form.get('inlet_flows', '{}')
        try:
            inlet_flows = json.loads(flows_raw)
            inlet_flows = {k: float(v) for k, v in inlet_flows.items() if v}
        except (json.JSONDecodeError, ValueError):
            pass

        ss_start = int(form.get('ss_start', 1))
        ss_end   = int(form.get('ss_end',   999))

        # ── Output directory ──────────────────────────────────────────────────
        output_base = form.get('output_dir', '').strip()
        if not output_base or not os.path.isdir(output_base):
            output_base = os.path.join(BASE_DIR, 'results')

        ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_id = re.sub(r'[^\w\-]', '_', metadata['catalyst_id'])
        output_dir = os.path.join(output_base, f'{safe_id}_{ts}')

        # ── Run processor ─────────────────────────────────────────────────────
        result = gc_processor.run(
            filepath        = upload_path,
            output_dir      = output_dir,
            reaction_config = reaction_config,
            metadata        = metadata,
            inlet_flows     = inlet_flows,
            ss_start        = ss_start,
            ss_end          = ss_end,
        )

        # Return plot as base64 for inline display
        import base64
        with open(result['plot_path'], 'rb') as img:
            plot_b64 = base64.b64encode(img.read()).decode('utf-8')

        result['plot_b64']    = plot_b64
        result['output_dir']  = output_dir

        # Clean keys for JSON
        clean = {k: v for k, v in result.items() if k != 'plot_path'}
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
    """Opens the output folder in Windows Explorer."""
    path = request.args.get('path', '')
    if os.path.isdir(path):
        os.startfile(path)
        return jsonify({'ok': True})
    return jsonify({'error': 'Folder not found'}), 404


# ── Launch ────────────────────────────────────────────────────────────────────


@app.route('/api/xrd/search', methods=['POST'])
def xrd_search():
    try:
        data = request.get_json()
        elements = data.get('elements', [])
        name     = data.get('name', '').strip()
        wavelength = float(data.get('wavelength', 1.54056))

        if name:
            results = xrd_processor.search_by_name(name)
        elif elements:
            results = xrd_processor.search_by_elements(elements)
        else:
            return jsonify({'error': 'Provide elements or a phase name.'}), 400

        if isinstance(results, dict) and 'error' in results:
            return jsonify(results), 500

        # Add stick patterns for preview
        for r in results[:8]:
            try:
                r['stick_pattern'] = xrd_processor.get_stick_pattern(r, wavelength)
            except Exception:
                r['stick_pattern'] = []

        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/xrd/fetch_cif', methods=['POST'])
def xrd_fetch_cif():
    try:
        data   = request.get_json()
        cod_id = data.get('cod_id')
        if not cod_id:
            return jsonify({'error': 'cod_id required'}), 400
        structure = xrd_processor.fetch_cif(cod_id)
        return jsonify(structure)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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

        import json as _json
        phases_raw = form.get('phases', '[]')
        phases     = _json.loads(phases_raw)

        if not phases:
            return jsonify({'error': 'No phases selected for refinement.'}), 400

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
                'phases':            phases,
                'wavelength':        wavelength,
                'wavelength_label':  wl_label,
                'tt_min':            tt_min,
                'tt_max':            tt_max,
                'n_bg_coeffs':       6,
            }
        )

        import base64
        with open(result['plot_path'], 'rb') as img:
            plot_b64 = base64.b64encode(img.read()).decode()

        return jsonify({
            'plot_b64':     plot_b64,
            'statistics':   result['statistics'],
            'phase_results': result['phase_results'],
            'zero_shift':    result['zero_shift'],
            'summary_path':  result['summary_path'],
            'output_dir':    out_dir,
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

def open_browser():
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("\n" + "━"*50)
    print("  Catalysis Data Toolkit")
    print("  http://localhost:5000")
    print("━"*50 + "\n")
    Timer(1.2, open_browser).start()
    app.run(debug=False, port=5000)
