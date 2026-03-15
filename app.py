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
                  'status': xrd_processor.MODULE_INFO['status']},
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

def open_browser():
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("\n" + "━"*50)
    print("  Catalysis Data Toolkit")
    print("  http://localhost:5000")
    print("━"*50 + "\n")
    Timer(1.2, open_browser).start()
    app.run(debug=False, port=5000)
