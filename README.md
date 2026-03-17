# Catalysis Data Toolkit

A local web app for processing heterogeneous catalysis data. Drag-and-drop interface, no coding required after setup.

## Features

- **GC Analysis** — molar flows, conversion, carbon selectivity, carbon balance from Shimadzu GC output files
- **XRD Refinement** — Le Bail, Rietveld, and GSAS-II refinement with COD/Materials Project phase search
- **Modular design** — add new reaction types via YAML config files; add new data types via Python modules
- **Planned modules** — TGA/TPR/TPO, BET/isotherm

---

## Quick Start (Windows)

### 1. Install Python (one time)
Download from https://www.python.org/downloads/  
✅ Check **"Add Python to PATH"** during installation.

### 2. Download this toolkit
**Option A — with Git:**
```
git clone https://github.com/YOUR-USERNAME/catalysis-toolkit.git
cd catalysis-toolkit
```
**Option B — without Git:**  
Click the green **Code** button on GitHub → **Download ZIP** → extract the folder.

### 3. (Optional) Enable GSAS-II refinement

GSAS-II is a powerful refinement engine that can be used alongside the built-in Le Bail and Rietveld methods. It requires **git** and is installed automatically from GitHub.

1. Install **Git** from https://git-scm.com/downloads (if not already installed)
2. That's it — `run.bat` will clone and install GSAS-II automatically on the next launch

**Manual install** (if you prefer):
```
git clone --depth 1 https://github.com/AdvancedPhotonSource/GSAS-II.git
cd GSAS-II
pip install .
```

The purple **GSAS-II Refinement** button will appear in the XRD panel once it's installed.

> **Note:** If your folder path contains spaces (e.g. `C:\Users\Marc Porosoff\...`), conda may warn about this. It usually still works, but moving the toolkit to `C:\catalysis-toolkit` avoids the issue entirely.

> **GSAS-II is optional.** Le Bail and Rietveld work without it. If conda is not installed, `run.bat` falls back to a standard Python venv automatically.

### 4. Run it
Double-click **`run.bat`**

### 5. (Optional) Configure Materials Project API
For complete coverage of metals, carbides, nitrides and all single-element phases,
add your free Materials Project API key to `config.yaml`:
```yaml
materials_project:
  api_key: "your_32_char_key_here"
```
Get a free key at https://materialsproject.org → sign in → dashboard → API key.

> **Important:** `config.yaml` is listed in `.gitignore` — your key will never
> be committed to GitHub.

That's it. The browser opens automatically at `http://localhost:5000`.

**First launch:** `run.bat` creates a `.conda_env` folder (if conda is available) or a `.venv` folder and installs all dependencies there (including pymatgen ~500 MB, and GSAS-II if using conda). This takes several minutes once. Every subsequent launch is instant.

**Self-contained:** Everything lives in the toolkit folder. Nothing is installed to your system Python. To uninstall completely, just delete the toolkit folder.

---

## How to Use

### GC Analysis
1. **Drop** your `.xlsx` GC file onto the upload area
2. **Select** the reaction type (FTS, CO2 hydrogenation, etc.)
3. **Fill in** catalyst ID, conditions, and MFC inlet flows
4. **Set** the steady-state injection range
5. **Choose** an output folder (or leave blank to save in `results/`)
6. Click **Process GC Data**

Results appear immediately: conversion, selectivities, carbon balance, and a 3-panel plot. Click **Open Output Folder** to access your files.

### XRD Refinement
1. **Drop** your XRD data file (`.xy`, `.xye`, `.dat`, `.csv`, `.txt`, `.ras`)
2. **Select** the X-ray source wavelength
3. **Search** for phases by element, name, or formula (pulls from COD and optionally Materials Project)
4. **Add** phases to the refinement list
5. Click **Le Bail**, **Rietveld**, or **GSAS-II** to run refinement

Results include Rwp, Rp, χ², GoF, refined cell parameters, crystallite sizes, weight fractions, and a fit plot.

---

## Adding a New Reaction Type

Create a new `.yaml` file in `modules/reaction_configs/`. Copy `custom_template.yaml` as a starting point. The app detects it automatically on next launch.

Key fields:
```yaml
name: My Reaction
reactant: CO          # must match a species label below
internal_standard: Ar
inlet_species:
  - { label: CO,  default_sccm: 10 }
  - { label: H2,  default_sccm: 20 }
  - { label: Ar,  default_sccm: 15 }
species:
  "Column Header in GC File": { label: CO, cn: 1, det: TCD }
```

---

## File Format

Your GC `.xlsx` file should follow the Shimadzu sequence output format:
- Row 1: Sequence name
- Row 3: Species names (column headers)
- Row 4: `Amount` / `Peak Area` labels
- Row 5+: Data rows — first column is injection label (e.g. `Bypass 01`, `ExptName 06`)

---

## Adding a New Module (for developers)

1. Create `modules/your_module_processor.py`
2. Add a `MODULE_INFO` dict and a `run(filepath, output_dir, metadata, params)` function
3. Import it in `app.py` and add an entry to the `MODULES` list
4. Add a route in `app.py` and a panel in `templates/index.html`

---

## Project Structure

```
catalysis-toolkit/
├── run.bat                          Windows double-click launcher
├── app.py                           Flask web server
├── requirements.txt
├── README.md
├── modules/
│   ├── gc_processor.py              GC calculation engine
│   ├── tga_processor.py             TGA stub (coming soon)
│   ├── bet_processor.py             BET stub (coming soon)
│   ├── xrd/                         XRD refinement engine
│   │   ├── __init__.py              Entry point, file parsers
│   │   ├── lebail.py                Le Bail and Rietveld refinement
│   │   ├── gsasii_backend.py        GSAS-II integration (optional)
│   │   ├── crystallography.py       Peak shape, statistics, space groups
│   │   ├── xrd_plots.py             Plot generation
│   │   ├── cod_api.py               Crystallography Open Database search
│   │   ├── mp_api.py                Materials Project API search
│   │   └── cif_cache.py             CIF file caching
│   └── reaction_configs/
│       ├── fts.yaml                 Fischer-Tropsch
│       ├── co2_hydrogenation.yaml   CO2 + H2
│       ├── methane_oxidation.yaml   Partial oxidation of CH4
│       ├── co_oxidation.yaml        CO + O2
│       └── custom_template.yaml     Template for new reactions
├── templates/
│   └── index.html                   Web interface
├── uploads/                         Temp upload storage (auto-created)
└── results/                         Default output folder (auto-created)
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python not found` | Re-install Python, check "Add to PATH" |
| Browser doesn't open | Manually go to `http://localhost:5000` |
| Port 5000 in use | Edit `app.py`, change `port=5000` to `port=5001` |
| File won't upload | Check it's a `.xlsx` (not `.xls` or `.csv`) |
| FID flows all zero | CH4 TCD bridge unavailable — see GC_SKILL.md |
| New `.yaml` not showing | Restart the app (close and re-run `run.bat`) |
| GSAS-II button not appearing | Run `conda init cmd.exe` in Command Prompt, restart, re-run `run.bat` |
| `CondaError: Run 'conda init'` | Open Command Prompt, run `C:\miniforge\condabin\conda init cmd.exe`, restart |
| Conda warns about spaces in path | Move toolkit folder to `C:\catalysis-toolkit` to avoid spaces |
