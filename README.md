# Catalysis Data Toolkit

A local web app for processing heterogeneous-catalysis data. Drag-and-drop interface, no coding required after the one-time install.

Supported workflows:

- **GC analysis** — molar flows, conversion, carbon selectivity, and carbon balance from Shimadzu GC output.
- **XRD analysis** — phase identification via Materials Project and COD lookup, Le Bail and Rietveld refinement, full GSAS-II refinement with per-phase controls, instrument-profile import, and a WC/W2C production preset.
- **Modular design** — drop a new reaction config (`.yaml`) into `modules/reaction_configs/` and the GC engine picks it up; add a new data type by writing a Python module under `modules/`.

Coming later: TGA / TPR / TPO, BET / isotherm, broader XRD presets.

---

## Quick start (Windows)

### 1. Install Python (one time)

Download from <https://www.python.org/downloads/>. Tick **"Add Python to PATH"** during installation.

### 2. Get the toolkit

```bat
git clone https://github.com/The-Porosoff-Group/catalysis-toolkit.git
cd catalysis-toolkit
```

Or download the green **Code → Download ZIP** and extract the folder.

### 3. Add your Materials Project API key

```bat
copy config.yaml.example config.yaml
notepad config.yaml
```

Get a free key at <https://next-gen.materialsproject.org/api> and paste it into the `api_key` field. Save and close.

> `config.yaml` is git-ignored. Your key never leaves your machine.

### 4. Launch

Double-click **`run.bat`**. It creates a self-contained `.venv` folder on first run (~500 MB, a couple of minutes) and installs everything inside the toolkit folder. Future launches are instant.

The browser opens at `http://localhost:5000`.

To uninstall completely, delete the toolkit folder.

---

## Quick start (macOS)

### 1. Install Miniforge (one time)

Miniforge gives you conda and is required for the easiest GSAS-II support. It includes the `conda-forge` channel by default, which provides pre-compiled binaries for scientific packages with Fortran/C++ backends, making installs faster and more reliable on macOS than standard pip.

Download from <https://github.com/conda-forge/miniforge> and run the installer, or install via Homebrew:

```bash
brew install miniforge
```

After installing, initialize conda for your shell:

```bash
conda init zsh
```

Then close and reopen your terminal.

### 2. Get the toolkit

```bash
git clone https://github.com/The-Porosoff-Group/catalysis-toolkit.git
cd catalysis-toolkit
```

### 3. Create and activate a conda environment

```bash
conda create -n catalysis python=3.11 -y
conda activate catalysis
```

> Run `conda activate catalysis` each time you open a new terminal session.

### 4. Install dependencies

Install the scientific packages via conda-forge first, then the remaining Python packages via pip:

```bash
conda install -c conda-forge numpy scipy matplotlib pandas pymatgen -y
python -m pip install flask pyyaml requests pycifrw xmltodict openpyxl
```

> **First run:** pymatgen is ~500 MB, so this may take a few minutes.

### 5. Optional: enable GSAS-II refinement

```bash
conda install gsas2pkg -c briantoby -y
```

The purple **GSAS-II Refinement** button appears in the XRD panel once GSAS-II is detected.

> GSAS-II is optional. Le Bail and in-house Rietveld work without it.

### 6. Add your Materials Project API key

```bash
cp config.yaml.example config.yaml
nano config.yaml
```

Get a free key at <https://next-gen.materialsproject.org/api> and paste it into the `api_key` field. Save and close.

> `config.yaml` is git-ignored. Your key never leaves your machine.

### 7. Run the app

```bash
python app.py
```

Then open your browser at `http://localhost:5000`.

---

## XRD / Rietveld workflow

The XRD panel has three engines, in increasing power:

| Engine | What it does | When to use |
|--------|--------------|-------------|
| **Le Bail** | Free peak-intensity fit. Refines cell, profile, scale. | Phase ID and quick cell parameters. |
| **Rietveld** | Structure-constrained intensities. Requires CIF with atoms. | When you have CIFs and want weight fractions. |
| **GSAS-II** | Full Rietveld via `GSASIIscriptable`. Per-phase controls, instrument-profile import, phase isolation, March-Dollase preferred orientation, configurable refinement stages. | Production refinements. |

### Common steps

1. **Upload** your `.dat` / `.xy` / `.xye` powder pattern.
2. **Pick wavelength** (Cu Kα default).
3. **Pick the 2θ window** (auto-detected from the file).
4. **Search Materials Project or COD** by elements, formula, or name. Add phases to the refinement list.
5. A **per-phase refinement card** appears for each selected phase. Each card has size / mustrain checkboxes, PO mode (off / fixed / refined), PO axis (h k l), and PO value. Defaults are conservative — tick what is appropriate for your sample.
6. **Optional:** tick the green **WC/W2C refinement preset**. It pre-fills the production recipe: verification mode, cell, phase isolation, PO hex [001], Refine Zero (fix Disp), Free X + Y ≥ 0, Fix WC PO at 0.905, and configures the WC phase card with the validated March-Dollase ratio.
7. **Optional:** click **▶ Advanced** to see all individual refinement toggles. Every option the preset turns on is also exposed here for manual override.
8. **Click GSAS-II Refinement.** Stats and per-phase results render below the plot.

### Instrument calibration

If you have a NIST line standard (Si 640g, LaB6, etc.) measured on your diffractometer, you can produce a `.instprm` file that pins U/V/W/X/SH/L/Zero to physical instrument values:

1. Upload the standard's pattern.
2. Search Materials Project for the standard, for example Si.
3. Tick **Advanced → Calibration (instprm)**.
4. Click **GSAS-II Refinement**. It runs the calibration pipeline instead of a normal refinement and writes `<instrument>_<standard>.instprm` to the toolkit root.
5. The next time you select the matching instrument profile in the GUI, that file is auto-loaded so U/V/W/X are pinned and only Y + sample displacement refine for your real sample.

A pre-computed file (`smartlab_Si640g.instprm`) is shipped for the Rigaku SmartLab. For other instruments, run the calibration once yourself.

### Local CIF fixtures

Some Materials Project entries import incorrectly into GSAS-II when round-tripped through pymatgen's CIF writer. For example, `mp-2034` W2C used to land as P1 instead of Pbcn. The toolkit ships canonical CIFs in `fixtures/` that override the round-tripped MP CIF for these entries:

```text
fixtures/w2c_pbcn_mp_2034.cif      # W2C, Pbcn — overrides mp-2034
```

To add a fixture for another MP entry, drop the CIF into `fixtures/` and add an entry to `_LOCAL_FIXTURES` in `modules/xrd/mp_api.py`.

---

## GC workflow

1. **Drop** your `.xlsx` GC file onto the upload area.
2. **Select** the reaction type: FTS, CO₂ hydrogenation, methane oxidation, CO oxidation, or your own.
3. **Fill in** catalyst ID, conditions, and MFC inlet flows.
4. **Set** the steady-state injection range.
5. **Choose** an output folder, or leave blank to save in `results/`.
6. **Click Process GC Data.**

Results appear immediately: conversion, selectivities, carbon balance, and a 3-panel plot. Click **Open Output Folder** to access the saved files.

### Adding a new reaction type

Copy `modules/reaction_configs/custom_template.yaml` as a starting point. The app detects the new file on next launch. Required fields:

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

GC file format: Shimadzu sequence output (`.xlsx`) — row 1 sequence name, row 3 species headers, row 4 `Amount` / `Peak Area` labels, row 5+ data with the first column as the injection label, such as `Bypass 01` or `ExptName 06`.

---

## Adding a new data-processing module

1. Create `modules/your_module_processor.py`.
2. Provide a `MODULE_INFO` dict and a `run(filepath, output_dir, metadata, params)` function.
3. Register it in `app.py` by adding it to the `MODULES` list, adding a route, and adding a panel in `templates/index.html`.

---

## Project structure

```text
catalysis-toolkit/
├── run.bat                          Windows launcher
├── run_debug.bat                    Launcher with extra logging
├── app.py                           Flask web server
├── requirements.txt                 Python dependencies
├── config.yaml.example              Template for config.yaml
├── README.md
├── LICENSE
├── smartlab_Si640g.instprm          Measured SmartLab instrument profile
├── cal_si.py                        Helper for Si-standard prep
├── calibrate_instprm.py             Instrument calibration runner
├── fixtures/                        Canonical CIFs overriding MP round-trips
│   └── w2c_pbcn_mp_2034.cif
├── modules/
│   ├── gc_processor.py              GC engine
│   ├── tga_processor.py             TGA stub
│   ├── bet_processor.py             BET stub
│   ├── xrd_processor.py             XRD entry-point
│   ├── reaction_configs/
│   │   ├── fts.yaml
│   │   ├── co2_hydrogenation.yaml
│   │   ├── methane_oxidation.yaml
│   │   ├── co_oxidation.yaml
│   │   └── custom_template.yaml
│   └── xrd/
│       ├── __init__.py              Routing (Le Bail / Rietveld / GSAS-II)
│       ├── cod_api.py               COD search + stick-pattern preview
│       ├── mp_api.py                Materials Project search + fixture override
│       ├── cif_cache.py             On-disk CIF cache (fixture-aware)
│       ├── crystallography.py       hkl, d-spacing, structure factors
│       ├── lebail.py                Le Bail / in-house Rietveld
│       ├── gsasii_backend.py        Full GSAS-II Rietveld pipeline
│       ├── gsasii_calibration.py    Instrument profile calibration
│       └── xrd_plots.py             Plot rendering
├── templates/
│   └── index.html                   Web UI
├── results/                         Default output folder (auto-created)
└── uploads/                         Temp upload storage (auto-created)
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python not found` | Re-install Python and tick **Add to PATH**. |
| `conda: command not found` | Run `conda init zsh`, close and reopen Terminal. |
| `CondaError: Run 'conda init'` | Open Command Prompt, run `C:\miniforge\condabin\conda init cmd.exe`, then restart. |
| Browser does not open | Manually go to `http://localhost:5000`. |
| Port 5000 in use | Edit `app.py`, change `port=5000` to `port=5001`. |
| File will not upload | Verify `.xlsx` for GC or `.dat` / `.xy` / `.xye` for XRD. |
| FID flows all zero | CH4 TCD bridge unavailable — see `GC_SKILL.md`. |
| New `.yaml` reaction does not show | Restart the app. |
| MP search returns nothing | Confirm `config.yaml` has a valid API key. |
| GSAS-II button does not appear or refinement will not run | GSAS-II is optional; install it from <https://github.com/AdvancedPhotonSource/GSAS-II>. Le Bail and in-house Rietveld work without it. |
| GSAS-II has trouble with paths containing spaces | Move the toolkit folder to a path like `C:\catalysis-toolkit` on Windows. |
| Stale CIF in cache | Delete `~/.catalysis_toolkit_cache/` and re-fetch. |

---

## License

MIT (see `LICENSE`).
