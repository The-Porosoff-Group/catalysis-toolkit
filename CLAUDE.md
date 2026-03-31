# Catalysis Data Toolkit

## Project Overview

Scientific data processing toolkit by **The Porosoff Group** for catalysis research. Flask single-page application with two active modules (GC analysis, XRD refinement) and two placeholder modules (TGA, BET — coming soon). MIT License, Copyright 2026.

Primary platform: Windows (run.bat launcher with conda environment at `.conda_env/`).

## Tech Stack

- **Backend**: Python 3.13, Flask 2.3+, port 5000
- **Frontend**: Vanilla JavaScript + HTML + inline CSS (no frameworks)
- **Fonts**: Syne (sans-serif, UI text), JetBrains Mono (monospace, data/code) — Google Fonts
- **Plots**: HTML5 Canvas for XRD preview (not Plotly/Chart.js), matplotlib for result images
- **Templating**: Jinja2 (Flask)
- **No test suite** — no test files exist in the project

## Directory Structure

```
catalysis-toolkit/
  app.py                          # Flask entry point, all API routes
  config.yaml                     # User config (API keys, cache, performance)
  config.yaml.example             # Config template
  requirements.txt                # Python dependencies
  run.bat                         # Windows launcher (conda env + GSAS-II install)
  run_debug.bat                   # Debug variant
  LICENSE                         # MIT
  README.md                       # User documentation

  templates/
    index.html                    # Single-page app (81KB, all CSS/JS inline)

  modules/
    gc_processor.py               # GC data processing engine (Shimadzu .xlsx)
    xrd_processor.py              # XRD module entry point (thin wrapper)
    tga_processor.py              # TGA stub (coming soon)
    bet_processor.py              # BET stub (coming soon)

    reaction_configs/             # YAML reaction type definitions
      fts.yaml                    # Fischer-Tropsch Synthesis
      co2_hydrogenation.yaml      # CO2 + H2 hydrogenation
      methane_oxidation.yaml      # CH4 partial oxidation
      co_oxidation.yaml           # CO + O2 oxidation
      custom_template.yaml        # User template for new reactions

    xrd/                          # XRD refinement engine (~230KB total)
      __init__.py                 # File parsers, validate_phases(), run() orchestrator
      crystallography.py          # Pure-Python crystallography (peak shapes, space groups, F^2)
      lebail.py                   # Le Bail + Rietveld refinement (scipy.optimize)
      gsasii_backend.py           # GSAS-II integration (optional, conda-only)
      cod_api.py                  # Crystallography Open Database API
      mp_api.py                   # Materials Project API (requires API key)
      cif_cache.py                # Disk-based CIF cache (~500MB limit, LRU pruning)
      xrd_plots.py                # Matplotlib plot generation (dark theme)

  uploads/                        # Auto-created temp file storage
  results/                        # Auto-created output directory
  .gsas_tmp/                      # GSAS-II working directory (gitignored)
```

No `static/` directory — all CSS and JavaScript are embedded in `index.html`.

## Design System

### Color Palette (Dark Theme, GitHub-inspired)

| Variable    | Value     | Usage                              |
|-------------|-----------|-------------------------------------|
| `--bg`      | `#0d1117` | Main background                    |
| `--surface` | `#161b22` | Cards, header, sidebar             |
| `--surface2`| `#1c2128` | Inputs, nested surfaces            |
| `--border`  | `#2d333b` | All borders                        |
| `--accent`  | `#58a6ff` | Primary blue (active state, links) |
| `--accent2` | `#39d353` | Green (success, weight%)           |
| `--warn`    | `#e3b341` | Yellow (warnings, "Coming Soon")   |
| `--danger`  | `#f78166` | Red (errors)                       |
| `--text`    | `#e6edf3` | Primary text                       |
| `--muted`   | `#7d8590` | Secondary text, labels             |

### Typography

- **Syne** (sans-serif): UI text, headings, buttons. Weights: 400, 600, 700, 800.
- **JetBrains Mono** (monospace): Data, filenames, numeric inputs, status. Weights: 400, 600.
- Header logo: Syne 18px weight 800.
- Form labels: 11px uppercase, muted color, 0.8px letter-spacing.

### Layout

- **Grid**: 220px sidebar + flexible main, 56px header, `height: 100vh`
- **Cards**: `border-radius: 10px`, `border: 1px solid var(--border)`, padding 14-20px
- **Form grids**: `grid-template-columns: 1fr 1fr`, gap 14px
- **Transitions**: `all 0.15s` on hover/focus states
- **Scrollbars**: Custom 6px width, border-colored track, muted thumb

### Buttons

- **Le Bail**: Blue (`#58a6ff`)
- **Rietveld**: Green gradient (`linear-gradient(135deg, #238636, #2ea043)`)
- **GSAS-II**: Purple gradient (`linear-gradient(135deg, #8b5cf6, #6d28d9)`)
- **Process GC**: Blue with dark text, weight 800
- **Secondary**: Border only, accent color on hover

## Frontend Architecture

Single HTML file (`templates/index.html`) with 4 module panels switched via sidebar navigation:

1. **GC Analysis** (`#module-gc`) — File upload, reaction type selector, metadata form, inlet flows table, steady-state range picker, process button, results display
2. **XRD Refinement** (`#module-xrd`) — File upload + wavelength selector, Canvas preview with zoom/pan, phase search (COD/MP), candidate list, selected phases, refinement buttons, results with fit plot
3. **TGA** (`#module-tga`) — "Coming Soon" stub
4. **BET** (`#module-bet`) — "Coming Soon" stub

XRD preview: HTML5 Canvas with scroll-wheel zoom, drag pan, double-click reset, `devicePixelRatio` for retina. 6 phase colors cycle: `#f78166, #56d364, #e3b341, #bc8cff, #79c0ff, #ffa657`.

## API Routes

All routes under `/api/`:

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Render index.html |
| `/api/status` | GET | pymatgen/MP/cache status |
| `/api/reaction_configs` | GET | List GC reaction types |
| `/api/process_gc` | POST | Process GC Excel file |
| `/api/download` | GET | Download result file |
| `/api/open_folder` | GET | Open folder in Explorer |
| `/api/xrd/preview` | POST | Parse XRD file, return plot data |
| `/api/xrd/search` | POST | Search COD/MP for phases |
| `/api/xrd/fetch_cif` | POST | Fetch and cache CIF |
| `/api/process_xrd` | POST | Run Le Bail/Rietveld/GSAS-II |
| `/api/xrd/gsas2_status` | GET | Check if GSAS-II installed |
| `/api/xrd/validate_mp_key` | POST | Validate Materials Project key |
| `/api/cache/clear` | POST | Clear CIF cache |

## XRD Module Architecture

### File Parsers (`__init__.py`)
Supports: `.dat` (PowderGraph), `.xy`/`.xye`, `.csv`, `.txt`, `.ras` (Rigaku). Auto-detects format.

### Phase Sources
- **COD** (Crystallography Open Database): ~500k structures, free, no key
- **Materials Project**: ~45k structures, requires API key
- **Manual CIF upload**: User-provided

### Refinement Pipeline
```
XRD file --> parse_xrd_file()
         --> validate_phases() [fill cell params, fetch CIF]
         --> run_lebail() / run_rietveld() / run_gsas2()
         --> make_xrd_plot() --> PNG
         --> write_summary_xlsx() --> Excel
```

### crystallography.py (Pure-Python Engine)
Zero external crystallography dependency. Implements:
- Unit cell geometry (`cell_volume`, `d_spacing` for all crystal systems)
- Systematic absences (`is_allowed`) for ~25 common space groups
- Atomic scattering factors (Cromer-Mann, H through Pb)
- Structure factors with Debye-Waller (`structure_factor_sq_dw`)
- Thompson-Cox-Hastings pseudo-Voigt profiles (`tch_fwhm_eta`)
- Caglioti broadening (`caglioti_fwhm`)
- Scherrer crystallite size
- CIF parsing (`parse_cif`)
- Symmetry expansion (`expand_sites_from_cif` — uses pymatgen if available, built-in fallback)

### lebail.py (In-House Refinement)
Iterative refinement loop:
1. Le Bail I_hkl updates (intensity partitioning by profile projection)
2. Per-phase linear scale factors (analytical least squares)
3. Background (Chebyshev polynomial)
4. Cell parameters + profile widths (scipy least_squares, trust-region reflective)
5. B_iso (Debye-Waller) added after iteration 7

Multi-phase: joint residual with explicit inter-phase subtraction. Scale computed analytically — guaranteed optimal.

### gsasii_backend.py (GSAS-II Integration)
Wraps GSASIIscriptable. Returns same result dict format as Le Bail/Rietveld.

## GSAS-II Integration Details

These are hard-won lessons from debugging W2C (Pbcn) fitting failures:

### CIF Space Group Format
GSAS-II requires H-M symbols with spaces: `'P b c n'` not `'Pbcn'`. COD CIFs use compact notation which GSAS-II silently misreads as P 1. The `_patch_cif_for_gsas()` function preprocesses CIF tags to fix this using the `_SG_HM` lookup table (~40 common space groups).

### CIF Strategy
Use the **original CIF from COD** as the primary source (GSAS-II handles symmetry expansion internally). Only fall back to synthetic CIF when no original is available. The synthetic CIF pipeline (`_reduce_to_asymmetric_unit` + `_build_conventional_cif`) can lose atom sites when pymatgen is unavailable.

### Scale Factor Handling
- **Single-phase**: Fix phase scale at 1.0, refine ONLY histogram scale. Phase scale and histogram scale are 100% correlated for single-phase — refining both causes SVD singularity.
- **Multi-phase**: Explicitly turn OFF histogram scale, refine each phase scale sequentially, then all together with histogram scale.

### Profile Refinement Strategy
Split into 3 sub-stages to avoid high correlations:
1. Gaussian: U, V, W
2. Lorentzian Y (size broadening)
3. Lorentzian X (strain broadening)

X and Y are ~99% correlated when refined simultaneously.

### Unit Conversions
GSAS-II stores profile parameters in centidegrees:
- U, V, W: centideg^2 -> divide by 10,000 for deg^2
- X, Y: centideg -> divide by 100 for deg

### Instrument Parameter Defaults
Initial Y should be 2.0 centideg (not 0.0). Zero Lorentzian broadening is unrealistic for nanocrystalline materials and produces wrong initial profiles.

### Cell Refinement
For orthorhombic/cubic/tetragonal: angles are clamped to 90 degrees before cell refinement. When GSAS-II correctly reads the space group, it constrains angles automatically, but if SG falls back to P 1, angles float freely and cause `arccos` errors.

### Weight Fractions
Hill & Howard formula: `w_i = (S_i * Z_i * M_i * V_i) / sum(S_j * Z_j * M_j * V_j)`. ZMV computed from refined scale, formula units Z, molar mass M, cell volume V.

## GC Module

Reads Shimadzu `.xlsx` GC output via direct XML parsing (avoids openpyxl bugs). Calculates:
- Molar flows from peak areas + calibration
- Conversion (reactant-based)
- Carbon selectivity (carbon-number weighted)
- Carbon balance

5 reaction types defined in YAML configs under `modules/reaction_configs/`. Each specifies: reactant, internal standard, inlet species with default flows, species definitions with labels/carbon numbers/detector types (TCD or FID).

Generates 3-panel matplotlib plots + Excel summary with data sheets.

## Configuration

`config.yaml` (gitignored, copy from `config.yaml.example`):
```yaml
materials_project:
  api_key: "YOUR_API_KEY_HERE"
cache:
  directory: "~/.catalysis_toolkit_cache"
  max_size_mb: 500
performance:
  max_outer_iterations: 10
  preload_pymatgen: true
```

`.gitignore`: `__pycache__/`, `*.pyc`, `*.pyo`, `.gsas_tmp/`

## User Preferences

- **Windows primary** development environment (conda, run.bat)
- Prefers **robust, general solutions** over one-at-a-time patches (e.g. handle ALL space groups, not just the failing one)
- Values **diagnostic output** for debugging — print what GSAS-II sees, what CIF was loaded, atom counts
- **Scientific accuracy** is paramount — correct space groups, systematic absences, structure factors
- Uses both in-house Rietveld (always works) and GSAS-II (optional, more powerful but fragile)
- Typical samples: tungsten carbides (W2C Pbcn, WC), metallic tungsten (Im-3m), nanocrystalline catalysts
- XRD data from lab powder diffractometer (Cu K-alpha, ~10-90 deg 2-theta)
