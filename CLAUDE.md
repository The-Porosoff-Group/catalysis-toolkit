# CLAUDE.md — Catalysis Data Toolkit

## Overview

Local web application for processing catalysis research data. Built with **Flask + vanilla JS** (no frontend framework). Runs via `run.bat` (double-click) which sets up a conda environment, installs GSAS-II and dependencies, and launches the server at `http://localhost:5000`.

**Owner:** Marc Porosoff (The Porosoff Group)
**Repo:** `The-Porosoff-Group/catalysis-toolkit` on GitHub

---

## How to Run

```
cd C:\catalysis-toolkit
run.bat          # double-click or run from command prompt
```

`run.bat` handles everything: conda env creation (`.conda_env/`), GSAS-II installation, pip dependencies, and launching `python app.py`. Falls back to a Python venv if conda is unavailable.

---

## Project Structure

```
catalysis-toolkit/
├── app.py                  # Flask server — all routes, module loading, config
├── config.yaml             # Local config (API keys, cache, performance) — NOT committed
├── config.yaml.example     # Template for config.yaml
├── requirements.txt        # Python deps: flask, numpy, scipy, pymatgen, etc.
├── run.bat                 # Windows launcher (conda env + GSAS-II + deps + server)
├── run_debug.bat           # Debug variant of launcher
├── templates/
│   └── index.html          # Single-page app — ALL HTML, CSS, and JS in one file
├── modules/
│   ├── gc_processor.py     # GC analysis engine (xlsx parsing, selectivity, conversion)
│   ├── tga_processor.py    # TGA/TPR/TPO — placeholder (coming soon)
│   ├── bet_processor.py    # BET/isotherm — placeholder (coming soon)
│   ├── xrd_processor.py    # XRD module entry point (re-exports from xrd/)
│   ├── reaction_configs/   # YAML configs for GC reactions
│   │   ├── co2_hydrogenation.yaml
│   │   ├── co_oxidation.yaml
│   │   ├── fts.yaml
│   │   ├── methane_oxidation.yaml
│   │   └── custom_template.yaml
│   └── xrd/                # XRD subpackage — the most complex module
│       ├── __init__.py     # File parsers, run(), space group registry, COMMON_WAVELENGTHS
│       ├── crystallography.py  # Core math: d-spacing, structure factors, Caglioti/TCH profiles,
│       │                       # pseudo-Voigt, Chebyshev background, systematic absences,
│       │                       # Cromer-Mann scattering factors, Lorentz-polarization
│       ├── lebail.py       # In-house Le Bail + Rietveld refinement (scipy-based)
│       ├── gsasii_backend.py   # GSAS-II integration via GSASIIscriptable
│       ├── cod_api.py      # Crystallography Open Database REST API + stick patterns
│       ├── mp_api.py       # Materials Project API (new API, not legacy v2)
│       ├── cif_cache.py    # Persistent disk cache for CIF files (~/.catalysis_toolkit_cache/)
│       └── xrd_plots.py    # Matplotlib 4-panel refinement plot (data+fit, ticks, phases, residuals)
└── uploads/                # User-uploaded data files (created at runtime)
```

---

## Modules

### GC Analysis (active)
- Processes GC `.xlsx` files from gas chromatography experiments
- Calculates conversion, selectivity, carbon balance
- Supports configurable reaction types via YAML (CO2 hydrogenation, FTS, CO oxidation, etc.)
- Parses inlet MFC flows, steady-state injection ranges
- Outputs summary Excel + flow CSV + plot PNG

### XRD Analysis (active)
- Parses `.dat`, `.xy`, `.xye`, `.csv`, `.txt` diffraction data files
- Phase search via COD (Crystallography Open Database) and Materials Project APIs
- Three refinement backends:
  - **Le Bail** — in-house, scipy-based, profile fitting without structure factors
  - **Rietveld** — in-house, scipy-based, uses structure factors for intensity constraints
  - **GSAS-II** — external Fortran-accelerated engine via GSASIIscriptable wrapper
- Outputs: weight fractions (Hill & Howard), crystallite size (Scherrer), lattice parameters, R-factors

### TGA / TPR / TPO (coming soon)
### BET / Isotherm (coming soon)

---

## Frontend Design

**Single-page app** in `templates/index.html` — all HTML, CSS, and JS in one file. No build step, no framework, no external JS libraries.

### Theme
- **Dark-only** (GitHub dark inspired), no light mode toggle
- Background: `#0d1117`
- Surface: `#161b22`, `#1c2128`
- Border: `#2d333b`
- Accent blue: `#58a6ff`
- Success green: `#39d353`
- Warning yellow: `#e3b341`
- Danger red: `#f78166`
- Text: `#e6edf3`
- Muted: `#7d8590`

### Fonts
- **Headings/UI:** `Syne` (Google Fonts) — weights 400, 600, 700, 800
- **Code/Numbers:** `JetBrains Mono` (Google Fonts) — weights 400, 600
- Loaded via `fonts.googleapis.com`

### Layout
- CSS Grid: `220px sidebar | flexible main area`, 56px header
- Sidebar navigation with active state (blue left border + blue text)
- Main area: scrollable, card-based workflow with numbered steps
- Not responsive — fixed 220px sidebar, no mobile breakpoints

### Charts
- **XRD preview:** Custom Canvas 2D rendering (no charting library)
  - Scroll-to-zoom, drag-to-pan, double-click to reset
  - Stick pattern overlay for phase identification
- **Results plots:** Server-rendered Matplotlib PNGs, base64-embedded
- Phase color palette: `['#f78166','#56d364','#e3b341','#bc8cff','#79c0ff','#ffa657']`

### XRD Workflow (4 steps)
1. **Upload** — file drop zone + live Canvas preview
2. **Instrument** — wavelength selector (Cu/Mo/Co/Cr/Fe/Ag/custom), 2theta range, background terms
3. **Phases** — search COD/MP by elements/formula/name, select candidates, manual CIF upload
4. **Refine** — choose Le Bail / Rietveld / GSAS-II, view results (stats, phases, plot, download)

### GC Workflow (4 steps)
1. **Upload** — `.xlsx` file drop zone
2. **Reaction type** — select from YAML config cards
3. **Details** — catalyst ID, T, P, GHSV, inlet flows, SS injection range, output folder
4. **Process** — run analysis, view results (conversion, selectivities, carbon balance, plot)

---

## XRD Architecture Details

### Refinement Pipeline
1. Parse data file → `(tt, intensity, sigma)`
2. Search COD/MP for candidate phases → CIF text + cell params + space group
3. Generate reflections per phase (systematic absence filtering + F² filtering)
4. Run refinement:
   - **Le Bail/Rietveld:** scipy `least_squares` with outer Le Bail loop
   - **GSAS-II:** 6-stage sequential refinement:
     1. Background + phase scales (one-at-a-time to break correlation)
     2. Profile params (U, V, W, X, Y — Caglioti + TCH)
     3. Cell parameters (angles clamped by crystal system)
     4. Atomic displacement (B_iso / Uiso)
     5. Atom positions (XYZ) — critical for carbides
     6. Final background + scale polish
5. Extract per-phase patterns (GSAS-II phase isolation or profile reconstruction)
6. Compute weight fractions via Hill & Howard (1987): `W_α = S_α·Z_α·M_α·V_α / Σ`

### Key Constraints in GSAS-II
- Histogram scale fixed at 1.0 (only phase scales refined) to avoid N+1 scale degeneracy
- Cell angles constrained by crystal system (90° for cubic/orthorhombic/tetragonal)
- Complex phases (>6 asymmetric atoms): refinement cycles doubled
- Adaptive symprec for asymmetric unit reduction (0.01 → 0.05 → 0.1 Å)

### Profile Model
- **Gaussian:** Caglioti FWHM² = U·tan²θ + V·tanθ + W
- **Lorentzian:** H_L = X·tanθ + Y/cosθ (strain + size broadening)
- **Pseudo-Voigt:** η·Lorentzian + (1−η)·Gaussian, mixing via TCH 5th-order formula
- Initial U/V/W estimated from observed peak widths; X/Y from FWQM/FWHM ratio analysis

### Structure Factors
- Cromer-Mann 9-parameter atomic scattering factors (30+ elements)
- Debye-Waller thermal damping: `exp(-B_iso·(sinθ/λ)²)`
- Two-pass ghost reflection filter: absolute threshold (1e-4) + relative threshold (0.1% of max F²)
- Systematic absence rules implemented for 20+ space groups (Pbcn, P63/mmc, Pm-3n, Fd-3m, etc.)

### External APIs
- **COD:** REST CSV endpoint at `crystallography.net/cod/result.php` — requires custom User-Agent header (COD blocks default python-requests UA)
- **Materials Project:** New API at `api.materialsproject.org` — requires API key in `config.yaml`, uses `X-API-KEY` header, `_fields` parameter (not `fields`)
- **CIF cache:** Persistent disk cache at `~/.catalysis_toolkit_cache/`, keyed by `cod:<id>` or `mp:<id>`, never expires, capped at 500 MB

---

## Known Issues / Active Work

### W2C / Carbide Fitting
The primary ongoing challenge. Metallic tungsten (W, BCC Im-3m) fits well with all backends, but tungsten carbides (W2C Pbcn, WC P-6m2) have had convergence issues with GSAS-II:

1. **Space group loaded as P1** — Materials Project phases sometimes arrive without proper spacegroup_number in the phase dict, causing GSAS-II to treat the phase as triclinic (no angle constraints → metric tensor divergence)
2. **Ghost reflections** — Multi-element compounds (W+C) have F²≈0 at certain (hkl) due to destructive interference, but incomplete site lists cause these to appear as real peaks
3. **Profile mismatch** — Carbides have different broadening characteristics than metals; Lorentzian X/Y initialization is important
4. **Asymmetric unit reduction** — Tight symprec can incorrectly merge non-equivalent W and C sites

### Commits addressing these (most recent first):
- `8c46542` — Fix MP phases loaded as P1: merge symmetry data into phase dict
- `f2ad0b8` — Use GSAS-II refined Fc² for tick positions
- `53876dc` — Multi-element fallback and relative F² filter for ghost reflections
- `396a469` — Add XYZ refinement stage and Lorentzian profile initialization
- `287f635` — Fix histogram/phase scale degeneracy
- `f349262` — Adaptive symprec, F² stick filtering, Caglioti profile init

---

## Config

`config.yaml` (gitignored) — see `config.yaml.example` for template:
- `materials_project.api_key` — MP API key (from next-gen.materialsproject.org/api)
- `cache.directory` — CIF cache location (default `~/.catalysis_toolkit_cache`)
- `cache.max_size_mb` — cache size limit (default 500)
- `performance.max_outer_iterations` — Le Bail iteration limit (default 10)
- `performance.preload_pymatgen` — pre-import pymatgen at startup (default true)

---

## Dependencies

Core: `flask`, `numpy`, `scipy`, `matplotlib`, `pandas`, `pyyaml`, `requests`, `pymatgen`, `openpyxl`
Optional: GSAS-II (`gsas2pkg` via conda or GitHub clone + pip install)
