# XRD Fitting Toolkit

**Release 1.2.1** — simpler instrument selection and calibration controls.

Standalone XRD fitting interface from the Catalysis Data Toolkit.

This version opens directly into the XRD workflow and hides unfinished modules. It is intended as a local desktop web app for importing CIFs, previewing XRD tick patterns, and running GSAS-II refinements from a browser-based GUI.

## Quick Start

1. Download or clone this repository.
2. Add your Materials Project API key:

   ```bat
   copy config.yaml.example config.yaml
   notepad config.yaml
   ```

3. Start the XRD-only GUI:

   ```bat
   xrd_toolkit\run_xrd_toolkit.bat
   ```

4. Open, if the browser does not open automatically:

   ```text
   http://localhost:5000/xrd
   ```

The first launch creates a local Python environment and installs dependencies. GSAS-II installation can take several minutes.

You can also add, replace, test, or remove the key under **Phase Identification → Materials Project API key** after launching. Changes apply immediately and are saved in the local, git-ignored `config.yaml`. Editing that file manually requires restarting the toolkit.

Name searches accept element names (`tungsten`), chemical names (`tungsten carbide`), and formulas (`W2C`). **+ Loose** allows additional elements. **Filter results** searches full element names and text available in returned cards; database search does not search arbitrary descriptions or mineral names.

## Instruments and calibration

The **Instrument** selector beside **Run Refinement** has four choices:
**Rigaku SmartLab (BB)**, **Synergy-S (capillary)**,
**Benchtop Cu — flat plate (Si 640g)**, and **None / calibration**.
Choose the instrument used for a sample scan. Each named instrument loads its
bundled calibration and geometry. Scan range, source, and other scan inputs
remain in **Scan Settings**.

The **Benchtop Cu — flat plate (Si 640g)** entry loads the bundled
`benchtop_Cu_Si640g.instprm` automatically. It is included in Git for other users
of that same instrument. The selected 20–90° Cu-doublet calibration achieved
Rwp 8.502%; see [settings, comparisons, and limits](docs/benchtop_calibration.md).
The instrument make/model is not recorded, so this is not a generic calibration
for other benchtop machines.

To calibrate from **NIST Si 640g**:

1. Upload the standard measured with the same optics and configuration as your
   samples. Set an angular range with several Si peaks.
2. Select **None / calibration** beside **Run Refinement**, then choose the
   standard's geometry. Flat plate is the initial choice. No saved `.instprm`
   or optional file override is used in calibration.
3. Click **Run calibration**. Ordinary Cu data use a Cu Kα1/Kα2 doublet
   automatically. The certified Si cell (`a = 5.431109 Å`) is built in; no
   database/API key or selected sample phase is required. Selected sample
   phases are ignored.
4. Inspect the fit and warnings, then download the candidate `.instprm`, report,
   and GSAS-II project. Parameter checks cover the fitted angular range; they
   are not a fit-quality certification. Chi-square requires valid uncertainties,
   which cannot be inferred from counts-per-second data without counting times.
5. Keep the downloaded file locally and test it with a repeat standard scan
   before relying on sample broadening. Select a named instrument again to
   return to sample fitting.

The calibrator explicitly applies the selected geometry. It starts from fresh
profile values, fixes the certified cell and sample broadening, initializes the
reflection list, then extracts Si peak intensities independently with Le Bail
while fitting instrument widths and Zero in stages. This avoids forcing texture
or slit-dependent intensity ratios into width parameters. Cu doublet intensity
ratio is fixed at 0.5. A mounting-dependent position error can still affect Zero;
check peak positions and repeatability. Instrument/optics-specific asymmetry and
absorption can require further work in GSAS-II.

To use another `.instprm` for a sample fit, select a named instrument with the
matching geometry, expand **GSAS controls**, and choose the optional instrument
file override. The single-bank GSAS-II `Type:PXC` file supplies the wavelength
and peak profile; the selected instrument supplies the geometry. Clear the
override to use the bundled file again. Legacy `.prm`/`.inst`, TOF, and multi-bank
files must first be converted/exported as a supported `.instprm`. Other
calibration standards can be fitted externally and imported this way.

Calibration candidates and reports remain together in the run's results folder.
Previously saved files in `local_instruments/` and the local-profile API are
preserved, but saved profiles do not add entries to the main dropdown. The
interface no longer asks for a local instrument name or saves named profiles.

## Main Features

- XRD file upload with live preview (`.dat`, `.xy`, `.xye`, `.csv`, `.txt`, `.xlsx`)
- Materials Project phase search
- Manual CIF upload
- CIF caching and validation
- Correct preview tick generation from imported phases
- GSAS-II refinement backend
- `.instprm` instrument parameter support
- Built-in WC/W2C Synergy-S production preset
- Saved user presets
- GSAS HAP, Scherrer-equivalent, or combined size reporting with an explicit K
- Light/dark figure exports with an optional title; single-phase legends omit
  the normalized weight percentage
- Legend position selector, including automatic and outside-right placement;
  finished figures can be updated without refitting
- **Fit Parameters** workbook tab with input scans/uncertainties, fit settings,
  initial/final native GSAS-II parameters and flags, refinement stages,
  CIF/instrument inputs, and software versions; companion `.gpx` project download
- Per-phase controls for:
  - crystallite size
  - microstrain
  - March-Dollase preferred orientation
  - diagnostic uniform-cell handling for W2C-like phases
- Fit warnings and baseline comparison
- Outputs for phase fraction, uncertainty notes, FWHM reference peak, crystallite size, preferred-orientation value, and cell-change percentages

**Wt%** reports GSAS-II mass fractions normalized over the modeled crystalline
phases. It does not include amorphous material or unmodeled phases. **Diffraction
area (%)** is a separate intensity diagnostic, not a weight percentage. Le Bail
and legacy in-house Rietveld fits report diffraction area only; their Wt% is
unavailable. Refit and regenerate older results to replace percentages that were
previously mislabeled as Wt%.

Figure exports use Arial for text, subscripts, and crystallographic overbars.
Install Arial on the computer generating the figures; the font is not bundled.
If unavailable, exports warn and use Liberation Sans, then DejaVu Sans. Existing
PNGs keep their original typography until regenerated.

The **Fit Parameters** tab stores values as JSON text to retain their precision.
Join numbered parts before decoding a long value. Open the companion `.gpx`
project in GSAS-II to inspect the fitted model; use matching software versions
to reproduce a fit. Older workbooks require a new fit to capture the native state.
After fitting, use **Results → Plot settings → Legend location** to update the
preview and both PNG exports without rerunning the fit. Batch runs also accept
`--legend-location "outside right"`.

## Recommended Workflow

1. Upload the measured XRD pattern.
2. Set wavelength and 2-theta range.
3. Search Materials Project or upload CIFs for the expected phases.
4. Run a constrained baseline fit.
5. Mark/save the baseline in the GUI.
6. Add one refinement freedom at a time.
7. Compare the new fit against the baseline.
8. Save a validated preset for related samples.

Do not keep extra fit freedoms just because Rwp improves. Preferred orientation, Uiso, size, microstrain, and atom-position refinement can all improve the statistic while also changing phase fractions or absorbing model error.

## WC/W2C Preset

The built-in WC/W2C Synergy-S preset uses a fixed WC [001] March-Dollase preferred-orientation value near `0.905`. That value came from a comparison workflow and is meant as a production prior for this specific recipe. It is not a universal WC constant.

## Important Files

```text
app.py                         Flask backend and routes
run.bat                        Full toolkit launcher
xrd_toolkit/run_xrd_toolkit.bat
                               XRD-only launcher
templates/xrd_toolkit/index.html
                               XRD-only GUI
modules/xrd/                   XRD, CIF, crystallography, and GSAS-II code
fixtures/                      Canonical CIF fixtures
config.yaml.example            API-key template
```

## Notes on GitHub Pages

Raw scans, generated results, and `xrd_refinement_presets.json` are local user
files and are excluded from Git. Keep measurements in `data/` or `uploads/`
and fit outputs in `results/`. Canonical `fixtures/` and the bundled
instrument reference profiles remain versioned. Ignore rules do not remove
files from older Git commits.

This toolkit cannot run directly as a static GitHub Pages site because it depends on Python, Flask, GSAS-II, local file uploads, and local refinement outputs. GitHub can host the source code and documentation, but users run the app locally with the launcher.
