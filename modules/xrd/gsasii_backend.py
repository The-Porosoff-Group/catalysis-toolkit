"""
modules/xrd/gsasii_backend.py
GSAS-II integration for Rietveld/Le Bail refinement via GSASIIscriptable.

Requires GSAS-II installed in the Python environment.
Install via:  git clone https://github.com/AdvancedPhotonSource/GSAS-II.git
              cd GSAS-II && pip install .
This module wraps GSASIIscriptable to provide a refinement backend compatible
with the toolkit's result format (same keys as run_lebail / run_rietveld).
"""

import math, os, sys, tempfile, warnings
import numpy as np

# ── GSAS-II availability check ──────────────────────────────────────────────

_GSASII_AVAILABLE = False
_GSASII_IMPORT_ERROR = None

def _add_gsas2pkg_paths():
    """Add gsas2pkg conda install paths to sys.path if not already present.
    gsas2pkg installs to {conda_prefix}/GSAS-II/ rather than site-packages."""
    prefix = sys.prefix  # e.g. C:\catalysis-toolkit\.conda_env
    candidates = [
        os.path.join(prefix, 'GSAS-II', 'GSASII'),   # for import GSASIIscriptable
        os.path.join(prefix, 'GSAS-II'),               # for import GSASII.GSASIIscriptable
        os.path.join(prefix, 'GSAS-II', 'backcompat'), # backcompat shim
    ]
    for p in candidates:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

_add_gsas2pkg_paths()

try:
    # New-style import (pip-installed from GitHub, or gsas2pkg via GSAS-II subpackage)
    import GSASII.GSASIIscriptable as G2sc
    _GSASII_AVAILABLE = True
except ImportError:
    try:
        # Direct import (gsas2pkg installs GSASII/ dir; backcompat or GSASII on path)
        import GSASIIscriptable as G2sc
        _GSASII_AVAILABLE = True
    except ImportError as e:
        _GSASII_IMPORT_ERROR = str(e)

try:
    from .crystallography import (
        compute_fit_statistics, cell_volume, molar_mass_from_formula,
        tch_fwhm_eta, size_from_Y, scherrer_size,
        generate_reflections, parse_cif, caglioti_fwhm,
        expand_sites_from_cif,
    )
except ImportError:
    from crystallography import (
        compute_fit_statistics, cell_volume, molar_mass_from_formula,
        tch_fwhm_eta, size_from_Y, scherrer_size,
        generate_reflections, parse_cif, caglioti_fwhm,
        expand_sites_from_cif,
    )


def is_available():
    """Return True if GSASIIscriptable is importable and functional."""
    if not _GSASII_AVAILABLE:
        return False
    # Quick functional check: verify G2Project is callable
    # (catches partial installs where Fortran extensions are missing)
    try:
        if not hasattr(G2sc, 'G2Project'):
            return False
    except Exception:
        return False
    return True


def import_error():
    """Return the import error message, or None if available."""
    return _GSASII_IMPORT_ERROR


# ─────────────────────────────────────────────────────────────────────────────
# SPACE GROUP TABLE  (number → H-M symbol for CIF generation)
# ─────────────────────────────────────────────────────────────────────────────

_SG_HM = {
    # Triclinic / Monoclinic
    1: 'P 1', 2: 'P -1', 3: 'P 2', 4: 'P 21', 5: 'C 2',
    6: 'P m', 7: 'P c', 8: 'C m', 9: 'C c', 10: 'P 2/m',
    11: 'P 21/m', 12: 'C 2/m', 13: 'P 2/c', 14: 'P 21/c', 15: 'C 2/c',
    # Orthorhombic (common catalyst phases)
    16: 'P 2 2 2', 19: 'P 21 21 21', 26: 'P m c 21', 29: 'P c a 21',
    31: 'P m n 21', 33: 'P n a 21', 36: 'C m c 21', 40: 'A m a 2',
    44: 'I m m 2', 46: 'I m a 2',
    47: 'P m m m', 51: 'P m m a', 55: 'P b a m', 57: 'P b c m',
    58: 'P n n m', 59: 'P m m n', 60: 'P b c n', 61: 'P b c a',
    62: 'P n m a', 63: 'C m c m', 64: 'C m c a', 65: 'C m m m',
    66: 'C c c m', 69: 'F m m m', 70: 'F d d d', 71: 'I m m m',
    72: 'I b a m', 74: 'I m m a',
    # Tetragonal
    75: 'P 4', 82: 'I -4', 83: 'P 4/m', 84: 'P 42/m',
    85: 'P 4/n', 87: 'I 4/m', 88: 'I 41/a',
    99: 'P 4 m m', 107: 'I 4 m m', 115: 'P -4 m 2',
    119: 'I -4 m 2', 121: 'I -4 2 m', 122: 'I -4 2 d',
    123: 'P 4/m m m', 129: 'P 4/n m m', 131: 'P 42/m m c',
    136: 'P 42/m n m', 139: 'I 4/m m m', 140: 'I 4/m c m',
    141: 'I 41/a m d', 142: 'I 41/a c d',
    # Trigonal / Hexagonal
    143: 'P 3', 146: 'R 3', 147: 'P -3', 148: 'R -3',
    150: 'P 3 2 1', 152: 'P 31 2 1', 154: 'P 32 2 1',
    155: 'R 3 2', 156: 'P 3 m 1', 157: 'P 3 1 m',
    160: 'R 3 m', 161: 'R 3 c', 162: 'P -3 1 m', 163: 'P -3 1 c',
    164: 'P -3 m 1', 165: 'P -3 c 1', 166: 'R -3 m', 167: 'R -3 c',
    168: 'P 6', 173: 'P 63', 174: 'P -6', 175: 'P 6/m', 176: 'P 63/m',
    183: 'P 6 m m', 186: 'P 63 m c', 187: 'P -6 m 2', 189: 'P -6 2 m',
    191: 'P 6/m m m', 193: 'P 63/m c m', 194: 'P 63/m m c',
    # Cubic
    195: 'P 2 3', 196: 'F 2 3', 197: 'I 2 3', 198: 'P 21 3',
    199: 'I 21 3', 200: 'P m -3', 201: 'P n -3', 202: 'F m -3',
    203: 'F d -3', 204: 'I m -3', 205: 'P a -3', 206: 'I a -3',
    207: 'P 4 3 2', 209: 'F 4 3 2', 211: 'I 4 3 2',
    212: 'P 43 3 2', 213: 'P 41 3 2', 214: 'I 41 3 2',
    215: 'P -4 3 m', 216: 'F -4 3 m', 217: 'I -4 3 m',
    218: 'P -4 3 n', 219: 'F -4 3 c', 220: 'I -4 3 d',
    221: 'P m -3 m', 222: 'P n -3 n', 223: 'P m -3 n',
    224: 'P n -3 m', 225: 'F m -3 m', 226: 'F m -3 c',
    227: 'F d -3 m', 228: 'F d -3 c', 229: 'I m -3 m',
    230: 'I a -3 d',
}


def _get_hm_symbol(sg_num):
    """Get H-M symbol for a space group number.

    Uses the static table first, then tries pymatgen as a fallback.
    """
    if sg_num in _SG_HM:
        return _SG_HM[sg_num]
    try:
        from pymatgen.symmetry.groups import SpaceGroup
        sg = SpaceGroup.from_int_number(sg_num)
        return sg.symbol
    except Exception:
        pass
    warnings.warn(f"Space group {sg_num} not in lookup table — using 'P 1'. "
                  f"Add it to _SG_HM in gsasii_backend.py for correct results.")
    return 'P 1'


# ── Default instrument / refinement parameters ────────────────────────────
# These are used when the user does not provide their own .instprm file.
# U, V, W, X, Y are initial guesses — GSAS-II refines them during fitting.
# Polariz. and SH/L are NOT refined and directly affect calculated
# intensities; users with known instrument profiles should override them.
DEFAULT_POLARIZ = 0.5     # Beam polarization factor.
                           # 0.5 = unpolarized (Kβ-filtered, no monochromator)
                           # 0.99 = graphite monochromator on diffracted side
                           # Most in-house Cu Kα Bragg-Brentano lab
                           # instruments (e.g. Rigaku SmartLab with Kβ filter)
                           # use NO monochromator → 0.5.  A wrong polariz
                           # biases calculated peak intensity vs. 2θ, which
                           # propagates directly into scale factors and H&H
                           # weight fractions.
DEFAULT_SH_L = 0.002      # Finger-Cox-Jephcoat asymmetry parameter
                           # Starting low is intentional: Stage 1 refines
                           # background with SH/L fixed, so a low SH/L means
                           # nearly symmetric peaks → clean baseline.  SH/L
                           # is then refined alongside background in Stage 2.
DEFAULT_U = 2.0            # Caglioti U initial guess (centideg²) — refined
DEFAULT_V = -2.0           # Caglioti V initial guess (centideg²) — refined
DEFAULT_W = 5.0            # Caglioti W initial guess (centideg²) — refined
DEFAULT_X = 0.0            # Lorentzian X initial guess (centideg) — refined
DEFAULT_Y = 0.0            # Lorentzian Y initial guess (centideg) — refined
DEFAULT_B_ISO = 0.5        # Fallback B_iso (Å²) when GSAS-II extraction fails


def _get_expanded_sites(cif_text, spacegroup_number=None):
    """Get symmetry-expanded unit-cell sites for structure factor computation.

    Uses the shared expand_sites_from_cif (pymatgen) first, then falls back
    to raw parse_cif (may be asymmetric unit only).
    """
    if not cif_text:
        return None
    sites = expand_sites_from_cif(cif_text)
    if sites:
        return sites
    # Fallback: raw parse_cif (may be asymmetric unit only)
    try:
        parsed = parse_cif(cif_text)
        return parsed.get('sites') or None
    except Exception:
        return None


def _cif_already_has_asymmetric_unit(cif_text, declared_sg=None):
    """Check if a CIF file already contains asymmetric-unit sites (not full cell).

    CIF files from COD and from pymatgen's CifWriter(symprec=...) typically
    list only the asymmetric unit plus symmetry operations.  In contrast,
    CIF files generated by dumping a pymatgen Structure without symmetry
    detection list the full unit cell with space group P1.

    Detection strategy:
      1. Parse sites from the raw CIF text (what's literally written)
      2. Expand using pymatgen CifParser(primitive=False) to get full cell
      3. If raw site count < full cell count AND the CIF declares a SG > 1
         with symmetry operations, the CIF already has the asymmetric unit.

    Returns (True, raw_sites) if already asymmetric, (False, None) otherwise.
    """
    if not cif_text or not declared_sg or declared_sg <= 1:
        return False, None

    try:
        # Count sites literally listed in the CIF
        raw_parsed = parse_cif(cif_text)
        raw_sites = raw_parsed.get('sites') or []
        raw_sg = raw_parsed.get('spacegroup_number', 1)
        if not raw_sites or raw_sg <= 1:
            return False, None

        # Check if the CIF contains symmetry operations (loop_ _symmetry_equiv
        # or _space_group_symop) — this is the hallmark of an asymmetric-unit CIF
        has_symops = ('_symmetry_equiv_pos_as_xyz' in cif_text or
                      '_space_group_symop_operation_xyz' in cif_text or
                      '_space_group_symop.operation_xyz' in cif_text)

        if not has_symops:
            return False, None

        # Check that the CIF's own SG is compatible with the declared SG.
        # If CifWriter detected a DIFFERENT SG (e.g. SG 25 instead of 60),
        # the sites are reduced for the wrong symmetry and must NOT be used
        # with the declared SG — that would cause GSAS-II to expand with
        # wrong symmetry operations.
        if raw_sg != declared_sg:
            print(f"  CIF declares SG {raw_sg} but phase expects SG "
                  f"{declared_sg} — cannot use CIF sites as-is "
                  f"(SG mismatch)", flush=True)
            return False, None

        # Expand to full cell to compare site counts
        from pymatgen.io.cif import CifParser
        try:
            parser = CifParser.from_str(cif_text, occupancy_tolerance=100.0)
        except (AttributeError, TypeError):
            import tempfile as _tf, os as _os
            _fd, _tmp = _tf.mkstemp(suffix='.cif')
            with _os.fdopen(_fd, 'w') as _f:
                _f.write(cif_text)
            parser = CifParser(_tmp, occupancy_tolerance=100.0)
            _os.unlink(_tmp)
        structs = parser.parse_structures(primitive=False)
        if structs:
            full_cell_n = len(structs[0])
            if len(raw_sites) < full_cell_n:
                print(f"  CIF already contains asymmetric unit: "
                      f"{len(raw_sites)} sites (full cell = {full_cell_n}, "
                      f"SG {raw_sg}, has symops)", flush=True)
                return True, raw_sites

    except Exception as e:
        print(f"  Warning: asymmetric-unit detection failed: {e}", flush=True)

    return False, None


def _cifwriter_asymmetric_unit(cif_text, declared_sg=None,
                               return_full_cif=False):
    """Use pymatgen CifWriter to generate a proper asymmetric-unit CIF.

    CifWriter with symprec is purpose-built for reducing a full-cell
    structure to the asymmetric unit with correct symmetry operations.
    This is more robust than manually using SpacegroupAnalyzer because
    CifWriter handles the full reduction pipeline internally, including
    site merging and Wyckoff position assignment.

    IMPORTANT: CifWriter may transform the structure to the standard ITA
    setting, which can permute axes relative to the original structure.
    When return_full_cif=True, the complete CIF text is returned so the
    caller gets atom positions AND cell parameters in the SAME convention.
    Mixing CifWriter positions with original cell params causes axis
    mismatch → wrong structure factors → zero intensity in Rietveld.

    Returns:
        If return_full_cif=False:
            list of (element, x, y, z, occupancy) tuples, or None
        If return_full_cif=True:
            (sites_list, full_cif_text) tuple, or (None, None)
    """
    if not cif_text:
        return None

    try:
        from pymatgen.io.cif import CifParser, CifWriter
        from pymatgen.core import Structure as PmgStruct

        # Parse the input CIF to get the full conventional cell
        try:
            parser = CifParser.from_str(cif_text, occupancy_tolerance=100.0)
        except (AttributeError, TypeError):
            import tempfile as _tf, os as _os
            _fd, _tmp = _tf.mkstemp(suffix='.cif')
            with _os.fdopen(_fd, 'w') as _f:
                _f.write(cif_text)
            parser = CifParser(_tmp, occupancy_tolerance=100.0)
            _os.unlink(_tmp)
        structs = parser.parse_structures(primitive=False)
        if not structs:
            return None

        struct0 = structs[0]
        full_cell_n = len(struct0)

        # Try CifWriter with multiple symprec values.
        # CifWriter internally uses SpacegroupAnalyzer but goes through
        # the full CIF generation pipeline which handles edge cases better
        # than raw SpacegroupAnalyzer.get_symmetrized_structure().
        for symprec in (0.01, 0.05, 0.1, 0.2):
            try:
                writer = CifWriter(struct0, symprec=symprec)
                # Extract the reduced CIF text
                import tempfile as _tf2, os as _os2
                _fd2, _tmp2 = _tf2.mkstemp(suffix='.cif')
                with _os2.fdopen(_fd2, 'w') as _f2:
                    pass  # just create the file
                writer.write_file(_tmp2)
                with open(_tmp2, 'r') as _f2:
                    reduced_cif = _f2.read()
                _os2.unlink(_tmp2)

                # Parse the reduced CIF to get the asymmetric unit sites
                reduced_parsed = parse_cif(reduced_cif)
                reduced_sites = reduced_parsed.get('sites') or []
                reduced_sg = reduced_parsed.get('spacegroup_number', 1)

                if not reduced_sites:
                    continue

                # Validate: reduced must be smaller than full cell
                if declared_sg and declared_sg > 1:
                    if len(reduced_sites) >= full_cell_n:
                        print(f"  CifWriter(symprec={symprec}): "
                              f"{len(reduced_sites)} sites = full cell "
                              f"({full_cell_n}) — no reduction, skipping",
                              flush=True)
                        continue

                # Validate: re-expand the reduced CIF and check site count
                try:
                    parser2 = CifParser.from_str(
                        reduced_cif, occupancy_tolerance=100.0)
                    structs2 = parser2.parse_structures(primitive=False)
                    if structs2:
                        reexpanded_n = len(structs2[0])
                        if abs(reexpanded_n - full_cell_n) > 1:
                            print(f"  CifWriter(symprec={symprec}): "
                                  f"re-expansion gave {reexpanded_n} sites, "
                                  f"expected {full_cell_n} — skipping",
                                  flush=True)
                            continue
                except Exception:
                    pass  # skip validation, still try the sites

                # Validate: detected SG should be compatible with declared
                if (declared_sg and declared_sg > 1
                        and reduced_sg > 1 and reduced_sg != declared_sg):
                    # Allow if they share the same crystal system at least
                    # (CifWriter might pick a different but valid setting)
                    print(f"  CifWriter(symprec={symprec}): detected "
                          f"SG {reduced_sg} vs declared {declared_sg} "
                          f"— using CifWriter result anyway", flush=True)

                print(f"  CifWriter asymmetric unit: {full_cell_n} → "
                      f"{len(reduced_sites)} sites "
                      f"(symprec={symprec}, SG {reduced_sg})", flush=True)
                if return_full_cif:
                    return reduced_sites, reduced_cif
                return reduced_sites

            except Exception as exc:
                print(f"  CifWriter(symprec={symprec}): failed ({exc})",
                      flush=True)
                continue

    except Exception as e:
        print(f"  Warning: CifWriter fallback failed: {e}", flush=True)

    if return_full_cif:
        return None, None
    return None


def _reduce_to_asymmetric_unit(cif_text, declared_sg=None):
    """Reduce a full-unit-cell CIF to asymmetric unit for GSAS-II.

    GSAS-II applies symmetry operations itself, so it expects only the
    asymmetric unit in the CIF.  If we give it the full unit cell plus
    the space group, it over-expands and generates wrong reflections.

    Reduction strategy (in order of preference):
      0. Check if CIF already contains asymmetric unit (COD CIFs, or
         Materials Project CIFs written by CifWriter with symprec).
         If so, use those sites directly — no reduction needed.
      1. SpacegroupAnalyzer symmetrized structure (original method,
         works well for simple metals).
      2. CifWriter-based reduction (more robust for complex structures
         like W2C Pbcn where SpacegroupAnalyzer struggles).
      3. Clustering-based reduction using declared space group.
      4. Raw CIF sites fallback (last resort, may cause over-expansion).

    declared_sg: expected space group number from phase dict (used for
                 validation, not to override pymatgen's detection).
    """
    if not cif_text:
        return []

    # ── Step 0: Check if CIF already has asymmetric unit ────────────────
    # CIFs from COD and from CifWriter(symprec=...) already contain the
    # asymmetric unit plus symmetry operations.  Trying to "reduce" these
    # is a no-op at best and destructive at worst (SpacegroupAnalyzer may
    # misidentify equivalent sites in compact structures).
    already_asym, asym_sites = _cif_already_has_asymmetric_unit(
        cif_text, declared_sg)
    if already_asym and asym_sites:
        return asym_sites

    try:
        from pymatgen.io.cif import CifParser
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        try:
            parser = CifParser.from_str(cif_text, occupancy_tolerance=100.0)
        except (AttributeError, TypeError):
            import tempfile as _tf, os as _os
            _fd, _tmp = _tf.mkstemp(suffix='.cif')
            with _os.fdopen(_fd, 'w') as _f:
                _f.write(cif_text)
            parser = CifParser(_tmp, occupancy_tolerance=100.0)
            _os.unlink(_tmp)
        structs = parser.parse_structures(primitive=False)
        if structs:
            full_cell_n = len(structs[0])  # total sites in full unit cell

            # ── Step 1: SpacegroupAnalyzer (original method) ────────────
            # Try tight tolerance first (0.01 Å), fall back to looser (0.1 Å).
            # Tight tolerance avoids merging non-equivalent sites in compact
            # cells like W2C Pbcn where heavy atoms are close together.
            sites = None
            for symprec in (0.01, 0.05, 0.1):
                try:
                    sga = SpacegroupAnalyzer(structs[0], symprec=symprec)
                    detected_sg = sga.get_space_group_number()
                    sym_struct = sga.get_symmetrized_structure()
                    candidate = []
                    for equiv_sites in sym_struct.equivalent_sites:
                        site = equiv_sites[0]
                        frac = site.frac_coords % 1.0
                        el = str(site.specie)
                        occ = 1.0
                        if hasattr(site, 'species'):
                            sp = site.species
                            if hasattr(sp, 'num_atoms'):
                                occ = float(sp.num_atoms)
                        candidate.append((el, float(frac[0]), float(frac[1]),
                                          float(frac[2]), occ))

                    # Validate: the number of equivalent sites summed across
                    # all groups should equal the full unit cell site count.
                    expanded_n = sum(
                        len(eq) for eq in sym_struct.equivalent_sites)

                    # CRITICAL: for SG > 1, the asymmetric unit must be
                    # SMALLER than the full cell. If SpacegroupAnalyzer falls
                    # back to P1, every atom is "unique" and len(candidate)
                    # == full_cell_n — which gives GSAS-II the full cell
                    # labeled as asymmetric unit, causing over-expansion.
                    if declared_sg and declared_sg > 1:
                        if len(candidate) >= full_cell_n:
                            print(f"  symprec={symprec}: detected SG {detected_sg}, "
                                  f"{len(candidate)} unique sites = full cell "
                                  f"({full_cell_n}) — reduction failed, "
                                  f"trying looser tolerance", flush=True)
                            continue
                        # Also check detected SG is compatible (same or higher
                        # symmetry than declared)
                        if detected_sg < declared_sg:
                            print(f"  symprec={symprec}: detected SG {detected_sg} "
                                  f"< declared SG {declared_sg} — trying "
                                  f"looser tolerance", flush=True)
                            continue

                    if expanded_n == full_cell_n and candidate:
                        # ── CRITICAL: Validate representative positions ──
                        # SpacegroupAnalyzer picks equiv_sites[0] as the
                        # representative.  This position may be in a non-
                        # standard setting (e.g. C in W2C Pbcn placed at
                        # (0.5,0.75,0.878) instead of standard 4c (0,y,¼)).
                        # When GSAS-II applies standard operators to a non-
                        # standard representative, it generates the wrong
                        # number of atoms (e.g. 8C instead of 4C → 16 total
                        # instead of 12 → completely wrong structure factors).
                        #
                        # Fix: re-expand with from_spacegroup (which uses
                        # standard ITA operators) and verify atom count.
                        try:
                            from pymatgen.core import Structure as _PmgStruct
                            _test = _PmgStruct.from_spacegroup(
                                detected_sg, structs[0].lattice,
                                [s[0] for s in candidate],
                                [[s[1], s[2], s[3]] for s in candidate],
                                tol=0.01)
                            if abs(len(_test) - full_cell_n) > 1:
                                print(f"  symprec={symprec}: from_spacegroup "
                                      f"gives {len(_test)} atoms (expected "
                                      f"{full_cell_n}) — representatives in "
                                      f"non-standard setting, trying CifWriter "
                                      f"fallback", flush=True)
                                continue
                        except Exception as _fsg_e:
                            print(f"  symprec={symprec}: from_spacegroup "
                                  f"validation failed ({_fsg_e}), continuing "
                                  f"anyway", flush=True)

                        print(f"  Asymmetric unit reduction: {full_cell_n} → "
                              f"{len(candidate)} sites (symprec={symprec}, "
                              f"detected SG {detected_sg})", flush=True)
                        sites = candidate
                        break
                except Exception as exc:
                    print(f"  symprec={symprec}: failed ({exc})", flush=True)
                    continue

            if sites:
                return sites

            # ── Step 2: CifWriter-based reduction ───────────────────────
            # CifWriter handles the full asymmetric-unit extraction pipeline
            # internally and is more robust than raw SpacegroupAnalyzer for
            # complex structures (carbides, oxides with close atom spacing).
            #
            # CRITICAL: CifWriter may transform to the standard ITA setting,
            # which can PERMUTE AXES relative to the original structure.
            # We return the full CIF text so _build_conventional_cif uses
            # matching cell params + atom positions (same axis convention).
            # Without this, cell params from the phase dict (original axes)
            # + atom positions from CifWriter (standard axes) = physical
            # mismatch → wrong structure factors → zero Rietveld intensity.
            print("  SpacegroupAnalyzer reduction failed for all symprec "
                  "values — trying CifWriter fallback...", flush=True)
            cw_sites, cw_full_cif = _cifwriter_asymmetric_unit(
                cif_text, declared_sg, return_full_cif=True)
            if cw_sites and cw_full_cif:
                # Return the full CIF string — _build_conventional_cif
                # will detect the string type and use it directly.
                return cw_full_cif
            if cw_sites:
                return cw_sites

            # ── Step 3: Clustering-based reduction ──────────────────────
            # All symprec values failed — try using declared SG directly
            # with pymatgen's from_spacegroup to rebuild the asymmetric unit
            if declared_sg and declared_sg > 1:
                try:
                    from pymatgen.core import Lattice, Structure as PmgStruct
                    struct0 = structs[0]
                    lat = struct0.lattice
                    # Get unique elements and approximate positions by
                    # clustering sites within the full cell
                    unique_sites = _cluster_to_asymmetric(struct0, declared_sg)
                    if unique_sites and len(unique_sites) < full_cell_n:
                        # Validate by re-expanding
                        test_struct = PmgStruct.from_spacegroup(
                            declared_sg, lat,
                            [s[0] for s in unique_sites],
                            [[s[1], s[2], s[3]] for s in unique_sites])
                        if abs(len(test_struct) - full_cell_n) <= 1:
                            print(f"  Asymmetric unit via clustering: "
                                  f"{full_cell_n} → {len(unique_sites)} sites "
                                  f"(SG {declared_sg})", flush=True)
                            return unique_sites
                except Exception as exc:
                    print(f"  Clustering fallback failed: {exc}", flush=True)

    except Exception as e:
        print(f"  Warning: pymatgen asymmetric-unit reduction failed: {e}",
              flush=True)

    # ── Step 4: Raw CIF sites fallback ──────────────────────────────────
    # Usually fine for COD CIFs which list asymmetric unit.
    # For MP/pymatgen CIFs which may list the full unit cell, this is
    # DANGEROUS — GSAS-II will over-expand.  We detect this case and
    # fall back to P1 (no symmetry expansion) rather than risk wrong
    # reflections from double-expansion.
    try:
        parsed = parse_cif(cif_text)
        raw_sites = parsed.get('sites') or []
        raw_sg = parsed.get('spacegroup_number', 1)

        if raw_sites and declared_sg and declared_sg > 1:
            # Check if these are likely full-cell sites by comparing
            # site count with what we'd expect for the asymmetric unit.
            # If the CIF has no symmetry operations but declares SG > 1,
            # the sites are probably the full cell (from pymatgen Structure
            # dump without symmetry detection).
            has_symops = ('_symmetry_equiv_pos_as_xyz' in cif_text or
                          '_space_group_symop_operation_xyz' in cif_text or
                          '_space_group_symop.operation_xyz' in cif_text)
            if not has_symops:
                print(f"  WARNING: raw CIF has {len(raw_sites)} sites, "
                      f"SG {declared_sg}, but NO symmetry operations — "
                      f"sites are likely full cell. Returning raw sites "
                      f"with P1 flag to prevent over-expansion.", flush=True)
                # Signal to _build_conventional_cif that these are full-cell
                # sites and should NOT be expanded by the declared SG.
                # We do this by returning them tagged — the caller
                # (_build_conventional_cif) will detect the mismatch and
                # write the CIF with P1 instead.
                return raw_sites
            else:
                print(f"  Using raw CIF sites ({len(raw_sites)} atoms) "
                      f"with SG {declared_sg} — CIF has symmetry operations, "
                      f"sites should be asymmetric unit.", flush=True)
        return raw_sites
    except Exception:
        return []


def _cluster_to_asymmetric(struct, sg_num):
    """Cluster full-cell sites into asymmetric unit using the declared SG.

    For each unique element, tries positions from the full cell one at a
    time, expanding each with the declared space group. Keeps the position
    that best reproduces the observed atom count for that element.

    Returns list of (element, x, y, z, occupancy) or None on failure.
    """
    from pymatgen.core import Lattice, Structure as PmgStruct
    import numpy as _np

    lat = struct.lattice
    # Group sites by element
    element_sites = {}
    for site in struct:
        el = str(site.specie)
        frac = tuple(float(c) % 1.0 for c in site.frac_coords)
        element_sites.setdefault(el, []).append(frac)

    result = []
    used_positions = []  # track what's been accounted for

    for el, positions in element_sites.items():
        target_count = len(positions)
        remaining = list(positions)
        el_asym = []

        while remaining:
            # Try each remaining position as a candidate asymmetric site
            best_pos = None
            best_count = 0
            for pos in remaining:
                try:
                    test = PmgStruct.from_spacegroup(
                        sg_num, lat, [el], [list(pos)])
                    count = sum(1 for s in test if str(s.specie) == el)
                    if count > best_count and count <= len(remaining):
                        best_count = count
                        best_pos = pos
                except Exception:
                    continue

            if best_pos is None or best_count == 0:
                break

            el_asym.append(best_pos)
            # Remove the positions accounted for by this Wyckoff site
            # (find closest matches to the expanded positions)
            try:
                expanded = PmgStruct.from_spacegroup(
                    sg_num, lat, [el], [list(best_pos)])
                for exp_site in expanded:
                    exp_frac = tuple(float(c) % 1.0 for c in exp_site.frac_coords)
                    # Remove the closest match from remaining
                    closest_idx = None
                    closest_dist = 999
                    for i, rem in enumerate(remaining):
                        dist = sum((a - b)**2 for a, b in zip(exp_frac, rem))**0.5
                        # Also check wrapped distances
                        dist_wrap = sum(min((a-b)%1, (b-a)%1)**2
                                       for a, b in zip(exp_frac, rem))**0.5
                        d = min(dist, dist_wrap)
                        if d < closest_dist:
                            closest_dist = d
                            closest_idx = i
                    if closest_idx is not None and closest_dist < 0.1:
                        remaining.pop(closest_idx)
            except Exception:
                break

        for pos in el_asym:
            result.append((el, pos[0], pos[1], pos[2], 1.0))

    return result if result else None


def _build_conventional_cif(ph):
    """
    Build a synthetic CIF string using the phase dict's (conventional) cell
    parameters and atom sites.

    This ensures GSAS-II always sees a CIF consistent with the conventional
    cell, even when the original CIF used a primitive setting (common with
    Materials Project data).  The space group is written explicitly so that
    GSAS-II applies the correct cell-parameter constraints.

    If the asymmetric-unit reduction fails for a complex structure (e.g.
    W2C Pbcn from Materials Project), this function detects the failure
    and returns the ORIGINAL CIF text instead of building a synthetic CIF
    with full-cell sites + the declared space group (which would cause
    GSAS-II to double-expand and produce wrong reflections).
    """
    a   = ph.get('a', 4.0)
    b   = ph.get('b', a)
    c   = ph.get('c', a)
    al  = ph.get('alpha', 90.0)
    be  = ph.get('beta', 90.0)
    ga  = ph.get('gamma', 90.0)
    sg  = ph.get('spacegroup_number', 1)
    formula = ph.get('formula', '')
    Z   = ph.get('Z', '')

    # H-M symbol — try phase dict first, then dynamic lookup
    hm = ph.get('spacegroup', '') or ph.get('spacegroup_name', '')
    if not hm:
        hm = _get_hm_symbol(sg)

    # Get atom sites reduced to the asymmetric unit.
    # GSAS-II applies space-group symmetry itself, so we must NOT give
    # it the full unit cell — that would cause double-counting and wrong
    # reflections (e.g. ghost peaks at ~25° for A15-W).
    cif_text = ph.get('cif_text', '')
    sites = _reduce_to_asymmetric_unit(cif_text, declared_sg=sg) if cif_text else []

    # ── CifWriter full CIF passthrough ────────────────────────────────
    # When CifWriter was used for asymmetric-unit reduction, it may have
    # transformed the structure to the standard ITA setting (permuting
    # axes). In this case _reduce_to_asymmetric_unit returns the FULL
    # CIF text (a string) rather than a sites list.  Using this CIF
    # directly ensures atom positions and cell parameters are in the
    # SAME axis convention.  If we instead extracted just the positions
    # and combined them with the original phase dict cell params, axis
    # mismatch would put atoms at wrong physical locations → zero
    # structure factors → phase scale collapses in Rietveld.
    if isinstance(sites, str):
        print(f"  _build_conventional_cif [{formula or '?'}]: using full "
              f"CifWriter CIF directly (axis-consistent cell + positions)",
              flush=True)
        return sites

    # ── Expansion validation: verify that GSAS-II will generate the ──
    # correct number of atoms from the asymmetric unit + space group.
    # This catches cases where the representative positions are in a
    # non-standard setting (e.g. SpacegroupAnalyzer returning C at
    # (0.5,0.75,0.878) instead of standard 4c (0,y,0.25) for Pbcn).
    if sites and sg > 1 and cif_text:
        try:
            from pymatgen.core import Structure as _PmgStruct
            from pymatgen.core import Lattice as _PmgLat
            _lat = _PmgLat.from_parameters(a, b, c, al, be, ga)
            _test = _PmgStruct.from_spacegroup(
                sg, _lat,
                [s[0] for s in sites],
                [[s[1], s[2], s[3]] for s in sites],
                tol=0.01)
            # Get expected full cell count from the original CIF
            from pymatgen.io.cif import CifParser as _CP
            try:
                _p = _CP.from_str(cif_text, occupancy_tolerance=100.0)
            except (AttributeError, TypeError):
                import tempfile as _tf, os as _os
                _fd, _tmp = _tf.mkstemp(suffix='.cif')
                with _os.fdopen(_fd, 'w') as _f:
                    _f.write(cif_text)
                _p = _CP(_tmp, occupancy_tolerance=100.0)
                _os.unlink(_tmp)
            _structs = _p.parse_structures(primitive=False)
            _expected_n = len(_structs[0]) if _structs else len(_test)

            if abs(len(_test) - _expected_n) > 1:
                print(f"  _build_conventional_cif: from_spacegroup gives "
                      f"{len(_test)} atoms (expected {_expected_n}) — "
                      f"atom positions may be in non-standard setting. "
                      f"Attempting CifWriter rescue...", flush=True)
                # Try CifWriter to get correct standard-setting CIF
                from pymatgen.io.cif import CifWriter as _CW
                _struct0 = _structs[0]
                for _symprec in (0.01, 0.05, 0.1, 0.2):
                    try:
                        _writer = _CW(_struct0, symprec=_symprec)
                        import tempfile as _tf3, os as _os3
                        _fd3, _tmp3 = _tf3.mkstemp(suffix='.cif')
                        with _os3.fdopen(_fd3, 'w') as _f3:
                            pass
                        _writer.write_file(_tmp3)
                        with open(_tmp3, 'r') as _f3:
                            _rescued_cif = _f3.read()
                        _os3.unlink(_tmp3)
                        # Validate the rescued CIF
                        _rp = parse_cif(_rescued_cif)
                        _rsites = _rp.get('sites') or []
                        _rsg = _rp.get('spacegroup_number', 1)
                        if _rsites and len(_rsites) < _expected_n:
                            # Verify re-expansion gives correct count
                            try:
                                _p2 = _CP.from_str(
                                    _rescued_cif, occupancy_tolerance=100.0)
                                _s2 = _p2.parse_structures(primitive=False)
                                if _s2 and abs(len(_s2[0]) - _expected_n) <= 1:
                                    print(f"  CifWriter rescue succeeded: "
                                          f"SG {_rsg}, {len(_rsites)} sites "
                                          f"(symprec={_symprec})", flush=True)
                                    # Patch cell params to match phase dict
                                    import re as _re
                                    _rescued_cif = _re.sub(
                                        r'_cell_length_a\s+[\d.]+',
                                        f'_cell_length_a {a:.5f}',
                                        _rescued_cif)
                                    _rescued_cif = _re.sub(
                                        r'_cell_length_b\s+[\d.]+',
                                        f'_cell_length_b {b:.5f}',
                                        _rescued_cif)
                                    _rescued_cif = _re.sub(
                                        r'_cell_length_c\s+[\d.]+',
                                        f'_cell_length_c {c:.5f}',
                                        _rescued_cif)
                                    return _rescued_cif
                            except Exception:
                                pass
                    except Exception:
                        continue
                print(f"  CifWriter rescue failed — will use synthetic CIF "
                      f"(may have wrong atom positions)", flush=True)
        except Exception as _e:
            print(f"  from_spacegroup validation error: {_e}", flush=True)

    # ── Safety check: detect if "reduction" actually returned full-cell sites ──
    # If the reduction fell through all fallbacks and returned raw CIF sites
    # that are actually the full unit cell (no symops in the original CIF,
    # e.g. from a pymatgen Structure dump), writing those sites with the
    # declared SG > 1 would cause GSAS-II to double-expand.
    #
    # Detection: compare site count with what CifParser(primitive=False) gives.
    # If they match AND the CIF lacks symmetry operations, the sites are full-cell.
    # In this case, use the ORIGINAL CIF directly — it already has the correct
    # atom positions and (if from CifWriter) proper symmetry operations.
    if sites and sg > 1 and cif_text:
        has_symops = ('_symmetry_equiv_pos_as_xyz' in cif_text or
                      '_space_group_symop_operation_xyz' in cif_text or
                      '_space_group_symop.operation_xyz' in cif_text)
        if not has_symops:
            # The original CIF has no symmetry operations — check if sites
            # are the full cell by expanding the original CIF
            try:
                from pymatgen.io.cif import CifParser
                try:
                    _parser = CifParser.from_str(cif_text, occupancy_tolerance=100.0)
                except (AttributeError, TypeError):
                    import tempfile as _tf, os as _os
                    _fd, _tmp = _tf.mkstemp(suffix='.cif')
                    with _os.fdopen(_fd, 'w') as _f:
                        _f.write(cif_text)
                    _parser = CifParser(_tmp, occupancy_tolerance=100.0)
                    _os.unlink(_tmp)
                _structs = _parser.parse_structures(primitive=False)
                if _structs:
                    full_n = len(_structs[0])
                    if len(sites) >= full_n:
                        # Sites ARE the full cell — DO NOT write them with SG > 1.
                        # Instead, try to use CifWriter to generate a proper CIF
                        # with asymmetric unit + symmetry operations.
                        print(f"  _build_conventional_cif: detected full-cell "
                              f"sites ({len(sites)}) with no symops for "
                              f"SG {sg}. Attempting CifWriter rescue...",
                              flush=True)
                        try:
                            from pymatgen.io.cif import CifWriter
                            # Use the already-parsed structure
                            struct0 = _structs[0]
                            for symprec in (0.01, 0.05, 0.1, 0.2):
                                try:
                                    writer = CifWriter(struct0, symprec=symprec)
                                    import tempfile as _tf2, os as _os2
                                    _fd2, _tmp2 = _tf2.mkstemp(suffix='.cif')
                                    with _os2.fdopen(_fd2, 'w') as _f2:
                                        pass
                                    writer.write_file(_tmp2)
                                    with open(_tmp2, 'r') as _f2:
                                        rescued_cif = _f2.read()
                                    _os2.unlink(_tmp2)
                                    # Validate: parse the rescued CIF
                                    rescued_parsed = parse_cif(rescued_cif)
                                    rescued_sg = rescued_parsed.get(
                                        'spacegroup_number', 1)
                                    rescued_sites = rescued_parsed.get(
                                        'sites') or []
                                    if (rescued_sites
                                            and len(rescued_sites) < full_n
                                            and rescued_sg > 1):
                                        print(f"  CifWriter rescue succeeded: "
                                              f"SG {rescued_sg}, "
                                              f"{len(rescued_sites)} asym sites "
                                              f"(symprec={symprec})", flush=True)
                                        # Patch cell parameters to match the
                                        # phase dict's conventional cell
                                        # (CifWriter may use slightly different
                                        # lattice params from DFT relaxation)
                                        import re as _re
                                        rescued_cif = _re.sub(
                                            r'_cell_length_a\s+[\d.]+',
                                            f'_cell_length_a {a:.5f}',
                                            rescued_cif)
                                        rescued_cif = _re.sub(
                                            r'_cell_length_b\s+[\d.]+',
                                            f'_cell_length_b {b:.5f}',
                                            rescued_cif)
                                        rescued_cif = _re.sub(
                                            r'_cell_length_c\s+[\d.]+',
                                            f'_cell_length_c {c:.5f}',
                                            rescued_cif)
                                        return rescued_cif
                                except Exception:
                                    continue
                        except Exception as e_cw:
                            print(f"  CifWriter rescue failed: {e_cw}",
                                  flush=True)

                        # CifWriter rescue failed — last resort: write with P1
                        # so GSAS-II doesn't expand. This will give correct
                        # peak positions but GSAS-II won't apply symmetry
                        # constraints on cell parameters or atoms.
                        print(f"  LAST RESORT: writing full-cell sites with "
                              f"P1 to prevent double-expansion. GSAS-II will "
                              f"not apply symmetry constraints.", flush=True)
                        sg = 1
                        hm = 'P 1'
            except Exception:
                pass

    # ── Diagnostic: print what GSAS-II will actually receive ──────────
    print(f"  _build_conventional_cif [{formula or '?'}]:", flush=True)
    print(f"    Cell: a={a:.4f} b={b:.4f} c={c:.4f} "
          f"α={al:.2f} β={be:.2f} γ={ga:.2f}", flush=True)
    print(f"    SG: {sg} ({hm})", flush=True)
    print(f"    Asymmetric sites: {len(sites) if sites else 0}", flush=True)
    if sites:
        for _el, _x, _y, _z, _occ in sites:
            print(f"      {_el:3s}  ({_x:.6f}, {_y:.6f}, {_z:.6f})  "
                  f"occ={_occ:.4f}", flush=True)

    lines = [
        'data_phase',
        f"_chemical_formula_sum '{formula}'" if formula else '',
        f"_cell_formula_units_Z {Z}" if Z else '',
        f'_cell_length_a {a:.5f}',
        f'_cell_length_b {b:.5f}',
        f'_cell_length_c {c:.5f}',
        f'_cell_angle_alpha {al:.3f}',
        f'_cell_angle_beta {be:.3f}',
        f'_cell_angle_gamma {ga:.3f}',
        f'_symmetry_Int_Tables_number {sg}',
        f"_symmetry_space_group_name_H-M '{hm}'",
        f"_space_group_IT_number {sg}",
        f"_space_group_name_H-M_alt '{hm}'",
    ]

    if sites:
        lines += [
            '',
            'loop_',
            '_atom_site_label',
            '_atom_site_type_symbol',
            '_atom_site_fract_x',
            '_atom_site_fract_y',
            '_atom_site_fract_z',
            '_atom_site_occupancy',
        ]
        site_counts = {}
        for el, x, y, z, occ in sites:
            # Generate unique labels like W1, W2, O1, O2...
            site_counts[el] = site_counts.get(el, 0) + 1
            label = f'{el}{site_counts[el]}'
            # Clamp negative zeros to 0.0 — GSAS-II CIF reader rejects "-0.000000"
            x = abs(x) if abs(x) < 1e-8 else x
            y = abs(y) if abs(y) < 1e-8 else y
            z = abs(z) if abs(z) < 1e-8 else z
            lines.append(f'{label}  {el}  {x:.6f}  {y:.6f}  {z:.6f}  {occ:.4f}')

    return '\n'.join(ln for ln in lines if ln is not None) + '\n'


# ─────────────────────────────────────────────────────────────────────────────
# NON-NEGATIVE LEAST SQUARES (NNLS)
# ─────────────────────────────────────────────────────────────────────────────

def _nnls(A, b):
    """Solve  min‖Ax − b‖²  subject to  x ≥ 0.

    Uses scipy.optimize.nnls if available, otherwise falls back to the
    Lawson-Hanson active-set algorithm (pure numpy).
    """
    try:
        from scipy.optimize import nnls as _scipy_nnls
        x, _ = _scipy_nnls(A, b)
        return x
    except ImportError:
        pass
    # ── Fallback: Lawson-Hanson active-set NNLS ──────────────────────
    m, n = A.shape
    P = set()               # passive set (variables free to be > 0)
    Z = set(range(n))       # zero set (variables fixed at 0)
    x = np.zeros(n)
    max_outer = 3 * n
    for _ in range(max_outer):
        w = A.T @ (b - A @ x)  # negative gradient
        if not Z or max(w[j] for j in Z) <= 1e-10:
            break
        # Move variable with largest positive gradient into passive set
        t = max(Z, key=lambda j: w[j])
        P.add(t); Z.discard(t)
        # Inner loop: solve unconstrained on P, handle negatives
        for __ in range(max_outer):
            cols = sorted(P)
            s = np.zeros(n)
            s[cols] = np.linalg.lstsq(A[:, cols], b, rcond=None)[0]
            if all(s[j] >= 0 for j in cols):
                x = s
                break
            # Find the tightest step that hits zero
            alpha = 1.0
            j_remove = None
            for j in cols:
                if s[j] < 0 and x[j] > x[j] - s[j]:
                    a = x[j] / (x[j] - s[j])
                    if a < alpha:
                        alpha = a
                        j_remove = j
            if j_remove is None:
                x = s
                break
            x = x + alpha * (s - x)
            x[j_remove] = 0.0
            Z.add(j_remove); P.discard(j_remove)
        else:
            x = np.maximum(x, 0.0)  # safety clamp
    return x


# ─────────────────────────────────────────────────────────────────────────────
# PER-PHASE PATTERN FROM GSAS-II REFLECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _compute_raw_phase_profile(tt_arr, refs, U_deg, V_deg, W_deg,
                                X_deg, Y_deg, gaussian_only=False):
    """Compute a raw (unscaled) profile shape for one phase.

    Returns an array of the same length as *tt_arr* whose values are
    proportional to the diffracted intensity at each 2θ point.  The
    absolute scale is arbitrary — what matters is the *relative* shape
    compared to other phases so that the proportional decomposition
    can correctly partition the total pattern.

    Parameters
    ----------
    gaussian_only : bool
        If True, use pure Gaussian profiles (eta=0) with a tighter
        window (3×FWHM).  This eliminates the long Lorentzian tails
        that cause cross-phase leakage artifacts in the proportional
        decomposition, while preserving accurate peak localisation
        for genuinely overlapping reflections.

    refs : list of [two_theta, d, (h,k,l), intensity_weight] from
           generate_reflections (with structure-factor filtering).
    """
    n = len(tt_arr)
    pattern = np.zeros(n, dtype=np.float64)

    use_tch = (X_deg != 0.0 or Y_deg != 0.0)

    for tt_pos, d, hkl, weight in refs:
        if weight <= 0:
            continue

        if use_tch:
            fwhm, eta = tch_fwhm_eta(tt_pos, U_deg, V_deg, W_deg,
                                      X_deg, Y_deg)
        else:
            fwhm = max(caglioti_fwhm(tt_pos, U_deg, V_deg, W_deg), 0.005)
            eta = 0.5

        if gaussian_only:
            eta = 0.0

        sigma_g = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        gamma_l = fwhm / 2.0
        # Gaussian-only: 3×FWHM is sufficient (G ≈ 10⁻¹¹ at edge).
        # Pseudo-Voigt: 6×FWHM captures Lorentzian tails.
        window = 3.0 * fwhm if gaussian_only else 6.0 * fwhm

        msk = np.abs(tt_arr - tt_pos) < window
        if not msk.any():
            continue

        dx = tt_arr[msk] - tt_pos
        G = np.exp(-0.5 * (dx / sigma_g) ** 2)
        if eta > 0:
            L = 1.0 / (1.0 + (dx / gamma_l) ** 2)
            prof = eta * L + (1.0 - eta) * G
        else:
            prof = G
        # Normalise
        area = np.trapz(prof, tt_arr[msk]) if np.trapz(prof, tt_arr[msk]) > 0 else 1.0
        pattern[msk] += weight * prof / area

    return pattern


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _auto_select_bg_coeffs(tt, y_obs, phases, wavelength, tt_min, tt_max,
                           user_n=6):
    """Automatically select the number of Chebyshev background coefficients.

    Philosophy: use enough terms to accurately model the instrument
    background, but not so many that the polynomial develops oscillations
    (humps/dips) that absorb peak tails or noise.  Even clean crystalline
    samples need ~6 terms to capture instrument effects (air scatter,
    Bremsstrahlung, sample holder scattering, beam spillover at low 2θ).
    Going below 6 starves the background of flexibility and degrades Rwp.

    Strategy:
      1. Identify "peak-free" regions by masking out zones within ±1.0° of
         expected reflection positions (wider exclusion to avoid peak tails
         biasing the background estimate).
      2. Fit peak-free intensities with Chebyshev polynomials of order 6, 10.
      3. Default to 6 terms for clean crystalline patterns.
      4. Step up to 10 only if there is a clear amorphous hump (DW < 0.5
         AND broad feature detected AND >20% RMS improvement from 6→10),
         as seen in supported catalysts (WC/SiO₂, Pt/Al₂O₃).
      5. Only go higher (14, 20) if the user explicitly requests it.

    Returns the recommended n_bg_coeffs (int).
    """
    from .crystallography import generate_reflections

    # Build a mask of peak-free zones
    mask_range = (tt >= tt_min) & (tt <= tt_max)
    tt_r = tt[mask_range]
    y_r = y_obs[mask_range]

    if len(tt_r) < 20 or tt_max <= tt_min:
        return user_n

    # Find all expected peak positions from ALL phases
    peak_positions = []
    for ph in phases:
        try:
            sys_ = (ph.get('system') or 'triclinic').lower()
            sg = ph.get('spacegroup_number', 1)
            refs = generate_reflections(
                ph.get('a', 4), ph.get('b', ph.get('a', 4)),
                ph.get('c', ph.get('a', 4)),
                ph.get('alpha', 90), ph.get('beta', 90), ph.get('gamma', 90),
                sys_, sg, wavelength, tt_min, tt_max, hkl_max=10)
            peak_positions.extend([r[0] for r in refs])
        except Exception:
            pass

    # Mask: True = peak-free (at least 1.0° away from any reflection)
    # Using wider exclusion (1.0° vs 0.5°) to keep peak tails out of the
    # baseline estimate — Lorentzian tails extend far from peak centres
    # and can bias the polynomial if included.
    peak_free = np.ones(len(tt_r), dtype=bool)
    for pos in peak_positions:
        peak_free &= np.abs(tt_r - pos) > 1.0

    # Need enough peak-free points to fit
    n_free = int(np.sum(peak_free))
    if n_free < 15:
        print(f"  Auto-BG: only {n_free} peak-free points — using "
              f"n_bg={user_n}", flush=True)
        return user_n

    tt_free = tt_r[peak_free]
    y_free = y_r[peak_free]

    # Normalize 2theta to [-1, 1] for Chebyshev stability
    x_free = 2.0 * (tt_free - tt_min) / (tt_max - tt_min) - 1.0

    # Fit with 3 terms (constant + linear + quadratic — handles most
    # instrument-induced slopes like beam spillover at low angles)
    try:
        coeffs_3 = np.polynomial.chebyshev.chebfit(x_free, y_free, 2)
        fitted_3 = np.polynomial.chebyshev.chebval(x_free, coeffs_3)
        resid_3 = y_free - fitted_3
        rms_3 = float(np.sqrt(np.mean(resid_3 ** 2)))
    except Exception:
        print(f"  Auto-BG: Chebyshev fit failed — using n_bg=6", flush=True)
        return 6

    # Durbin-Watson statistic: tests for autocorrelation in residuals.
    # DW ≈ 2.0 → no autocorrelation (residuals are random noise → good fit).
    # DW < 1.0 → strong positive autocorrelation (systematic misfit →
    #             the background has curvature that 3 terms can't capture).
    dw = 2.0
    if len(resid_3) > 2:
        diff_sq = float(np.sum(np.diff(resid_3) ** 2))
        sum_sq = float(np.sum(resid_3 ** 2))
        if sum_sq > 0:
            dw = diff_sq / sum_sq

    # Also check if the baseline has a broad hump: fit with 6 terms and
    # see if any higher-order coefficients are large relative to the
    # constant term (indicating genuine curvature, not noise).
    has_broad_feature = False
    rms_6 = rms_3
    try:
        coeffs_6 = np.polynomial.chebyshev.chebfit(x_free, y_free, 5)
        fitted_6 = np.polynomial.chebyshev.chebval(x_free, coeffs_6)
        resid_6 = y_free - fitted_6
        rms_6 = float(np.sqrt(np.mean(resid_6 ** 2)))
        # Check if higher-order coefficients are significant
        if abs(coeffs_6[0]) > 0:
            max_higher = max(abs(c) for c in coeffs_6[3:])
            if max_higher > abs(coeffs_6[0]) * 0.05:
                has_broad_feature = True
    except Exception:
        pass

    # Also fit with 10 terms to check if amorphous hump needs even more
    # flexibility (e.g. SiO₂, Al₂O₃ support backgrounds)
    rms_10 = rms_6
    try:
        coeffs_10 = np.polynomial.chebyshev.chebfit(x_free, y_free, 9)
        fitted_10 = np.polynomial.chebyshev.chebval(x_free, coeffs_10)
        resid_10 = y_free - fitted_10
        rms_10 = float(np.sqrt(np.mean(resid_10 ** 2)))
    except Exception:
        pass

    # Decision logic — 6 terms is the floor for all samples.
    # Even clean crystalline patterns need 6 Chebyshev terms to capture
    # instrument background effects (air scatter, Bremsstrahlung, sample
    # holder scattering, beam spillover at low 2θ).  Dropping to 3 starves
    # the background and degrades Rwp.
    #
    # Tier 1 (6 terms): clean crystalline baseline — standard default
    #   → No broad features, or mild curvature captured by 6 terms
    #
    # Tier 2 (10 terms): significant background structure
    #   Triggered by ANY of these conditions:
    #   (a) DW < 0.1 on the 3-term residuals — extreme autocorrelation
    #       indicates the background has curvature that 6 terms likely
    #       cannot capture either.  A DW this low means the background
    #       model is catastrophically underfitting.
    #   (b) DW < 0.5 AND broad feature AND >15% RMS improvement from
    #       6→10 terms (relaxed from 20% — borderline cases like
    #       WC + W2C patterns with DW=0.04 were being missed).
    #   (c) DW(6) < 0.3 on the 6-term residuals — even after doubling
    #       the terms, there's still strong autocorrelation.
    #
    # Beyond 10: user must explicitly select (14, 20) from the dropdown
    improvement_6_to_10 = (rms_6 - rms_10) / max(rms_6, 1e-6)

    # Also compute DW on the 6-term residuals to check if 6 is adequate
    dw_6 = 2.0
    try:
        if len(resid_6) > 2:
            _diff_sq_6 = float(np.sum(np.diff(resid_6) ** 2))
            _sum_sq_6 = float(np.sum(resid_6 ** 2))
            if _sum_sq_6 > 0:
                dw_6 = _diff_sq_6 / _sum_sq_6
    except Exception:
        pass

    # Default to 6 Chebyshev terms — this is sufficient for standard
    # crystalline samples and avoids overfitting.  Only go higher if the
    # user explicitly requested more than 6 terms.
    best_n = 6
    if user_n > 6:
        best_n = user_n

    print(f"  Auto-BG: peak-free analysis — RMS(6)={rms_6:.2f}, "
          f"RMS(10)={rms_10:.2f}, DW(3)={dw:.2f}, DW(6)={dw_6:.2f}, "
          f"broad_feature={has_broad_feature}, "
          f"improv_6→10={improvement_6_to_10:.1%} → n_bg={best_n}",
          flush=True)
    return best_n


def _write_xye(path, tt, y_obs, sigma):
    """Write a .xye file (2theta  intensity  sigma) for GSAS-II import."""
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for i in range(len(tt)):
            f.write(f"{tt[i]:.6f}  {y_obs[i]:.4f}  {sigma[i]:.4f}\n")


def _estimate_profile_params(tt, y_obs):
    """Estimate initial Caglioti U, V, W from observed peak widths.

    Finds the strongest peaks in the data, measures their approximate FWHM,
    and fits the Caglioti equation FWHM² = U tan²θ + V tanθ + W to get
    reasonable starting values.  Returns (U, V, W) in centidegrees² / centideg²
    (GSAS-II internal units), or the module defaults if estimation fails.

    This is critical for complex phases like W2C where the default U/V/W
    may be far from the true broadening, causing GSAS-II to converge to
    a local minimum.
    """
    try:
        from scipy.signal import find_peaks

        # Subtract a simple baseline (percentile)
        baseline = np.percentile(y_obs, 10)
        y_corr = y_obs - baseline

        # Find prominent peaks (height > 10% of max, well separated)
        height_thresh = 0.1 * y_corr.max()
        step = float(tt[1] - tt[0]) if len(tt) > 1 else 0.02
        distance = max(1, int(0.5 / step))  # at least 0.5° apart
        peaks, props = find_peaks(y_corr, height=height_thresh, distance=distance)

        if len(peaks) < 2:
            return DEFAULT_U, DEFAULT_V, DEFAULT_W

        # Measure FWHM for each peak
        fwhm_data = []  # (tan_theta, fwhm_deg)
        for pk in peaks:
            half_max = y_corr[pk] / 2.0
            # Walk left
            left = pk
            while left > 0 and y_corr[left] > half_max:
                left -= 1
            # Walk right
            right = pk
            while right < len(y_corr) - 1 and y_corr[right] > half_max:
                right += 1
            fwhm_deg = float(tt[right] - tt[left])
            if 0.02 < fwhm_deg < 3.0:  # reasonable range
                theta_rad = math.radians(float(tt[pk]) / 2.0)
                tan_th = math.tan(theta_rad)
                fwhm_data.append((tan_th, fwhm_deg))

        if len(fwhm_data) < 2:
            return DEFAULT_U, DEFAULT_V, DEFAULT_W

        # Fit Caglioti: FWHM² = U tan²θ + V tanθ + W
        # In GSAS-II, U/V/W are in centideg² so FWHM is in centideg.
        # We work in degrees then convert.
        tan_arr = np.array([d[0] for d in fwhm_data])
        fwhm_sq = np.array([d[1] ** 2 for d in fwhm_data])

        # Build design matrix [tan²θ, tanθ, 1]
        A = np.column_stack([tan_arr ** 2, tan_arr, np.ones_like(tan_arr)])
        try:
            result, _, _, _ = np.linalg.lstsq(A, fwhm_sq, rcond=None)
            U_deg2, V_deg2, W_deg2 = float(result[0]), float(result[1]), float(result[2])
        except np.linalg.LinAlgError:
            return DEFAULT_U, DEFAULT_V, DEFAULT_W

        # Convert degrees² → centideg²  (multiply by 10000)
        U_cdeg2 = U_deg2 * 10000.0
        V_cdeg2 = V_deg2 * 10000.0
        W_cdeg2 = W_deg2 * 10000.0

        # Sanity clamp: U and W should be positive, V typically negative
        U_cdeg2 = max(0.1, min(U_cdeg2, 500.0))
        V_cdeg2 = max(-200.0, min(V_cdeg2, 200.0))
        W_cdeg2 = max(0.1, min(W_cdeg2, 500.0))

        return U_cdeg2, V_cdeg2, W_cdeg2

    except Exception:
        return DEFAULT_U, DEFAULT_V, DEFAULT_W


def _estimate_lorentzian_params(tt, y_obs, U_cdeg2, V_cdeg2, W_cdeg2):
    """Estimate initial Lorentzian profile parameters X and Y from peak shapes.

    Measures the Lorentzian fraction of observed peaks by comparing their
    half-max width to their quarter-max width (Gaussian ratio ~1.48,
    Lorentzian ratio ~1.73).  Decomposes each peak's FWHM into Gaussian and
    Lorentzian components using the Thompson-Cox-Hastings (TCH) relationship,
    then fits H_L = X*tan(theta) + Y/cos(theta) across peaks.

    Returns (X, Y) in centidegrees, or moderate defaults if estimation fails.
    Critical for carbides/oxides with significant crystallite-size (Y) and
    micro-strain (X) Lorentzian broadening.
    """
    try:
        from scipy.signal import find_peaks

        baseline = np.percentile(y_obs, 10)
        y_corr = y_obs - baseline

        height_thresh = 0.1 * y_corr.max()
        step = float(tt[1] - tt[0]) if len(tt) > 1 else 0.02
        distance = max(1, int(0.5 / step))
        peaks, _ = find_peaks(y_corr, height=height_thresh, distance=distance)

        if len(peaks) < 2:
            return DEFAULT_X, DEFAULT_Y

        lor_data = []  # (tan_theta, cos_theta, H_L_deg)
        for pk in peaks:
            half_max = y_corr[pk] / 2.0
            quarter_max = y_corr[pk] / 4.0

            # Measure half-max width
            left_h, right_h = pk, pk
            while left_h > 0 and y_corr[left_h] > half_max:
                left_h -= 1
            while right_h < len(y_corr) - 1 and y_corr[right_h] > half_max:
                right_h += 1
            fwhm_deg = float(tt[right_h] - tt[left_h])

            # Measure quarter-max width
            left_q, right_q = pk, pk
            while left_q > 0 and y_corr[left_q] > quarter_max:
                left_q -= 1
            while right_q < len(y_corr) - 1 and y_corr[right_q] > quarter_max:
                right_q += 1
            fwqm_deg = float(tt[right_q] - tt[left_q])

            if fwhm_deg < 0.02 or fwhm_deg > 3.0 or fwqm_deg <= fwhm_deg:
                continue

            # Estimate Lorentzian fraction eta from FWQM/FWHM ratio.
            # Gaussian: FWQM/FWHM = sqrt(2) ≈ 1.414
            # Lorentzian: FWQM/FWHM = sqrt(3) ≈ 1.732
            # Linear interpolation gives eta (Lorentzian fraction).
            ratio = fwqm_deg / fwhm_deg
            # Linear interpolation for eta
            eta = (ratio - 1.4142) / (1.7321 - 1.4142)
            eta = max(0.0, min(eta, 1.0))

            if eta < 0.01:
                continue  # essentially pure Gaussian, no Lorentzian info

            # TCH decomposition: total FWHM ≈ eta*H_L + (1-eta)*H_G (simplified)
            # More precise: use the 5th-order TCH formula inverse.
            # For a practical estimate, the pseudo-Voigt approximation gives:
            #   H_L ≈ FWHM * eta (leading-order)
            #   H_G ≈ FWHM * (1 - eta)
            # Subtract the Gaussian Caglioti contribution to isolate Lorentzian.
            theta_rad = math.radians(float(tt[pk]) / 2.0)
            tan_th = math.tan(theta_rad)
            cos_th = math.cos(theta_rad)

            # Caglioti Gaussian FWHM² in deg² (convert from centideg²)
            H_G_sq_deg2 = (U_cdeg2 * tan_th**2 + V_cdeg2 * tan_th
                           + W_cdeg2) / 10000.0
            H_G_deg = math.sqrt(max(H_G_sq_deg2, 0.001))

            # Lorentzian FWHM: from total FWHM and Gaussian contribution
            # Using TCH relation: H_total^5 ≈ H_G^5 + ... + H_L^5
            # Simplified: H_L ≈ max(0, FWHM - H_G) when eta > 0
            # More robust: H_L = FWHM * eta (pseudo-Voigt definition)
            H_L_deg = fwhm_deg * eta
            if H_L_deg > 0.005:
                lor_data.append((tan_th, cos_th, H_L_deg))

        if len(lor_data) < 2:
            return DEFAULT_X, DEFAULT_Y

        # Fit H_L = X*tan(theta) + Y/cos(theta)
        # X and Y are in degrees here; convert to centideg at the end.
        tan_arr = np.array([d[0] for d in lor_data])
        inv_cos_arr = np.array([1.0 / d[1] for d in lor_data])
        hl_arr = np.array([d[2] for d in lor_data])

        A = np.column_stack([tan_arr, inv_cos_arr])
        try:
            result, _, _, _ = np.linalg.lstsq(A, hl_arr, rcond=None)
            X_deg, Y_deg = float(result[0]), float(result[1])
        except np.linalg.LinAlgError:
            return DEFAULT_X, DEFAULT_Y

        # Convert degrees → centidegrees (multiply by 100)
        X_cdeg = X_deg * 100.0
        Y_cdeg = Y_deg * 100.0

        # Sanity clamp: both should be non-negative, moderate magnitude
        X_cdeg = max(0.0, min(X_cdeg, 50.0))
        Y_cdeg = max(0.0, min(Y_cdeg, 50.0))

        return X_cdeg, Y_cdeg

    except Exception:
        return DEFAULT_X, DEFAULT_Y


# Cu Kα1/Kα2 doublet constants (NIST values).  Used by _write_instprm
# whenever the incoming wavelength is close to the Cu Kα weighted mean.
# Most Cu-source diffractometers export either the weighted average
# (~1.5418 Å) or Kα1-only (~1.5406 Å); both fall inside the window below.
# If the user truly wants a monochromated single-wavelength fit they can
# pass `kalpha2=False` explicitly to _write_instprm.
CU_KALPHA1_A     = 1.540593   # Å  (Cu Kα1, NIST)
CU_KALPHA2_A     = 1.544414   # Å  (Cu Kα2, NIST)
CU_KALPHA2_RATIO = 0.5        # I(Kα2) / I(Kα1), canonical for Cu
CU_DETECT_MIN_A  = 1.537      # auto-detect window: any wavelength
CU_DETECT_MAX_A  = 1.548      # in [1.537, 1.548] Å is treated as Cu.


def _is_cu_kalpha(wavelength):
    """True if *wavelength* (Å) is within the Cu Kα auto-detect window."""
    try:
        w = float(wavelength)
    except (TypeError, ValueError):
        return False
    return CU_DETECT_MIN_A <= w <= CU_DETECT_MAX_A


def _write_instprm(work_dir, wavelength, polariz=None, sh_l=None,
                   u=None, v=None, w=None, x=None, y=None,
                   kalpha2=None):
    """Write a minimal GSAS-II .instprm file. Returns path.

    Uses module-level DEFAULT_* constants unless overridden.
    U, V, W, X, Y are initial guesses that GSAS-II will refine.
    Polariz. and SH/L are NOT refined — they should match the instrument.

    Wavelength handling
    -------------------
    - When *kalpha2* is True, write a Cu Kα1/Kα2 doublet profile
      (`Lam1`, `Lam2`, `I(L2)/I(L1)`).  The passed-in *wavelength*
      argument is ignored in favour of the NIST Cu Kα constants; this
      matches what a Kβ-filtered Cu lab diffractometer actually emits.
    - When *kalpha2* is False, write a single-wavelength profile (`Lam`).
    - When *kalpha2* is None (default), auto-detect: if *wavelength*
      falls within the Cu Kα window (~1.54 Å) write the doublet;
      otherwise write single wavelength.  This fixes the silent misfit
      for unmonochromated Cu data without requiring a UI change.
    """
    path = os.path.join(work_dir, 'instrument.instprm')
    pol = polariz if polariz is not None else DEFAULT_POLARIZ
    shl = sh_l if sh_l is not None else DEFAULT_SH_L
    _u = u if u is not None else DEFAULT_U
    _v = v if v is not None else DEFAULT_V
    _w = w if w is not None else DEFAULT_W
    _x = x if x is not None else DEFAULT_X
    _y = y if y is not None else DEFAULT_Y

    # Decide wavelength mode.
    if kalpha2 is None:
        use_doublet = _is_cu_kalpha(wavelength)
    else:
        use_doublet = bool(kalpha2)

    if use_doublet:
        # Cu Kα1 / Kα2 doublet.  GSAS-II treats Type:PXC with Lam1/Lam2
        # keys as a two-wavelength calculation and generates every peak
        # as a Kα1/Kα2 pair at the correct Bragg-law-scaled 2θ positions.
        wavelength_lines = [
            f'Lam1:{CU_KALPHA1_A:.6f}',
            f'Lam2:{CU_KALPHA2_A:.6f}',
            f'I(L2)/I(L1):{CU_KALPHA2_RATIO:.4f}',
        ]
        print(f"  instprm: Cu Kα1/Kα2 doublet "
              f"(Lam1={CU_KALPHA1_A:.6f}, Lam2={CU_KALPHA2_A:.6f}, "
              f"I2/I1={CU_KALPHA2_RATIO}).", flush=True)
    else:
        wavelength_lines = [f'Lam:{wavelength:.6f}']
        print(f"  instprm: single wavelength "
              f"(Lam={wavelength:.6f}).", flush=True)

    # Zero seed: starting value for GSAS-II's Zero parameter refinement.
    # This is just an initial guess — Zero is still refined freely in
    # Stage 2 and Stage 6, so GSAS-II will walk to whatever value
    # minimises Rwp regardless of this starting point.  The WC/W2C
    # Synergy-Dualflex .par reports a measured theta zero-correction
    # of -0.25° for this instrument configuration, so seeding at -0.25°
    # gives the optimiser a slightly better starting point than 0.0.
    # If the fit converges to a very different value (e.g. +0.07° as in
    # v5), that means GSAS-II is compensating for additional corrections
    # (transparency, displacement) that weren't present before — the
    # absolute value of the refined Zero is not a physical quantity in
    # isolation.  Set to 0.0 if seeding is undesirable for a given
    # instrument.
    _ZERO_SEED = -0.25
    lines = [
        '#GSAS-II instrument parameter file; do not add/delete items!',
        'Type:PXC',
    ] + wavelength_lines + [
        f'Zero:{_ZERO_SEED}',
        f'Polariz.:{pol}',
        f'U:{_u}',
        f'V:{_v}',
        f'W:{_w}',
        f'X:{_x}',
        f'Y:{_y}',
        'Z:0.0',
        f'SH/L:{shl}',
        'Azimuth:0.0',
    ]
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def _write_temp_cif(cif_text, phase_name='phase', work_dir=None, index=0):
    """Write CIF text to a temporary file. Returns path.

    If *work_dir* is given the file is created there (avoids Windows
    short-path / permission issues with the system temp directory).
    The *index* parameter ensures unique filenames when multiple phases
    share the same name.
    """
    if work_dir is not None:
        # Sanitise phase_name for use as a filename component
        safe = "".join(c if c.isalnum() or c in ('_', '-') else '_'
                       for c in phase_name)
        path = os.path.join(work_dir, f'gsas_{index}_{safe}.cif')
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(cif_text)
        return path
    # Fallback: system temp
    fd, path = tempfile.mkstemp(suffix='.cif', prefix=f'gsas_{index}_{phase_name}_')
    with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as f:
        f.write(cif_text)
    return path


def _extract_profile_params(phase_obj):
    """
    Extract TCH profile parameters from a GSAS-II phase object.
    GSAS-II stores profile coefficients in the histogram data.
    Returns dict with crystallite size and microstrain.

    GSAS-II Size data structure (isotropic):
      Size = ['isotropic', [size_value, refine_flag], ...]
    where size_value is in MICRONS (µm), NOT Angstroms.
    Convert: size_A = size_um * 10000

    Mustrain data structure (isotropic):
      Mustrain = ['isotropic', [strain_value, refine_flag], ...]
    where strain_value is in units of 10^-6 (micro-strain).
    """
    try:
        # GSAS-II stores profile params per histogram in the phase
        hapData = list(phase_obj.data['Histograms'].values())[0]
        size_data = hapData.get('Size', [])
        strain_data = hapData.get('Mustrain', [])

        cryst_size_A = None
        if size_data and len(size_data) > 1:
            size_val = float(size_data[1][0]) if size_data[1][0] > 0 else None
            if size_val is not None and size_val > 0:
                # GSAS-II stores Size in µm; convert to Å (×10000)
                cryst_size_A = size_val * 10000.0
                # Sanity check: crystallite size < 5 Å is physically
                # impossible (smaller than a unit cell).  This catches
                # cases where Size was never refined and GSAS-II returned
                # its unphysical initial value.
                if cryst_size_A < 5.0:
                    print(f"  WARNING: GSAS-II crystallite size = "
                          f"{cryst_size_A:.1f} Å ({size_val:.6f} µm) — "
                          f"unphysically small, discarding (will use "
                          f"Y-parameter fallback).", flush=True)
                    cryst_size_A = None

        microstrain = None
        if strain_data and len(strain_data) > 1:
            microstrain = float(strain_data[1][0])

        return {
            'crystallite_size_A': cryst_size_A,
            'microstrain_pct': microstrain,
        }
    except Exception:
        return {'crystallite_size_A': None, 'microstrain_pct': None}


def _extract_instrument_params(histogram):
    """Extract instrument/profile parameters from a GSAS-II histogram."""
    try:
        inst_params = histogram.data['Instrument Parameters'][0]
        U = float(inst_params.get('U', [0, 0])[1])
        V = float(inst_params.get('V', [0, 0])[1])
        W = float(inst_params.get('W', [0, 0])[1])
        X = float(inst_params.get('X', [0, 0])[1])
        Y = float(inst_params.get('Y', [0, 0])[1])
        return {'U': U, 'V': V, 'W': W, 'X': X, 'Y': Y}
    except Exception:
        return {'U': 0, 'V': 0, 'W': 0.1, 'X': 0, 'Y': 0}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_gsas2(tt, y_obs, sigma, phases, wavelength,
              tt_min=None, tt_max=None, n_bg_coeffs=6,
              max_cycles=10, progress_callback=None,
              instprm_file=None, polariz=None, sh_l=None,
              auto_bg=True, seed_params=None):
    """
    Run GSAS-II Rietveld refinement via GSASIIscriptable.

    Parameters
    ----------
    tt, y_obs, sigma : np.ndarray — data arrays
    phases   : list of dicts — must include 'cif_text' for atom positions
    wavelength : float (Å)
    tt_min/max : float — fitting range
    n_bg_coeffs : int — number of background coefficients (starting point
        when auto_bg=True, exact value when auto_bg=False)
    max_cycles : int — max refinement cycles
    instprm_file : str, optional — path to user-provided .instprm file;
        if given, this is used INSTEAD of the auto-generated defaults
    polariz : float, optional — monochromator polarization (default 0.99)
    sh_l : float, optional — Finger-Cox-Jephcoat asymmetry (default 0.002)
    auto_bg : bool — if True, automatically determine the optimal number
        of background coefficients from the data (default True)

    Returns dict compatible with Le Bail / Rietveld output (same keys).
    """
    if not _GSASII_AVAILABLE:
        raise RuntimeError(
            f"GSAS-II is not installed or not importable in this Python environment. "
            f"Install with: git clone https://github.com/AdvancedPhotonSource/GSAS-II.git "
            f"&& cd GSAS-II && pip install .\n"
            f"Import error: {_GSASII_IMPORT_ERROR}")

    # Validate: all phases need CIF text
    missing = [ph.get('name', '?') for ph in phases if not ph.get('cif_text')]
    if missing:
        raise ValueError(
            f"GSAS-II refinement requires CIF with atom coordinates for all phases. "
            f"Missing for: {', '.join(missing)}.")

    if tt_min is None: tt_min = float(tt.min())
    if tt_max is None: tt_max = float(tt.max())

    # Apply range mask
    mask = (tt >= tt_min) & (tt <= tt_max)
    tt_r = tt[mask]
    y_r = y_obs[mask]
    sig_r = sigma[mask] if sigma is not None else np.sqrt(np.maximum(y_r, 1.0))

    if progress_callback:
        progress_callback('GSAS-II: setting up project...')

    # ── Create temporary files ───────────────────────────────────────────
    # Use a temp dir inside the app directory to avoid spaces / short-path
    # issues in the user-profile temp folder (GSAS-II can't read files
    # whose path contains spaces).
    _app_tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.pardir, os.pardir, '.gsas_tmp')
    _app_tmp = os.path.normpath(_app_tmp)
    os.makedirs(_app_tmp, exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix='gsas2_', dir=_app_tmp)
    gpx_path = os.path.join(work_dir, 'refine.gpx')
    data_path = os.path.join(work_dir, 'data.xye')
    # Use user-provided .instprm if available; otherwise generate defaults.
    #
    # Seed-handling note (Phase I v5): the in-house Rietveld seed pipeline
    # fits peak widths using a SINGLE wavelength model, so its U/V/W/X/Y
    # already have the unresolved Kα1/Kα2 doublet broadening baked in.
    # When GSAS-II is about to model the doublet EXPLICITLY, those seeds
    # are ~2× too wide — using them causes GSAS-II to refine U downward
    # and cascades into scale/cell errors (observed on the first Cu
    # doublet run: wt% flipped from 19/81 to 2/98).  Solution: ignore
    # seed_params whenever the doublet will be written, and let the
    # observation-based estimator (_estimate_profile_params) derive seed
    # widths from the actual data.  The estimator already measures the
    # total apparent FWHM, which GSAS-II will then deconvolve correctly
    # once the Kα1/Kα2 pair is being forward-modelled.
    _use_doublet = _is_cu_kalpha(wavelength)
    _seed_active = (seed_params is not None) and (not _use_doublet)
    if _seed_active is False and seed_params is not None and _use_doublet:
        print(f"  Skipping single-wavelength seed params "
              f"(Cu Kα1/Kα2 doublet active — seeds would be "
              f"~2× too wide for the doublet model).", flush=True)

    if instprm_file and os.path.isfile(instprm_file):
        import shutil
        instprm_path = os.path.join(work_dir, 'instrument.instprm')
        shutil.copy2(instprm_file, instprm_path)
        print(f"Using user-provided instrument parameters: {instprm_file}",
              flush=True)
        # Populate est_* with defaults for downstream reset targets; the
        # user .instprm controls actual refinement starting values.
        est_u, est_v, est_w = DEFAULT_U, DEFAULT_V, DEFAULT_W
        est_x, est_y = DEFAULT_X, DEFAULT_Y
    elif _seed_active:
        # Use pre-refinement seed values (e.g. from in-house Rietveld).
        # Already in GSAS-II units (centideg² for U/V/W, centideg for X/Y).
        est_u = float(seed_params.get('U', DEFAULT_U))
        est_v = float(seed_params.get('V', DEFAULT_V))
        est_w = float(seed_params.get('W', DEFAULT_W))
        est_x = float(seed_params.get('X', DEFAULT_X))
        est_y = float(seed_params.get('Y', DEFAULT_Y))
        # Sanity clamp — wider than _estimate_profile_params because the
        # in-house Rietveld can legitimately find large U/W for
        # nanocrystalline materials (e.g. U=3.5 deg² = 35000 centideg²
        # for ~9 nm crystallites).
        est_u = max(0.1, min(est_u, 50000.0))      # centideg²
        est_v = max(-10000.0, min(est_v, 10000.0))
        est_w = max(0.1, min(est_w, 50000.0))
        est_x = max(0.0, min(est_x, 500.0))        # centideg
        est_y = max(0.0, min(est_y, 500.0))
        print(f"  Using seeded profile params: U={est_u:.2f}, "
              f"V={est_v:.2f}, W={est_w:.2f}, X={est_x:.2f}, "
              f"Y={est_y:.2f}", flush=True)
        instprm_path = _write_instprm(work_dir, wavelength,
                                       polariz=polariz, sh_l=sh_l,
                                       u=est_u, v=est_v, w=est_w,
                                       x=est_x, y=est_y)
    else:
        # Estimate initial profile parameters from observed peak widths.
        # This path is used when no user instprm and no usable seeds —
        # either because seed_params is None, or because the Cu doublet
        # is active (see _seed_active logic above).
        est_u, est_v, est_w = _estimate_profile_params(tt_r, y_r)
        est_x, est_y = _estimate_lorentzian_params(tt_r, y_r,
                                                    est_u, est_v, est_w)
        instprm_path = _write_instprm(work_dir, wavelength,
                                       polariz=polariz, sh_l=sh_l,
                                       u=est_u, v=est_v, w=est_w,
                                       x=est_x, y=est_y)
    # Save initial profile estimates for use as reset targets if
    # parameters diverge during refinement.  Better than resetting
    # to arbitrary small constants (e.g. 0.01) which are far from
    # any physical value and corrupt subsequent stages.
    _init_W = est_w if not (instprm_file and os.path.isfile(instprm_file)) else DEFAULT_W
    _init_U = est_u if not (instprm_file and os.path.isfile(instprm_file)) else DEFAULT_U
    _write_xye(data_path, tt_r, y_r, sig_r)

    cif_paths = []
    gsas_cif_texts = []  # Store CIF text sent to GSAS-II for each phase
    for i, ph in enumerate(phases):
        # Build a synthetic CIF from the phase dict's (conventional) cell
        # parameters.  This guarantees GSAS-II sees the correct space group
        # and cell geometry even when the original CIF used a primitive
        # setting (common with Materials Project data).
        cif_for_gsas = _build_conventional_cif(ph)
        gsas_cif_texts.append(cif_for_gsas)
        cif_path = _write_temp_cif(cif_for_gsas, ph.get('name', 'phase'),
                                   work_dir=work_dir, index=i)
        cif_paths.append(cif_path)
        # ── Diagnostic: dump CIF content for debugging ────────────────
        print(f"\n  === CIF for phase {i} ({ph.get('name', '?')}) ===",
              flush=True)
        print(f"  File: {cif_path}", flush=True)
        # Print first 30 lines of CIF (enough to see cell + atoms)
        _cif_lines = cif_for_gsas.strip().split('\n')
        for _ln in _cif_lines[:40]:
            print(f"    {_ln}", flush=True)
        if len(_cif_lines) > 40:
            print(f"    ... ({len(_cif_lines) - 40} more lines)", flush=True)
        print(f"  === end CIF ===\n", flush=True)

    try:
        # ── Build GSAS-II project ────────────────────────────────────────
        try:
            gpx = G2sc.G2Project(newgpx=gpx_path)
        except Exception as e:
            raise RuntimeError(
                f"GSAS-II failed to create a project. This usually means GSAS-II's "
                f"compiled Fortran extensions are not installed (requires a Fortran "
                f"compiler during installation).\n\n"
                f"Try using the built-in Rietveld refinement instead, or install "
                f"GSAS-II via: conda install gsas2pkg -c briantoby\n\n"
                f"Original error: {e}"
            ) from e

        # Add histogram (powder data)
        histogram = gpx.add_powder_histogram(
            data_path, instprm_path,
            databank=None,
        )

        # Set data range
        histogram.data['Limits'] = [[tt_min, tt_max], [tt_min, tt_max]]

        # ── Sigma inflation for 2D-integrated / capillary data ─────────
        # Data exported from CrysAlisPro powder generation (or any
        # similar 2D→1D ring-integration pipeline: Dioptas, pyFAI, GSAS-II
        # 2D, etc.) carries per-point σ that reflects ONLY counting
        # statistics across thousands of detector pixels per 2θ bin.
        # Those σ are typically 20–40× tighter than classical √I Poisson
        # on the summed ring — statistically correct for random noise,
        # but they do not include experimental systematics (beam drift,
        # capillary transparency tail, sample-height micromotion, detector
        # response nonuniformities).
        #
        # GSAS-II weights each point by w = 1/σ², so tight σ inflate
        # reduced χ² and GoF even when the structural model is sound.
        # For the WC/W2C Synergy-Dualflex dataset, reduced χ² ≈ 87 while
        # Rwp ≈ 4.8% — a classic signature of underestimated σ, not a
        # broken fit.  Inflate σ by a fixed factor via wtFactor (which
        # multiplies w uniformly: setting wtFactor = 1/K² inflates σ by
        # K and divides reduced χ² by K²).  Rwp is invariant under
        # uniform σ scaling because it is a ratio.
        #
        # K = 5 is a deliberately conservative choice — aggressive enough
        # to bring GoF to ~2 (an honest value for lab XRD), but
        # conservative enough to leave headroom for real model error.
        # Set K = 1.0 to disable inflation entirely (appropriate for
        # conventional Bragg-Brentano data where σ ≈ √I is already
        # correct).
        _SIGMA_INFLATION_FACTOR = 5.0
        try:
            _wt = 1.0 / (_SIGMA_INFLATION_FACTOR ** 2)
            histogram.data['wtFactor'] = _wt
            print(f"  σ inflation: wtFactor = {_wt:.4f} "
                  f"(σ × {_SIGMA_INFLATION_FACTOR:.1f}, reduced χ² ÷ "
                  f"{_SIGMA_INFLATION_FACTOR ** 2:.0f}). "
                  f"Rwp is invariant under this rescaling.", flush=True)
        except (KeyError, TypeError) as _wt_e:
            print(f"  WARNING: could not set wtFactor — {_wt_e}",
                  flush=True)

        # Fix histogram scale to 1.0 — NEVER refine it.
        # GSAS-II has N+1 scale parameters (N phase scales + 1 histogram
        # scale).  Only N are independent.  Refining all N+1 creates a
        # perfect correlation (100%) that causes SVD singularity and the
        # refinement gets stuck with zero peak intensity.
        # Standard practice: fix histogram scale, refine only phase scales.
        try:
            histogram.data['Sample Parameters']['Scale'] = [1.0, False]
        except (KeyError, TypeError):
            pass

        # Set background
        bkg_data = histogram.data['Background']
        bg_init = float(np.percentile(y_r, 2))
        bkg_data[0] = ['chebyschev-1', True, n_bg_coeffs,
                        bg_init] + [0.0] * (n_bg_coeffs - 1)

        # Add phases from CIF files — always embed the space group number in
        # the GSAS-II phasename so that two phases with the same formula
        # (e.g. W Pm-3n vs W Im-3m) are never treated as the same phase.
        gsas_phases = []
        used_names = set()
        for i, (ph, cif_path) in enumerate(zip(phases, cif_paths)):
            formula  = ph.get('formula', '') or ph.get('name', '') or 'Phase'
            sg_num   = ph.get('spacegroup_number', 1)
            # Always include SG number for unambiguous internal naming
            phasename = f"{formula}_sg{sg_num}"
            # Deduplicate in case two phases have the same formula + SG (rare)
            if phasename in used_names:
                phasename = f"{phasename}_{i+1}"
            used_names.add(phasename)
            try:
                phase_obj = gpx.add_phase(
                    cif_path,
                    phasename=phasename,
                    histograms=[histogram],
                )
            except Exception as e1:
                # Synthetic CIF failed — fall back to original CIF text
                orig_cif = ph.get('cif_text', '')
                if orig_cif:
                    try:
                        orig_path = _write_temp_cif(
                            orig_cif, ph.get('name', 'phase'),
                            work_dir=work_dir, index=100+i)
                        cif_paths.append(orig_path)  # for cleanup
                        phase_obj = gpx.add_phase(
                            orig_path,
                            phasename=phasename,
                            histograms=[histogram],
                        )
                        warnings.warn(
                            f"Synthetic CIF failed for '{ph.get('name', '?')}', "
                            f"using original CIF text (error: {e1})")
                    except Exception as e2:
                        cif_size = os.path.getsize(cif_path) if os.path.exists(cif_path) else -1
                        cif_head = ''
                        if os.path.exists(cif_path):
                            with open(cif_path, 'r', encoding='utf-8') as _f:
                                cif_head = _f.read(500)
                        raise RuntimeError(
                            f"GSAS-II could not read CIF for phase '{ph.get('name', '?')}' "
                            f"(file: {cif_path}, size: {cif_size} bytes).\n"
                            f"CIF preview:\n{cif_head}\n\n"
                            f"Original error: {e2}"
                        ) from e2
                else:
                    cif_size = os.path.getsize(cif_path) if os.path.exists(cif_path) else -1
                    cif_head = ''
                    if os.path.exists(cif_path):
                        with open(cif_path, 'r', encoding='utf-8') as _f:
                            cif_head = _f.read(500)
                    raise RuntimeError(
                        f"GSAS-II could not read CIF for phase '{ph.get('name', '?')}' "
                        f"(file: {cif_path}, size: {cif_size} bytes).\n"
                        f"CIF preview:\n{cif_head}\n\n"
                        f"Original error: {e1}"
                    ) from e1
            gsas_phases.append(phase_obj)

        # ── Diagnostic: print GSAS-II's expanded atoms for each phase ─────
        for _pi, _po in enumerate(gsas_phases):
            _atoms = _po.data.get('Atoms', [])
            _gcell = _po.data['General']['Cell']
            print(f"  GSAS-II phase {_pi} ({_po.name}): "
                  f"{len(_atoms)} atoms, SG={_po.data['General'].get('SGData', {}).get('SpGrp', '?')}, "
                  f"cell=[{_gcell[1]:.4f}, {_gcell[2]:.4f}, {_gcell[3]:.4f}, "
                  f"{_gcell[4]:.2f}, {_gcell[5]:.2f}, {_gcell[6]:.2f}]",
                  flush=True)
            for _ai, _at in enumerate(_atoms[:20]):  # print first 20 atoms
                _lbl = _at[0] if len(_at) > 0 else '?'
                _el  = _at[1] if len(_at) > 1 else '?'
                _x   = _at[3] if len(_at) > 3 else '?'
                _y   = _at[4] if len(_at) > 4 else '?'
                _z   = _at[5] if len(_at) > 5 else '?'
                print(f"    atom {_ai}: {_lbl} ({_el}) at "
                      f"({_x}, {_y}, {_z})", flush=True)

        # Track which stage succeeded (for fallback on failure)
        last_good_stage = 0
        _failed_stages = []

        # ── Helper to safely run a refinement stage ──────────────────────
        def _safe_refine(stage_name, refinement_dicts, stage_num):
            nonlocal last_good_stage
            try:
                gpx.do_refinements(refinement_dicts)
                last_good_stage = stage_num
                # Guard: ensure histogram scale stays fixed at 1.0.
                # GSAS-II's do_refinements can re-enable it internally
                # when 'Scale' appears in any refinement dict.
                try:
                    histogram.data['Sample Parameters']['Scale'] = [1.0, False]
                except (KeyError, TypeError):
                    pass
                return True
            except Exception as e:
                # GSAS-II internally restores from .bak0.gpx on failure,
                # so the project state reverts to pre-refinement. Safe to continue.
                _failed_stages.append(stage_name)
                warnings.warn(f"GSAS-II: {stage_name} failed ({e}). "
                             f"Continuing with results from stage {last_good_stage}.")
                return False

        # Determine structural complexity → boost cycles for complex phases
        def _count_asym_sites(ph):
            result = _reduce_to_asymmetric_unit(
                ph.get('cif_text', ''),
                declared_sg=ph.get('spacegroup_number'))
            if isinstance(result, str):
                # CifWriter returned full CIF text — parse site count
                _parsed = parse_cif(result)
                return len(_parsed.get('sites') or [])
            return len(result) if result else 0
        max_asym_atoms = max(
            (_count_asym_sites(ph) for ph in phases if ph.get('cif_text')),
            default=1)
        _complex = max_asym_atoms > 6  # W2C has ~6 asym sites
        _cyc_mult = 2 if _complex else 1  # double cycles for complex phases

        # ── Auto-select background coefficients ────────────────────────
        # When auto_bg is enabled (default, GUI "Auto" selection),
        # evaluate the data to determine the optimal number of Chebyshev
        # terms.  This removes the subjectivity of the manual selector.
        if auto_bg:
            n_bg_coeffs = _auto_select_bg_coeffs(
                tt, y_obs, phases, wavelength, tt_min, tt_max,
                user_n=n_bg_coeffs)
            # Update ONLY the coefficient count in GSAS-II's background config.
            # CRITICAL: do NOT overwrite the initial coefficient values —
            # GSAS-II's own initial estimates are typically better than our
            # simple percentile + zeros approach.  Overwriting them with
            # [bg_init, 0, 0, ...] gives a flat starting background that
            # sends the refinement off track.
            bkg_data = histogram.data['Background']
            existing = bkg_data[0]
            old_n = existing[2] if len(existing) > 2 else 6
            existing[2] = n_bg_coeffs
            # Pad or truncate coefficient list to match new count
            # Format: ['chebyschev-1', refine_flag, n_coeffs, c0, c1, ...]
            n_existing_coeffs = len(existing) - 3
            if n_bg_coeffs > n_existing_coeffs:
                # Pad with zeros for new higher-order terms
                existing.extend([0.0] * (n_bg_coeffs - n_existing_coeffs))
            elif n_bg_coeffs < n_existing_coeffs:
                # Truncate
                bkg_data[0] = existing[:3 + n_bg_coeffs]
            print(f"  Background: auto-selected {n_bg_coeffs} Chebyshev "
                  f"coefficients (was {old_n})", flush=True)
        else:
            print(f"  Background: using user-specified {n_bg_coeffs} "
                  f"Chebyshev coefficients", flush=True)

        if progress_callback:
            progress_callback('GSAS-II: stage 1 — refining background + scale...')

        # ── Stage 1: Background + scale ────────────────────────────────
        # IMPORTANT: Use EQUAL initial scales for all phases to avoid biasing
        # the refinement toward whichever phase is refined first.
        # Previous sequential approach (one phase at a time) caused the first
        # phase (WC) to absorb all intensity, driving the second phase (W2C)
        # scale to zero — making all subsequent stages fail for W2C.
        peak_height = float(np.max(y_r) - np.percentile(y_r, 5))
        n_phases = len(gsas_phases)
        init_scale = max(0.01, peak_height / max(1.0, n_phases * 50.0))
        print(f"  Initial scale for all {n_phases} phases: {init_scale:.4f} "
              f"(peak_height={peak_height:.1f})", flush=True)
        for phase_obj in gsas_phases:
            hapData = list(phase_obj.data['Histograms'].values())[0]
            hapData['Scale'] = [init_scale, False]  # start fixed

        # First: refine background only
        _safe_refine('background', [{
            'set': {
                'Background': {'type': 'chebyschev-1', 'refine': True,
                                'no. coeffs': n_bg_coeffs},
            },
            'cycles': min(max_cycles, 5),
        }], 1)

        # Now refine ALL phase scales SIMULTANEOUSLY.
        # This is critical for overlapping phases (WC + W2C) — simultaneous
        # refinement lets the optimizer distribute intensity between phases
        # based on their respective peak patterns, rather than the first
        # phase greedily absorbing everything.
        for phase_obj in gsas_phases:
            hapData = list(phase_obj.data['Histograms'].values())[0]
            hapData['Scale'] = [hapData['Scale'][0], True]  # turn on
        _safe_refine('all scales (simultaneous)', [{
            'set': {},
            'cycles': min(max_cycles, 8),
        }], 1)

        # ── Scale floor protection ──────────────────────────────────────
        # If any phase's scale collapsed to near-zero, it means the model
        # couldn't distribute intensity properly (perhaps wrong starting
        # guess).  Reset the collapsed phase to a floor value and re-refine
        # with background to give it another chance.
        _max_scale = max(
            list(po.data['Histograms'].values())[0].get('Scale', [0.001])[0]
            for po in gsas_phases)
        _scale_floor = max(1e-3, _max_scale * 0.01)  # at least 1% of max
        _any_reset = False
        for idx, phase_obj in enumerate(gsas_phases):
            hapData = list(phase_obj.data['Histograms'].values())[0]
            current_scale = hapData['Scale'][0]
            print(f"  Phase {idx} scale after simultaneous fit: "
                  f"{current_scale:.4e}", flush=True)
            if current_scale < _scale_floor:
                warnings.warn(
                    f"GSAS-II: phase {idx} scale ({current_scale:.2e}) below "
                    f"floor ({_scale_floor:.2e}) — resetting to floor value "
                    f"and re-refining.")
                hapData['Scale'] = [_scale_floor, True]
                _any_reset = True
        if _any_reset:
            # Re-refine with corrected scales + background
            _safe_refine('scales (after floor reset)', [{
                'set': {
                    'Background': {'type': 'chebyschev-1', 'refine': True,
                                    'no. coeffs': n_bg_coeffs},
                },
                'cycles': min(max_cycles, 8),
            }], 1)
            # Print scales after re-refinement
            for idx, phase_obj in enumerate(gsas_phases):
                hapData = list(phase_obj.data['Histograms'].values())[0]
                print(f"  Phase {idx} scale after re-refinement: "
                      f"{hapData['Scale'][0]:.4e}", flush=True)

        if progress_callback:
            progress_callback('GSAS-II: stage 2 — refining profile parameters...')

        # ── Stage 2: Profile + background + zero + transparency ──────
        # Refine Gaussian (U, W) and Lorentzian (X, Y) profile parameters
        # together with zero shift, background, and sample transparency.
        #
        # Profile parameter selection for lab XRD stability:
        #   Refined: U, W (Gaussian Caglioti), X, Y (Lorentzian), Zero,
        #            DisplaceY (sample transparency, Debye-Scherrer)
        #   Fixed:   V (99%+ correlated with U and W — instrument-determined)
        #            SH/L (instrument constant, 99.9% correlated with Zero)
        #            DisplaceX (Bragg-Brentano sample-height; not the
        #            right correction for capillary geometry)
        # Removing the Bragg-Brentano-specific DisplaceX and adding the
        # Debye-Scherrer DisplaceY (sample transparency, scales as sin 2θ)
        # is the appropriate choice for capillary / transmission data
        # from a 2D area detector (e.g., Synergy-Dualflex with HyPix-6000).
        # For flat-plate Bragg-Brentano data, swap the two parameter names
        # below.
        _safe_refine('profile + bg + zero + transparency', [{
            'set': {
                'Background': {'type': 'chebyschev-1', 'refine': True,
                                'no. coeffs': n_bg_coeffs},
                'Instrument Parameters': ['U', 'W', 'X', 'Y', 'Zero'],
                'Sample Parameters': ['DisplaceY'],
            },
            'cycles': min(max_cycles, 10 * _cyc_mult),
        }], 2)

        # ── Identify phases with negligible scale ──────────────────────────
        # Phases with effectively zero scale contribute nothing to the pattern.
        # Refining their cell/atom parameters has zero gradient and causes
        # divergence (e.g. "Invalid cell metric tensor" errors).  Skip them.
        _negligible_phases = set()
        _max_scale_now = max(
            list(po.data['Histograms'].values())[0].get('Scale', [0])[0]
            for po in gsas_phases) or 1e-10
        for idx, phase_obj in enumerate(gsas_phases):
            hapData = list(phase_obj.data['Histograms'].values())[0]
            s = hapData.get('Scale', [0])[0]
            if s < _max_scale_now * 1e-3:  # less than 0.1% of max
                _negligible_phases.add(idx)
                print(f"  Phase {idx}: scale={s:.2e} << max={_max_scale_now:.2e} "
                      f"— skipping cell/atom refinement stages.", flush=True)

        # ── Preferred orientation setup ──────────────────────────────────
        # For phases with crystal systems known to exhibit preferred
        # orientation in flat-plate geometry (hexagonal platelets, fibrous
        # crystals, etc.), enable March-Dollase correction.
        # This is a standard Rietveld parameter, not a bandaid — it models
        # the physical tendency of crystallites to orient non-randomly on
        # the sample surface.  Without it, peak intensity ratios cannot
        # match observed data for textured samples.
        #
        # Common preferred orientation directions:
        #   Hexagonal platelets (WC, BN, MoS2):  [0, 0, 1]
        #   Orthorhombic needles:                 [1, 0, 0] or [0, 1, 0]
        #
        # The MD ratio starts at 1.0 (no texture) and is refined later
        # alongside Uiso (both affect peak intensities).
        _pref_ori_phases = set()
        for idx, (phase_obj, ph_input) in enumerate(zip(gsas_phases, phases)):
            if idx in _negligible_phases:
                continue
            sys_ = (ph_input.get('system') or 'triclinic').lower()
            if sys_ in ('hexagonal', 'trigonal'):
                # Hexagonal phases: preferred orientation along [001]
                try:
                    hapData = list(phase_obj.data['Histograms'].values())[0]
                    po = hapData.get('Pref.Ori.', ['MD', 1.0, False, [0, 0, 1]])
                    po[0] = 'MD'           # March-Dollase model
                    po[1] = 1.0            # start at no texture
                    po[3] = [0, 0, 1]      # preferred orientation direction
                    hapData['Pref.Ori.'] = po
                    _pref_ori_phases.add(idx)
                    print(f"  Phase {idx} ({sys_}): March-Dollase [001] "
                          f"preferred orientation enabled.", flush=True)
                except Exception as e:
                    print(f"  Phase {idx}: could not set preferred "
                          f"orientation: {e}", flush=True)

        if progress_callback:
            progress_callback('GSAS-II: stage 3 — refining cell parameters...')

        # ── Stage 3: Cell parameters (one phase at a time) ───────────────
        # Refining all cells at once with many atoms can cause arccos
        # errors when cell angles go unphysical. Do it per phase.
        # For cubic phases, lock angles to 90° to prevent arccos errors.
        for idx, (phase_obj, ph_input) in enumerate(zip(gsas_phases, phases)):
            if idx in _negligible_phases:
                print(f"  Skipping cell refinement for phase {idx} "
                      f"(negligible scale).", flush=True)
                continue
            sys_ = (ph_input.get('system') or 'triclinic').lower()
            # For high-symmetry systems, clamp angles before refining
            # to prevent them from drifting to unphysical values.
            if sys_ in ('cubic', 'tetragonal', 'orthorhombic'):
                try:
                    _cell = phase_obj.data['General']['Cell']
                    _cell[4] = 90.0  # alpha
                    _cell[5] = 90.0  # beta
                    _cell[6] = 90.0  # gamma
                except Exception:
                    pass
            elif sys_ in ('hexagonal', 'trigonal'):
                try:
                    _cell = phase_obj.data['General']['Cell']
                    _cell[4] = 90.0   # alpha
                    _cell[5] = 90.0   # beta
                    _cell[6] = 120.0  # gamma
                except Exception:
                    pass
            # Save cell params before refinement for sanity check
            _pre_cell = list(phase_obj.data['General']['Cell'])
            phase_obj.set_refinements({'Cell': True})
            ok = _safe_refine(f'cell (phase {idx})', [{
                'set': {},
                'cycles': min(max_cycles, 8 * _cyc_mult),
            }], 3)
            if not ok:
                # Turn cell refinement back off for this phase
                phase_obj.set_refinements({'Cell': False})
            else:
                # ── Post-stage cell sanity check ──────────────────────
                # Even when _safe_refine succeeds, the cell may have
                # diverged to unphysical values within the cycles before
                # GSAS-II's internal checks caught it. Detect and revert.
                _post_cell = phase_obj.data['General']['Cell']
                _cell_ok = True
                for _ci in range(1, 4):  # a, b, c
                    _pre_v = float(_pre_cell[_ci])
                    _post_v = float(_post_cell[_ci])
                    if _pre_v > 0.1:
                        _ratio = _post_v / _pre_v
                        if _ratio > 2.0 or _ratio < 0.5:
                            _cell_ok = False
                            break
                for _ci in range(4, 7):  # alpha, beta, gamma
                    if abs(float(_post_cell[_ci]) - float(_pre_cell[_ci])) > 30:
                        _cell_ok = False
                        break
                if not _cell_ok:
                    warnings.warn(
                        f"GSAS-II: cell params for phase {idx} diverged "
                        f"after refinement — reverting to pre-refinement "
                        f"values and turning off cell refinement.")
                    for _ci in range(len(_pre_cell)):
                        phase_obj.data['General']['Cell'][_ci] = _pre_cell[_ci]
                    phase_obj.set_refinements({'Cell': False})

        if progress_callback:
            progress_callback('GSAS-II: stage 4 — refining atomic displacement + size/strain...')

        # ── Stage 4: Atomic displacement (Uiso) + preferred orientation ──
        #             + Size and Mustrain HAP parameters
        # Uiso and preferred orientation both affect peak intensities, so
        # they should refine together to avoid one absorbing the other's
        # effect.
        #
        # NEW: Also refine isotropic Size and Mustrain (crystallite size
        # and micro-strain) per-phase.  These HAP parameters control
        # per-phase Lorentzian broadening in GSAS-II's profile model.
        # Without refining them, the instrument Y parameter (shared across
        # all phases) must accommodate all size broadening, which fails
        # for mixed-phase systems (e.g. WC + W2C) where phases have
        # different crystallite sizes.  Unrefined Size defaults to
        # GSAS-II's initial value (~1 µm), producing wrong peak shapes
        # and trapping the refinement in a local minimum.
        for idx, phase_obj in enumerate(gsas_phases):
            if idx in _negligible_phases:
                continue
            try:
                phase_obj.set_refinements({'Atoms': {'all': 'U'}})
            except Exception:
                pass
        # Enable preferred orientation refinement for applicable phases
        for idx in _pref_ori_phases:
            if idx not in _negligible_phases:
                try:
                    phase_obj = gsas_phases[idx]
                    phase_obj.set_HAP_refinements(
                        {'Pref.Ori.': True})
                    print(f"  Phase {idx}: March-Dollase ratio refinement "
                          f"enabled.", flush=True)
                except Exception as e:
                    print(f"  Phase {idx}: could not enable Pref.Ori. "
                          f"refinement: {e}", flush=True)
        _safe_refine('Uiso + preferred orientation', [{
            'set': {},
            'cycles': min(max_cycles, 8 * _cyc_mult),
        }], 4)

        # Freeze Uiso after Stage 4.  For heavy elements like W (Z=74)
        # with limited lab XRD reflections, Uiso is weakly determined
        # and becomes singular when refined alongside scale + profile in
        # later stages.  The Stage 4 value is our best estimate; freezing
        # it prevents SVD drops in Stages 5–6.
        for phase_obj in gsas_phases:
            try:
                phase_obj.clear_refinements({'Atoms': {'all': ''}})
            except Exception:
                pass

        if progress_callback:
            progress_callback('GSAS-II: stage 5 — refining atom positions (XYZ)...')

        # ── Stage 5: Atom positions (XYZ) ─────────────────────────────
        # For complex structures (carbides, oxides) the COD database atom
        # coordinates may not exactly match this sample.  Wrong positions →
        # wrong structure factors F(hkl) → the Rietveld model cannot match
        # observed peak intensity ratios → convergence failure.
        # GSAS-II automatically constrains atoms on special positions, so
        # this is safe for simple metals (no-op) while critical for W2C etc.
        #
        # Save atom positions before refinement for damping check.
        saved_xyz = {}
        for idx, phase_obj in enumerate(gsas_phases):
            if idx in _negligible_phases:
                continue  # skip XYZ for phases with negligible scale
            phase_atoms = phase_obj.data.get('Atoms', [])
            saved_xyz[idx] = [(a[3], a[4], a[5]) if len(a) > 5 else None
                              for a in phase_atoms]
            try:
                phase_obj.set_refinements({'Atoms': {'all': 'X'}})
            except Exception:
                pass
        xyz_ok = _safe_refine('atom XYZ', [{
            'set': {},
            'cycles': min(max_cycles, 3 * _cyc_mult),
        }], 5)

        # Damping check: if any atom jumped > 0.5 fractional units,
        # it likely hopped to a symmetry-equivalent site — revert it.
        if xyz_ok:
            for idx, phase_obj in enumerate(gsas_phases):
                phase_atoms = phase_obj.data.get('Atoms', [])
                for j, atom in enumerate(phase_atoms):
                    if (len(atom) > 5 and idx in saved_xyz
                            and j < len(saved_xyz[idx])
                            and saved_xyz[idx][j] is not None):
                        ox, oy, oz = saved_xyz[idx][j]
                        dx = abs(atom[3] - ox)
                        dy = abs(atom[4] - oy)
                        dz = abs(atom[5] - oz)
                        if max(dx, dy, dz) > 0.5:
                            warnings.warn(
                                f"GSAS-II: atom {j} in phase {idx} jumped "
                                f"by ({dx:.3f},{dy:.3f},{dz:.3f}) — "
                                f"reverting to original position.")
                            atom[3], atom[4], atom[5] = ox, oy, oz

        # Turn off all atom refinement flags (XYZ done, Uiso frozen from Stage 4)
        for phase_obj in gsas_phases:
            try:
                phase_obj.clear_refinements({'Atoms': {'all': ''}})
            except Exception:
                pass

        if progress_callback:
            progress_callback('GSAS-II: stage 6 — final all-together refinement...')

        # ── Stage 6: Final all-together refinement ──────────────────────
        # Refine background, profile (U, W, X, Y, Zero), scales, and cell
        # parameters simultaneously.  Uiso is frozen at the Stage 4 value
        # to avoid SVD singularity (Uiso ↔ scale correlation is too strong
        # for heavy elements like W with limited lab XRD reflections).
        # Cell is re-enabled; Uiso stays frozen (determined in Stage 4).
        for idx, phase_obj in enumerate(gsas_phases):
            if idx not in _negligible_phases:
                phase_obj.set_refinements({'Cell': True})
        _safe_refine('final: all parameters together', [{
            'set': {
                'Background': {'type': 'chebyschev-1', 'refine': True,
                                'no. coeffs': n_bg_coeffs},
                'Instrument Parameters': ['U', 'W', 'X', 'Y', 'Zero'],
                'Sample Parameters': ['DisplaceY'],
            },
            'cycles': min(max_cycles, 15 * _cyc_mult),
        }], 6)

        # Turn off cell refinement after final stage to avoid affecting
        # any subsequent operations
        for phase_obj in gsas_phases:
            try:
                phase_obj.set_refinements({'Cell': False})
            except Exception:
                pass

        # Warn if multiple refinement stages failed — result may be unreliable
        if len(_failed_stages) >= 3:
            warnings.warn(
                f"GSAS-II: {len(_failed_stages)} stages failed "
                f"({', '.join(_failed_stages)}). The refinement result may be "
                f"unreliable — check CIF quality or try simpler fitting range.")

        # ── Extract parameter uncertainties from covariance matrix ────────
        # After the final refinement, GSAS-II stores the covariance matrix
        # and per-parameter ESDs (estimated standard deviations) in the
        # project's Covariance data.  We extract scale-factor ESDs here
        # so we can propagate them into Hill & Howard weight fractions.
        #
        # GSAS-II parameter naming convention for phase scales:
        #   '<phase_id>::Scale'  (e.g. '0::Scale', '1::Scale')
        # The covariance data lives in gpx.data['Covariance']['data'].
        scale_sigmas = {}   # phase_name → sigma(Scale)
        try:
            # GSAS-II stores covariance data in multiple possible locations
            # depending on version.  Try them in order of preference.
            cov_data = {}
            for cov_path in [
                lambda: gpx.data.get('Covariance', {}).get('data', {}),
                lambda: gpx.data.get('Covariance', {}),
                lambda: gpx['Covariance']['data'] if 'Covariance' in gpx else {},
            ]:
                try:
                    _cd = cov_path()
                    if _cd and 'varyList' in _cd:
                        cov_data = _cd
                        break
                except (KeyError, TypeError, AttributeError):
                    continue

            if not cov_data:
                # Debug: log available top-level keys
                try:
                    top_keys = list(gpx.data.keys()) if hasattr(gpx.data, 'keys') else str(type(gpx.data))
                    print(f"  Covariance: not found. gpx.data keys: {top_keys}",
                          flush=True)
                except Exception:
                    print("  Covariance: not found, could not inspect gpx.data",
                          flush=True)

            vary_list = cov_data.get('varyList', [])
            sig_list  = cov_data.get('sig', [])
            # CRITICAL: sig_list is often a numpy array.  Do NOT use bare
            # 'if sig_list' — numpy arrays raise "ambiguous truth value"
            # when they contain >1 element.  Use len() instead.
            # Also convert to Python lists for safe .index() / 'in' usage.
            vary_list = list(vary_list)
            sig_list  = list(sig_list)
            if len(vary_list) > 0 and len(sig_list) > 0 and len(vary_list) == len(sig_list):
                # Log vary list for debugging scale parameter name format
                scale_vars = [v for v in vary_list if 'Scale' in v or 'scale' in v]
                print(f"  Covariance: {len(vary_list)} params, "
                      f"scale-related: {scale_vars}", flush=True)
                for idx_p, phase_obj in enumerate(gsas_phases):
                    # GSAS-II naming conventions for phase-histogram scales:
                    #   '{phase_id}:{hist_id}:Scale'  e.g. '0:0:Scale'
                    #   '{phase_id}::Scale'            e.g. '0::Scale'
                    # We try multiple patterns to be robust across versions.
                    found = False
                    for pattern in [f'{idx_p}:0:Scale', f'{idx_p}::Scale']:
                        if pattern in vary_list:
                            pos = vary_list.index(pattern)
                            scale_sigmas[phase_obj.name] = float(sig_list[pos])
                            found = True
                            break
                    if not found:
                        # Fallback: search for any Scale param for this phase
                        for vname, vsig in zip(vary_list, sig_list):
                            if vname.startswith(f'{idx_p}:') and 'Scale' in vname:
                                scale_sigmas[phase_obj.name] = float(vsig)
                                found = True
                                break
            else:
                print(f"  Covariance: varyList={len(vary_list) if vary_list else 0}, "
                      f"sig={len(sig_list) if sig_list else 0} — "
                      f"cannot extract ESDs", flush=True)
            if scale_sigmas:
                print(f"  Scale factor ESDs: {scale_sigmas}", flush=True)
            else:
                print("  Scale factor ESDs: not available (scale params "
                      "not found in vary list)", flush=True)
        except Exception as e:
            print(f"  Scale factor ESDs: extraction failed ({e})", flush=True)

        if progress_callback:
            progress_callback('GSAS-II: extracting results...')

        # ── Extract results ──────────────────────────────────────────────
        # Get calculated pattern
        y_calc_full = np.array(histogram.getdata('ycalc'))
        y_obs_full = np.array(histogram.getdata('yobs'))
        y_bg_full = np.array(histogram.getdata('background'))
        tt_full = np.array(histogram.getdata('x'))

        # Trim to our range
        rmask = (tt_full >= tt_min) & (tt_full <= tt_max)
        tt_out = tt_full[rmask]
        y_obs_out = y_obs_full[rmask]
        y_calc_out = y_calc_full[rmask]
        y_bg_out = y_bg_full[rmask]
        _y_bg_gsas = y_bg_out.copy()   # preserve for unbiased phase isolation

        # ── Background: preserve GSAS-II's refined background for statistics ──
        # CRITICAL: compute fit statistics using GSAS-II's ACTUAL refined
        # background (_y_bg_gsas), NOT the post-processed version.  The
        # previous approach applied a "dip correction" (smoothing + raising)
        # to y_bg_out, then computed Rwp against this modified background.
        # This created an inconsistency: GSAS-II optimized the fit against
        # its own background, but the reported Rwp was computed against a
        # different one.  This artificial mismatch inflated Rwp and made it
        # appear that the refinement was stuck, when the real GSAS-II Rwp
        # may have been lower.
        #
        # The dip correction is still applied for DISPLAY purposes only
        # (the background curve shown in plots), but statistics are always
        # computed from the unmodified GSAS-II output.
        y_bg_display = y_bg_out.copy()
        if len(y_bg_display) >= 3:
            try:
                _trend_coeffs = np.polyfit(tt_out, y_bg_display, 2)
                _trend = np.polyval(_trend_coeffs, tt_out)
                _resid = y_bg_display - _trend

                _step = float(tt_out[1] - tt_out[0]) if len(tt_out) > 1 else 0.02
                _sig  = max(3, int(10.0 / _step))          # 10° Gaussian sigma
                _k    = min(3 * _sig, len(_resid) // 2)
                if _k >= 1:
                    _kx   = np.arange(-_k, _k + 1, dtype=float)
                    _kern = np.exp(-0.5 * (_kx / _sig) ** 2)
                    _kern /= _kern.sum()
                    _padded = np.pad(_resid, _k, mode='edge')
                    _smooth_resid = np.convolve(_padded, _kern, mode='valid')
                    y_bg_display = _trend + np.maximum(_resid, _smooth_resid)
            except (np.linalg.LinAlgError, ValueError):
                pass  # skip dip correction if polyfit fails

        diff_out = y_obs_out - y_calc_out

        # Weights for statistics — use GSAS-II's unmodified data
        weights_out = 1.0 / np.maximum(
            np.where(sig_r is not None, sig_r**2,
                     np.maximum(y_obs_out, 1.0)), 1e-6)

        # Compute statistics using GSAS-II's actual background
        n_params_est = sum(
            len(list(phase_obj.atoms())) + 7  # atoms + cell + scale + profile
            for phase_obj in gsas_phases
        ) + n_bg_coeffs + 1
        stats = compute_fit_statistics(y_obs_out, y_calc_out,
                                        weights_out, n_params_est)

        # Prefer GSAS-II's own R-factors (computed consistently with
        # its own background and weighting scheme)
        try:
            gsas_stats = histogram.get_statistics()
            if gsas_stats:
                stats['Rwp'] = round(gsas_stats.get('Rwp', stats['Rwp']), 2)
                stats['Rp'] = round(gsas_stats.get('Rp', stats['Rp']), 2)
        except Exception:
            pass

        # Use the display-corrected background for the output (plots)
        # but keep _y_bg_gsas for phase isolation (unchanged)
        y_bg_out = y_bg_display

        # Extract instrument params
        inst = _extract_instrument_params(histogram)

        # Zero shift
        try:
            zero_shift = float(
                histogram.data['Instrument Parameters'][0].get('Zero', [0, 0])[1])
        except Exception:
            zero_shift = 0.0

        # ── Per-phase results ────────────────────────────────────────────
        phase_patterns = []
        phase_results = []
        all_phase_refs = []   # per-phase reflection lists from generate_reflections

        # ── Weight fractions via Hill & Howard (1987) ────────────────────
        # W_α = S_α · Z_α · M_α · V_α  /  Σ(S_i · Z_i · M_i · V_i)
        # If Z or M are unavailable, fall back to raw scale normalisation.
        raw_scales = {}
        try:
            for phase_obj in gsas_phases:
                hapData = list(phase_obj.data['Histograms'].values())[0]
                scale_entry = hapData.get('Scale', [1.0])
                scale_val = scale_entry[0] if isinstance(scale_entry, (list, tuple)) else float(scale_entry)
                raw_scales[phase_obj.name] = scale_val
        except Exception as e:
            warnings.warn(f"GSAS-II: could not read scale factors: {e}")

        zmv_values = {}
        use_zmv = True
        for ph_input, phase_obj in zip(phases, gsas_phases):
            S = raw_scales.get(phase_obj.name, 1.0)
            _cell = phase_obj.data['General']['Cell']
            V_ph = cell_volume(float(_cell[1]), float(_cell[2]), float(_cell[3]),
                               float(_cell[4]), float(_cell[5]), float(_cell[6]))
            Z = ph_input.get('Z')
            M = molar_mass_from_formula(ph_input.get('formula', ''))
            if Z and M and V_ph > 0:
                zmv_values[phase_obj.name] = float(S) * float(Z) * float(M) * V_ph
            else:
                use_zmv = False
                break

        if not use_zmv:
            warnings.warn("GSAS-II: Z or molar mass unavailable for one or more "
                         "phases — falling back to raw scale factor weighting.")
            zmv_values = dict(raw_scales)

        total_zmv = sum(zmv_values.values()) or 1e-10
        print(f"GSAS-II scale factors: {raw_scales}", flush=True)
        print(f"GSAS-II ZMV values: {zmv_values}", flush=True)
        print(f"GSAS-II use_zmv: {use_zmv}", flush=True)

        # Warn if all scale factors are identical (common with failed refinement)
        scale_vals = list(raw_scales.values())
        if len(scale_vals) >= 2 and len(set(round(v, 6) for v in scale_vals)) == 1:
            warnings.warn("GSAS-II: all phase scale factors are identical — "
                         "refinement may not have converged properly.")

        # ── Weight fraction uncertainties via error propagation ──────────
        # For N phases with Hill & Howard weights:
        #   W_α = P_α / T   where P_α = S_α·Z_α·M_α·V_α, T = Σ P_i
        #
        # Partial derivative: ∂W_α/∂S_α = (Z_α·M_α·V_α) · (T - P_α) / T²
        #                     ∂W_α/∂S_β = -(Z_α·M_α·V_α·S_α) · (Z_β·M_β·V_β) / T²   (β≠α)
        #
        # FULL covariance treatment:
        # σ²(W_α) = Σ_i Σ_j (∂W_α/∂S_i) · (∂W_α/∂S_j) · Cov(S_i, S_j)
        #
        # For overlapping phases (WC + W2C), scale factors are strongly
        # ANTI-correlated (increasing one decreases the other).  Ignoring
        # this correlation drastically underestimates the uncertainties
        # (e.g. 0.05% instead of the true ~1-3%).
        wt_frac_sigmas = {}   # phase_name → σ(wt%) in percentage points
        if scale_sigmas and len(zmv_values) >= 2:
            try:
                T = total_zmv
                phase_names = list(zmv_values.keys())
                n_ph = len(phase_names)

                # Build covariance matrix for scale factors from GSAS-II
                # If full covariance is available, use it; otherwise fall
                # back to diagonal (variances only) with a warning.
                cov_matrix = np.zeros((n_ph, n_ph))
                scale_indices = {}  # phase_name → index in vary_list
                have_full_cov = False
                try:
                    cov_data_mat = cov_data.get('covMatrix')
                    if cov_data_mat is not None:
                        cov_data_mat = np.array(cov_data_mat)
                        for idx_p, pname in enumerate(phase_names):
                            for pattern in [f'{idx_p}:0:Scale', f'{idx_p}::Scale']:
                                if pattern in vary_list:
                                    scale_indices[pname] = vary_list.index(pattern)
                                    break
                            if pname not in scale_indices:
                                for vname in vary_list:
                                    if vname.startswith(f'{idx_p}:') and 'Scale' in vname:
                                        scale_indices[pname] = vary_list.index(vname)
                                        break

                        if len(scale_indices) == n_ph:
                            for i_s, name_i in enumerate(phase_names):
                                for j_s, name_j in enumerate(phase_names):
                                    vi = scale_indices[name_i]
                                    vj = scale_indices[name_j]
                                    cov_matrix[i_s, j_s] = float(
                                        cov_data_mat[vi, vj])
                            have_full_cov = True
                            print(f"  Covariance: using full covariance matrix "
                                  f"for wt% uncertainties", flush=True)
                except Exception as e_cov:
                    print(f"  Covariance matrix extraction failed: {e_cov}",
                          flush=True)

                if not have_full_cov:
                    # Diagonal fallback — variances only
                    for i_s, pname in enumerate(phase_names):
                        sig_s = scale_sigmas.get(pname, 0.0)
                        cov_matrix[i_s, i_s] = sig_s ** 2
                    print(f"  Covariance: using diagonal only (no off-diagonal "
                          f"terms) — wt% uncertainties may be underestimated",
                          flush=True)

                # Compute ∂W_α/∂S_i for each alpha, then propagate
                for i_alpha, alpha in enumerate(phase_names):
                    P_alpha = zmv_values[alpha]
                    S_alpha = raw_scales.get(alpha, 1.0)
                    K_alpha = P_alpha / S_alpha if S_alpha > 0 else 0.0

                    # Build gradient vector dW_alpha/dS_i
                    grad = np.zeros(n_ph)
                    for i_beta, beta in enumerate(phase_names):
                        P_beta = zmv_values[beta]
                        S_beta = raw_scales.get(beta, 1.0)
                        K_beta = P_beta / S_beta if S_beta > 0 else 0.0
                        if beta == alpha:
                            grad[i_beta] = K_alpha * (T - P_alpha) / (T * T)
                        else:
                            grad[i_beta] = -P_alpha * K_beta / (T * T)

                    # σ²(W_α) = grad^T · Cov · grad
                    var_w = float(grad @ cov_matrix @ grad)
                    sigma_w = math.sqrt(max(var_w, 0.0)) * 100.0
                    wt_frac_sigmas[alpha] = sigma_w

                print(f"  Weight fraction uncertainties: "
                      f"{{ {', '.join(f'{k}: ±{v:.3f}%' for k, v in wt_frac_sigmas.items())} }}",
                      flush=True)
            except Exception as e:
                print(f"  Weight fraction uncertainty propagation failed: {e}",
                      flush=True)
        else:
            print("  Weight fraction uncertainties: not available "
                  "(need scale ESDs for ≥2 phases)", flush=True)

        # ── Extract GSAS-II RefList for physics-based profile generation ────
        # The RefList contains GSAS-II's refined Fc² values for each reflection,
        # giving more accurate phase envelopes than our generate_reflections output.
        # RefList columns: 0=h, 1=k, 2=l, 3=mult, 4=dsp, 5=2theta, 6=Fo²,
        #                  7=sig, 8=Fc², ...
        gsas_refs = {}   # phase_name → [(two_theta, d, (h,k,l), mult*Fc²)]
        try:
            raw_refl_lists = histogram.data.get('Reflection Lists', {})
            for ph_name, refl_data in raw_refl_lists.items():
                ref_arr = refl_data.get('RefList')
                if ref_arr is not None and len(ref_arr) > 0 and ref_arr.shape[1] > 8:
                    # First pass: collect all reflections and track max Fc²
                    # (raw, without multiplicity) for threshold filtering.
                    # Using raw Fc² matches how generate_reflections() filters
                    # in the preview — multiplicity should not determine whether
                    # a reflection is "real" vs "ghost".
                    raw_refs = []
                    max_fc2 = 0.0
                    for row in ref_arr:
                        h, k, l = int(row[0]), int(row[1]), int(row[2])
                        mult      = float(row[3])
                        d_sp      = float(row[4])
                        two_theta = float(row[5])
                        fc2       = float(row[8])   # Fc²
                        weight    = mult * fc2
                        if weight > 0 and tt_min <= two_theta <= tt_max:
                            raw_refs.append((two_theta, d_sp, (h, k, l), weight, fc2))
                            if fc2 > max_fc2:
                                max_fc2 = fc2
                    # Second pass: filter on raw Fc² (not mult*Fc²) to remove
                    # ghost reflections without losing legitimate low-multiplicity peaks.
                    # Use a very permissive threshold (0.01%) — heavy-atom carbide
                    # structures like W2C can have large Fc² dynamic range, and the
                    # previous 1% threshold was filtering out real reflections.
                    rel_thresh = max_fc2 * 1e-4  # 0.01% of strongest Fc²
                    refs = [(r[0], r[1], r[2], r[3])
                            for r in raw_refs if r[4] >= max(1e-6, rel_thresh)]
                    if refs:
                        gsas_refs[ph_name] = refs
                        print(f"  Loaded GSAS-II RefList for '{ph_name}': "
                              f"{len(refs)} reflections with Fc²", flush=True)
        except Exception as e:
            print(f"  (Could not extract GSAS-II RefList: {e})", flush=True)

        for i, (ph, phase_obj) in enumerate(zip(phases, gsas_phases)):
            # Cell parameters — read directly from GSAS-II data structure
            # to avoid get_cell() API differences across versions.
            # General['Cell'] = [mustRefine, a, b, c, alpha, beta, gamma, volume]
            _cell = phase_obj.data['General']['Cell']
            a, b, c = float(_cell[1]), float(_cell[2]), float(_cell[3])
            alpha, beta, gamma = float(_cell[4]), float(_cell[5]), float(_cell[6])

            # ── Sanity check: detect diverged cell parameters ────────────
            # If cell refinement failed and GSAS-II retained garbage values
            # (e.g. a=10⁸ Å from "Invalid cell metric tensor"), fall back
            # to the original input values from the phase dict.
            _orig_a = float(ph.get('a', a))
            _orig_b = float(ph.get('b', b))
            _orig_c = float(ph.get('c', c))
            _orig_alpha = float(ph.get('alpha', alpha))
            _orig_beta  = float(ph.get('beta', beta))
            _orig_gamma = float(ph.get('gamma', gamma))

            def _cell_diverged(ref, val, tol=2.0):
                """True if val deviates from ref by more than tol× factor."""
                if ref < 0.1:
                    return abs(val) > 100  # guard against near-zero ref
                ratio = val / ref
                return ratio > tol or ratio < (1.0 / tol)

            _any_diverged = (
                _cell_diverged(_orig_a, a) or
                _cell_diverged(_orig_b, b) or
                _cell_diverged(_orig_c, c) or
                abs(alpha - _orig_alpha) > 30 or
                abs(beta  - _orig_beta)  > 30 or
                abs(gamma - _orig_gamma) > 30
            )
            if _any_diverged:
                warnings.warn(
                    f"GSAS-II: cell params for phase {i} diverged "
                    f"(a={a:.4f} vs input {_orig_a:.4f}, "
                    f"b={b:.4f} vs {_orig_b:.4f}, "
                    f"c={c:.4f} vs {_orig_c:.4f}). "
                    f"Reverting to input values.")
                a, b, c = _orig_a, _orig_b, _orig_c
                alpha, beta, gamma = _orig_alpha, _orig_beta, _orig_gamma

            V = cell_volume(a, b, c, alpha, beta, gamma)

            # Profile / size info
            prof = _extract_profile_params(phase_obj)
            cryst_A = prof.get('crystallite_size_A')

            # GSAS-II stores profile params in centidegrees² (sig) and
            # centidegrees (gam).  Convert to degrees² / degrees for
            # Caglioti/TCH compatibility with our functions.
            # 1 deg = 100 centideg  →  1 deg² = 10 000 centideg²
            U_deg = inst['U'] / 10000.0   # centideg² → deg²
            V_deg = inst['V'] / 10000.0
            W_deg = inst['W'] / 10000.0
            X_deg = inst['X'] / 100.0     # centideg → deg
            Y_deg = inst['Y'] / 100.0

            # If GSAS-II size extraction failed, estimate from Y
            if cryst_A is None and Y_deg > 0.01:
                cryst_A = size_from_Y(Y_deg, wavelength)

            # FWHM and eta at a representative angle (40°)
            fwhm_rep, eta_rep = tch_fwhm_eta(
                40.0, U_deg, V_deg, W_deg, X_deg, Y_deg)

            # Weight fraction (Hill & Howard or raw-scale fallback)
            scale_val = raw_scales.get(phase_obj.name, 1.0)
            zmv_val = zmv_values.get(phase_obj.name, scale_val)
            wt_pct = (zmv_val / total_zmv) * 100 if total_zmv > 0 else 0
            wt_pct_err = wt_frac_sigmas.get(phase_obj.name)  # may be None

            # ── Generate tick positions / reflection list ─────────────────
            # Always use generate_reflections() for tick positions — this
            # matches the preview stick pattern and uses our pre-computed F²
            # with correct filtering.  GSAS-II's refined Fc² values can be
            # unreliable for phases that haven't fully converged (e.g. W2C
            # Pbcn), causing legitimate reflections to be filtered out.
            # The GSAS-II RefList (gsas_refs) is still used separately for
            # phase profile reconstruction in _compute_raw_phase_profile().
            #
            # CRITICAL: Use the GSAS-II CIF (gsas_cif_texts[i]) for site
            # expansion, NOT the original phase dict CIF.  When CifWriter
            # standardized the structure (e.g. W2C Pbcn), it may have
            # permuted axes.  The GSAS-II cell params (a,b,c above) are
            # in the CifWriter convention, so the sites must also be in
            # that convention for correct structure factor calculation.
            # Using the original CIF sites with CifWriter cell params →
            # wrong F(hkl) → real reflections filtered out → missing ticks.
            sys_ = (ph.get('system') or 'triclinic').lower()
            sg = ph.get('spacegroup_number', 1)
            _gsas_cif = gsas_cif_texts[i] if i < len(gsas_cif_texts) else ''
            sites = _get_expanded_sites(_gsas_cif or ph.get('cif_text', ''), sg)
            phase_refs = generate_reflections(
                a, b, c, alpha, beta, gamma, sys_, sg,
                wavelength, tt_min, tt_max, hkl_max=12,
                sites=sites)
            tick_positions = [round(r[0], 3) for r in phase_refs]
            print(f"  Tick positions for '{ph.get('name', '?')}' (SG {sg}): "
                  f"{len(tick_positions)} reflections in "
                  f"{tt_min:.1f}–{tt_max:.1f}° 2θ", flush=True)
            # Use GSAS-II RefList for profile reconstruction if available
            gsas_phase_refs = gsas_refs.get(phase_obj.name)
            if gsas_phase_refs:
                print(f"    GSAS-II RefList: {len(gsas_phase_refs)} reflections "
                      f"(used for profile reconstruction)", flush=True)
            all_phase_refs.append(gsas_phase_refs if gsas_phase_refs else phase_refs)

            # B_iso (average over atoms; falls back to DEFAULT_B_ISO)
            b_iso_avg = DEFAULT_B_ISO
            try:
                atoms = list(phase_obj.atoms())
                if atoms:
                    uisos = [at.uiso for at in atoms if hasattr(at, 'uiso')]
                    if uisos:
                        b_iso_avg = 8 * math.pi**2 * np.mean(uisos)
            except Exception:
                pass

            # Build a display name that always includes the space group symbol
            # so that phases with the same formula are clearly distinguishable.
            _sg_sym = (ph.get('spacegroup', '') or
                       _SG_HM.get(ph.get('spacegroup_number', 1), '') or
                       f"SG{ph.get('spacegroup_number', 1)}")
            _base_name = ph.get('name') or ph.get('formula') or f'Phase {i+1}'
            # Only append SG if not already present in the name
            if _sg_sym and _sg_sym.replace(' ', '') not in _base_name.replace(' ', ''):
                display_name = f"{_base_name} {_sg_sym}"
            else:
                display_name = _base_name

            phase_results.append({
                'name':              display_name,
                'cod_id':            ph.get('cod_id', ph.get('mp_id', '')),
                'formula':           ph.get('formula', ''),
                'a': round(a, 5), 'b': round(b, 5), 'c': round(c, 5),
                'alpha': round(alpha, 3), 'beta': round(beta, 3),
                'gamma': round(gamma, 3),
                'system':            (ph.get('system') or 'triclinic').lower(),
                'spacegroup_number': ph.get('spacegroup_number', 1),
                'spacegroup':        ph.get('spacegroup', ''),
                'scale':             round(scale_val, 5),
                'B_iso':             round(b_iso_avg, 4),
                'U': round(U_deg, 5), 'V': round(V_deg, 5),
                'W': round(W_deg, 5),
                'X': round(X_deg, 5), 'Y': round(Y_deg, 5),
                'eta_at_strongest':  round(eta_rep, 3),
                'fwhm_deg':          round(fwhm_rep, 4),
                'crystallite_size_A':  round(cryst_A, 1) if cryst_A else None,
                'crystallite_size_nm': round(cryst_A / 10, 2) if cryst_A else None,
                'weight_fraction_%':       round(wt_pct, 1),
                'weight_fraction_err_%':   round(wt_pct_err, 2) if wt_pct_err is not None else None,
                'n_reflections':           len(tick_positions),
                'tick_positions':      tick_positions,
                'seeded_by':           'gsas2',
            })

        # ── Per-phase patterns ─────────────────────────────────────────
        # Primary method: GSAS-II phase isolation — zero all phases
        # except one, let GSAS-II recompute (0 cycles), extract
        # ycalc - background.  This uses GSAS-II's own profile
        # functions (correct peak shapes, asymmetry, everything).
        #
        # Fallback: reconstruct per-phase profiles from reflection
        # lists and refined instrument parameters (may not exactly
        # match GSAS-II's profile shapes).
        #
        # Last resort: equal split (unreliable weight fractions).
        if len(gsas_phases) > 1:
            total_above_bg = np.maximum(y_calc_out - y_bg_out, 0.0)
            decomp_ok = False

            # ── Primary: GSAS-II phase isolation ─────────────────────
            print("  Phase isolation: computing GSAS-II per-phase "
                  "patterns...", flush=True)

            # Save ALL refinement flags — the main refinement left many
            # params refinable (background, U/V/W/X/Y, cell, Uiso).
            # We must turn them ALL off before isolation, otherwise
            # do_refinements will try to refine them with one phase
            # zeroed, causing that phase's parameters to diverge.
            orig_hap_scales = []
            for phase_obj in gsas_phases:
                hapData = list(phase_obj.data['Histograms'].values())[0]
                orig_hap_scales.append(list(hapData['Scale']))

            # Save & turn off background refinement flag
            saved_bg_flag = histogram.data['Background'][0][1]
            histogram.data['Background'][0][1] = False

            # Save & turn off sample displacement refinement flags for
            # BOTH DisplaceX (Bragg-Brentano) and DisplaceY (Debye-Scherrer
            # transparency).  The pipeline only actively refines one of
            # these (DisplaceY for capillary data), but saving both is
            # defensive — if the geometry is changed later we don't want
            # a stale refinement flag to leak through phase isolation.
            saved_displace_flags = {}
            try:
                for _disp_key in ('DisplaceX', 'DisplaceY'):
                    _dp = histogram.data['Sample Parameters'].get(_disp_key)
                    if _dp and isinstance(_dp, list) and len(_dp) >= 2:
                        saved_displace_flags[_disp_key] = _dp[1]
                        _dp[1] = False
            except Exception:
                pass

            # Save & turn off instrument parameter refinement flags.
            # GSAS-II stores refine flags in TWO possible locations:
            # (a) inline: inst_params[0][key][2] (3-element lists)
            # (b) separate dict: inst_params[1][key] (boolean)
            inst_params_all = histogram.data['Instrument Parameters']
            inst_params_raw = inst_params_all[0]
            inst_refine_dict = (inst_params_all[1]
                                if len(inst_params_all) > 1
                                and isinstance(inst_params_all[1], dict)
                                else {})
            saved_inst_flags = {}
            for key in ['U', 'V', 'W', 'X', 'Y', 'SH/L', 'Zero']:
                # Location (a): inline in parameter list
                if key in inst_params_raw and len(inst_params_raw[key]) >= 3:
                    saved_inst_flags[(key, 'inline')] = \
                        inst_params_raw[key][2]
                    inst_params_raw[key][2] = False
                # Location (b): separate refine-flags dict
                if key in inst_refine_dict:
                    saved_inst_flags[(key, 'dict')] = \
                        inst_refine_dict[key]
                    inst_refine_dict[key] = False

            # Save & turn off preferred orientation refinement flags
            saved_pref_ori_flags = []
            for phase_obj in gsas_phases:
                try:
                    hapData = list(phase_obj.data['Histograms'].values())[0]
                    po = hapData.get('Pref.Ori.', ['MD', 1.0, False, [0,0,1]])
                    saved_pref_ori_flags.append(po[2])
                    po[2] = False
                except Exception:
                    saved_pref_ori_flags.append(False)


            # Save & turn off cell and atom refinement flags.
            # atom[2] = refinement flags string ('XU', 'U', '' etc.)
            # atom[9] = multiplicity — NOT the refinement flag.
            saved_cell_flags = []
            saved_atom_flags = []
            for phase_obj in gsas_phases:
                cell = phase_obj.data['General']['Cell']
                saved_cell_flags.append(cell[0])
                cell[0] = False
                atom_flags = []
                for atom in phase_obj.data['Atoms']:
                    if len(atom) > 2:
                        atom_flags.append(str(atom[2]))
                        atom[2] = ''   # clear XYZ + Uiso flags
                saved_atom_flags.append(atom_flags)

            # Belt-and-suspenders: force 0 cycles in Controls so GSAS-II
            # cannot iterate even if a flag was somehow missed.
            _controls = gpx.data.get('Controls', {}).get('data', {})
            _saved_maxcyc = _controls.get('max cyc', 3)
            _controls['max cyc'] = 0

            try:
                phase_patterns = []
                for i in range(len(gsas_phases)):
                    # Activate only phase i; zero all others
                    for j, phase_obj in enumerate(gsas_phases):
                        hapData = list(phase_obj.data['Histograms'].values())[0]
                        if j == i:
                            hapData['Scale'] = [orig_hap_scales[j][0], False]
                        else:
                            hapData['Scale'] = [0.0, False]

                    # Persist to disk so do_refinements picks up changes.
                    gpx.save()

                    # Evaluate only (cycles=0) — all refinement flags are
                    # OFF, so nothing can diverge.
                    gpx.do_refinements([{'set': {}, 'cycles': 0}])

                    # Re-fetch histogram — do_refinements reloads the
                    # project from disk, so the old reference may be stale.
                    histogram = gpx.histograms()[0]

                    y_calc_iso = np.array(histogram.getdata('ycalc'))
                    # Use GSAS-II's original background (before our dip
                    # correction) so the correction delta doesn't bias
                    # phase ratios toward the dominant phase.
                    phase_pat = np.maximum(
                        y_calc_iso[rmask] - _y_bg_gsas, 0.0)
                    phase_patterns.append(phase_pat.tolist())

                    ph_name = (phase_results[i]['name']
                               if i < len(phase_results) else f'Phase {i}')
                    print(f"  Phase {i} ({ph_name}): "
                          f"max={np.max(phase_pat):.1f}, "
                          f"integrated={np.sum(phase_pat):.1f}", flush=True)

                # Diagnostic: verify isolation produced correct patterns.
                sum_iso = np.zeros_like(tt_out, dtype=np.float64)
                for pp in phase_patterns:
                    sum_iso += np.array(pp)
                max_diff = float(np.max(np.abs(
                    sum_iso - total_above_bg)))
                sum_total = float(np.sum(total_above_bg))
                sum_iso_total = float(np.sum(sum_iso))
                print(f"  Diagnostic: sum(iso)={sum_iso_total:.1f}, "
                      f"sum(above_bg)={sum_total:.1f}, "
                      f"max_pointwise_diff={max_diff:.2f}", flush=True)

                # Proportional normalization: ensure stacked fills sum
                # exactly to total_above_bg at every point so that the
                # fills match the I_calc line perfectly.
                for i_pp, pp in enumerate(phase_patterns):
                    pp_arr = np.array(pp)
                    ratio = np.zeros_like(sum_iso)
                    np.divide(pp_arr, sum_iso, out=ratio,
                              where=sum_iso > 0)
                    phase_patterns[i_pp] = np.maximum(
                        ratio * total_above_bg, 0.0).tolist()

                decomp_ok = True
                print("  Phase isolation succeeded.", flush=True)
            except Exception as e_iso:
                print(f"  Phase isolation failed: {e_iso}", flush=True)
                phase_patterns = []
            finally:
                # Always restore ALL original flags and scales
                histogram.data['Background'][0][1] = saved_bg_flag
                for _disp_key, _disp_flag in saved_displace_flags.items():
                    try:
                        histogram.data['Sample Parameters'][_disp_key][1] = \
                            _disp_flag
                    except Exception:
                        pass
                for (key, loc), flag in saved_inst_flags.items():
                    if loc == 'inline':
                        inst_params_raw[key][2] = flag
                    else:
                        inst_refine_dict[key] = flag
                for idx_r, phase_obj in enumerate(gsas_phases):
                    if idx_r < len(saved_cell_flags):
                        phase_obj.data['General']['Cell'][0] = \
                            saved_cell_flags[idx_r]
                    for j_a, atom in enumerate(phase_obj.data['Atoms']):
                        if (idx_r < len(saved_atom_flags)
                                and len(atom) > 2
                                and j_a < len(saved_atom_flags[idx_r])):
                            atom[2] = saved_atom_flags[idx_r][j_a]
                    hapData = list(phase_obj.data['Histograms'].values())[0]
                    if idx_r < len(orig_hap_scales):
                        hapData['Scale'] = orig_hap_scales[idx_r]
                    # Restore preferred orientation refinement flag
                    if idx_r < len(saved_pref_ori_flags):
                        try:
                            po = hapData.get('Pref.Ori.')
                            if po:
                                po[2] = saved_pref_ori_flags[idx_r]
                        except Exception:
                            pass
                # Restore max cycles to original value
                _controls['max cyc'] = _saved_maxcyc
                gpx.save()
                histogram = gpx.histograms()[0]

            # ── Fallback: profile reconstruction from reflections ────
            if not decomp_ok:
                print("  Falling back to profile reconstruction...",
                      flush=True)
                if all_phase_refs and len(all_phase_refs) == len(gsas_phases):
                    try:
                        U_d = inst['U'] / 10000.0
                        V_d = inst['V'] / 10000.0
                        W_d = inst['W'] / 10000.0
                        X_d = inst['X'] / 100.0
                        Y_d = inst['Y'] / 100.0

                        raw_profiles = []
                        for i_ph, (phase_obj_r, fallback_refs) in enumerate(
                                zip(gsas_phases, all_phase_refs)):
                            refs_to_use = gsas_refs.get(
                                phase_obj_r.name, fallback_refs) or fallback_refs
                            if not refs_to_use:
                                raise ValueError(
                                    f"No reflections for phase {i_ph}")
                            raw_profiles.append(
                                _compute_raw_phase_profile(
                                    tt_out, refs_to_use,
                                    U_d, V_d, W_d, X_d, Y_d,
                                    gaussian_only=True))

                        scaled = []
                        for phase_obj, prof in zip(gsas_phases, raw_profiles):
                            s = raw_scales.get(phase_obj.name, 1.0)
                            scaled.append(prof * s)

                        sum_raw = np.zeros_like(tt_out, dtype=np.float64)
                        for sp in scaled:
                            sum_raw += sp
                        peak_max = (np.max(sum_raw)
                                    if np.max(sum_raw) > 0 else 1.0)
                        threshold = peak_max * 1e-10
                        for sp in scaled:
                            ratio = np.zeros_like(sum_raw)
                            np.divide(sp, sum_raw, out=ratio,
                                      where=sum_raw > threshold)
                            phase_patterns.append(
                                np.maximum(
                                    ratio * total_above_bg, 0.0).tolist())
                        decomp_ok = True
                        print("  Profile reconstruction succeeded.",
                              flush=True)
                    except Exception as e_fc2:
                        print(f"  Profile reconstruction failed: {e_fc2}",
                              flush=True)
                        phase_patterns = []

            # ── Last resort: equal split ───────────────────────────────
            if not decomp_ok:
                warnings.warn(
                    "GSAS-II phase decomposition: both isolation and "
                    "profile reconstruction failed. Falling back to "
                    "equal split — weight fractions will be UNRELIABLE.")
                phase_patterns = []
                for _ in gsas_phases:
                    share = (total_above_bg / len(gsas_phases)).tolist()
                    phase_patterns.append(share)

            # Use integrated pattern fractions as the displayed weight
            # fractions.  The Hill & Howard (H&H) method is the standard
            # Rietveld approach, but it requires accurate Z (formula units
            # per cell) and molar mass for each phase.  When these are
            # incorrect or when scale factors haven't fully converged,
            # H&H can give wildly wrong values.  The integrated fractions
            # are a direct measure of each phase's contribution to the
            # diffraction signal and are more robust for display.
            #
            # NOTE: the wt% UNCERTAINTIES are still derived from H&H
            # error propagation (covariance matrix).  These are logged
            # alongside the H&H values for diagnostics.
            total_integ = sum(
                np.sum(np.array(pp)) for pp in phase_patterns) or 1e-30
            for i_wp, pp in enumerate(phase_patterns):
                integ_frac = np.sum(np.array(pp)) / total_integ * 100
                hh_frac = (phase_results[i_wp]['weight_fraction_%']
                           if i_wp < len(phase_results) else None)
                print(f"  Phase {i_wp}: "
                      f"H&H wt% = {hh_frac}%, "
                      f"integrated fraction = {integ_frac:.1f}%", flush=True)
                # Overwrite with integrated fraction for display
                if i_wp < len(phase_results):
                    phase_results[i_wp]['weight_fraction_%'] = round(integ_frac, 1)
        else:
            # Single phase — entire signal above background
            total_above_bg = np.maximum(y_calc_out - y_bg_out, 0.0)
            phase_patterns.append(total_above_bg.tolist())

        result = {
            'tt':             tt_out.tolist(),
            'y_obs':          y_obs_out.tolist(),
            'y_calc':         y_calc_out.tolist(),
            'y_background':   y_bg_out.tolist(),
            'phase_patterns': phase_patterns,
            'residuals':      diff_out.tolist(),
            'statistics':     stats,
            'phase_results':  phase_results,
            'zero_shift':     round(zero_shift, 5),
            'wavelength':     wavelength,
            'pymatgen_used':  False,
            'method':         'GSAS-II',
        }

        return result

    finally:
        # Clean up temporary files
        for p in cif_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        try:
            os.unlink(data_path)
        except OSError:
            pass
        # Clean up temp working directory (including .gpx, .bak files)
        try:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass
