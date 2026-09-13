"""
generate_mp1552_fixture.py
──────────────────────────
Fetches mp-1552 (Mo2C Pbcn) from the Materials Project API fresh,
applies the conventional-standard-cell conversion, validates the result,
and writes fixtures/mo2c_pbcn_mp_1552.cif if everything checks out.

Run from the toolkit root:
    python generate_mp1552_fixture.py --api-key YOUR_API_KEY

Or with the key already in config.yaml:
    python generate_mp1552_fixture.py
"""

import os, sys, argparse, textwrap

# ── Argument parsing ─────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Fetch and validate mp-1552 fixture")
parser.add_argument("--api-key", default=None,
                    help="Materials Project API key (falls back to config.yaml)")
parser.add_argument("--out-dir", default="fixtures",
                    help="Directory to write the .cif fixture (default: fixtures/)")
parser.add_argument("--dry-run", action="store_true",
                    help="Print validation info but do not write the fixture file")
args = parser.parse_args()

# ── Resolve API key ──────────────────────────────────────────────────────────

api_key = args.api_key
if not api_key:
    try:
        import yaml
        cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        api_key = (cfg.get("mp_api_key")
                   or cfg.get("materials_project", {}).get("api_key"))
    except Exception as e:
        print(f"  Could not read config.yaml: {e}")

if not api_key:
    sys.exit("ERROR: No API key found. Pass --api-key or set mp_api_key in config.yaml.")

# ── Fetch mp-1552 raw from MP, bypassing cache ───────────────────────────────

import requests

MP_SUMMARY = "https://api.materialsproject.org/materials/summary/"
MP_ID = "mp-1552"
EXPECTED_SG = 60          # Pbcn
EXPECTED_FORMULA = "Mo2C"
EXPECTED_ATOMS_CONV = 12  # 8 Mo + 4 C in conventional Pbcn cell

print(f"\n{'='*60}")
print(f"  Fetching {MP_ID} fresh from Materials Project API")
print(f"{'='*60}")

resp = requests.get(
    MP_SUMMARY,
    headers={"X-API-KEY": api_key, "Accept": "application/json"},
    params={
        "material_ids": MP_ID,
        "_fields": "material_id,formula_pretty,symmetry,structure",
        "deprecated": "false",
        "_limit": 1,
    },
    timeout=20,
)
resp.raise_for_status()
data = resp.json().get("data", [])
if not data:
    sys.exit(f"ERROR: No entry found for {MP_ID}. Check your API key and MP status.")

entry  = data[0]
struct_dict = entry["structure"]
formula     = entry.get("formula_pretty", "?")
sym         = entry.get("symmetry") or {}
sg_num_mp   = int(sym.get("number") or 0)
sg_sym_mp   = sym.get("symbol", "?")

print(f"\n  MP reports:")
print(f"    formula       : {formula}")
print(f"    space group   : {sg_sym_mp} (No. {sg_num_mp})")
print(f"    crystal system: {sym.get('crystal_system', '?')}")

lattice = struct_dict.get("lattice", {})
print(f"    primitive cell: a={lattice.get('a'):.4f}  b={lattice.get('b'):.4f}  "
      f"c={lattice.get('c'):.4f}  α={lattice.get('alpha'):.2f}  "
      f"β={lattice.get('beta'):.2f}  γ={lattice.get('gamma'):.2f}")
print(f"    primitive atoms: {len(struct_dict.get('sites', []))}")

# ── Convert to conventional standard cell ────────────────────────────────────

print(f"\n  Applying conventional-standard-cell conversion...")

try:
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.io.cif import CifWriter
except ImportError as e:
    sys.exit(f"ERROR: pymatgen not available: {e}")

prim_struct = Structure.from_dict(struct_dict)
print(f"  Primitive structure: {len(prim_struct)} atoms, "
      f"SG detected by pymatgen: ", end="")

try:
    sga_prim = SpacegroupAnalyzer(prim_struct, symprec=0.1)
    print(f"{sga_prim.get_space_group_symbol()} ({sga_prim.get_space_group_number()})")
except Exception as e:
    print(f"detection failed ({e})")

# Convert
for symprec in (0.01, 0.05, 0.1, 0.2):
    try:
        sga = SpacegroupAnalyzer(prim_struct, symprec=symprec)
        conv = sga.get_conventional_standard_structure()
        sg_detected = sga.get_space_group_number()
        print(f"  symprec={symprec}: conventional={len(conv)} atoms, "
              f"SG={sga.get_space_group_symbol()} ({sg_detected})")
        if sg_detected == EXPECTED_SG and len(conv) >= len(prim_struct):
            conv_struct = conv
            used_symprec = symprec
            print(f"  ✓ Correct SG {EXPECTED_SG} (Pbcn) found at symprec={symprec}")
            break
    except Exception as e:
        print(f"  symprec={symprec}: failed ({e})")
else:
    print("\n  WARNING: Could not confirm SG 60 at any symprec.")
    print("  Proceeding with symprec=0.1 result anyway.")
    sga = SpacegroupAnalyzer(prim_struct, symprec=0.1)
    conv_struct = sga.get_conventional_standard_structure()
    used_symprec = 0.1

conv_lattice = conv_struct.lattice
print(f"\n  Conventional cell:")
print(f"    a={conv_lattice.a:.4f}  b={conv_lattice.b:.4f}  c={conv_lattice.c:.4f}")
print(f"    α={conv_lattice.alpha:.2f}  β={conv_lattice.beta:.2f}  γ={conv_lattice.gamma:.2f}")
print(f"    atoms={len(conv_struct)}")

# ── Write CIF via CifWriter ───────────────────────────────────────────────────

print(f"\n  Writing CIF with CifWriter(symprec={used_symprec})...")

writer = CifWriter(conv_struct, symprec=used_symprec)
cif_text = str(writer)

# ── Validate the produced CIF ─────────────────────────────────────────────────

print(f"\n{'='*60}")
print("  VALIDATION")
print(f"{'='*60}")

# Parse back with pymatgen
from pymatgen.io.cif import CifParser as PymatgenCifParser
import io

parser_back = PymatgenCifParser(io.StringIO(cif_text))
try:
    structs_back = parser_back.parse_structures(primitive=False)
    s_back = structs_back[0]
    sga_back = SpacegroupAnalyzer(s_back, symprec=0.1)
    sg_back = sga_back.get_space_group_number()
    sym_back = sga_back.get_space_group_symbol()
    print(f"  Re-parsed CIF: {len(s_back)} atoms, SG={sym_back} ({sg_back})")
    lat_back = s_back.lattice
    print(f"  Lattice: a={lat_back.a:.4f}  b={lat_back.b:.4f}  c={lat_back.c:.4f}")
except Exception as e:
    print(f"  Re-parse failed: {e}")
    sg_back = 0

# Checks
issues = []
if formula != EXPECTED_FORMULA:
    issues.append(f"Formula mismatch: got {formula}, expected {EXPECTED_FORMULA}")
if sg_num_mp != EXPECTED_SG:
    issues.append(f"MP SG mismatch: MP reports {sg_num_mp}, expected {EXPECTED_SG}")
if len(conv_struct) != EXPECTED_ATOMS_CONV:
    issues.append(f"Atom count: got {len(conv_struct)}, expected {EXPECTED_ATOMS_CONV} "
                  f"(8 Mo + 4 C for Pbcn conventional cell)")
if sg_back != EXPECTED_SG:
    issues.append(f"Re-parsed SG: got {sg_back}, expected {EXPECTED_SG}")

# Expected Pbcn Mo2C lattice params (from Epicier et al. 1988 + MP typical range)
a_ok = 4.5 < conv_lattice.a < 5.0
b_ok = 5.7 < conv_lattice.b < 6.4
c_ok = 4.9 < conv_lattice.c < 5.5
if not (a_ok and b_ok and c_ok):
    issues.append(f"Lattice params outside expected Pbcn Mo2C range "
                  f"(a~4.73, b~6.04, c~5.20 Å): got "
                  f"a={conv_lattice.a:.3f} b={conv_lattice.b:.3f} c={conv_lattice.c:.3f}")

# Print Wyckoff sites
print(f"\n  Sites in conventional cell:")
site_counts = {}
for site in conv_struct:
    el = str(site.specie)
    site_counts[el] = site_counts.get(el, 0) + 1
for el, n in sorted(site_counts.items()):
    print(f"    {el}: {n} sites")

print(f"\n  Unique sites (asymmetric unit from SGA):")
try:
    sym_dataset = sga.get_symmetry_dataset()
    wyckoff = sym_dataset.get("wyckoffs") or []
    equiv   = sym_dataset.get("equivalent_atoms") or []
    seen = set()
    for i, site in enumerate(conv_struct):
        if equiv[i] not in seen:
            seen.add(equiv[i])
            abc = site.frac_coords
            w   = wyckoff[i] if i < len(wyckoff) else "?"
            print(f"    {str(site.specie):<4} Wyckoff {w:>3}   "
                  f"({abc[0]:.4f}, {abc[1]:.4f}, {abc[2]:.4f})")
except Exception as e:
    print(f"    (Wyckoff analysis failed: {e})")

# ── Print CIF excerpt ─────────────────────────────────────────────────────────

print(f"\n  CIF preview (atom_site block):")
in_atom_block = False
for line in cif_text.splitlines():
    if "_atom_site" in line and "loop_" not in line:
        in_atom_block = True
    if in_atom_block:
        print(f"    {line}")
    if in_atom_block and line.strip() == "" and "_atom_site" not in line:
        break

# ── Summary ──────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
if issues:
    print("  ⚠  ISSUES FOUND:")
    for iss in issues:
        print(f"     • {iss}")
    print("\n  The CIF has problems — do NOT use as a fixture without manual review.")
    print("  CIF text printed below for inspection:")
    print(f"{'─'*60}")
    print(cif_text)
else:
    print("  ✓ All checks passed — CIF looks correct for Mo2C Pbcn")
    print(f"    Formula: {EXPECTED_FORMULA}, SG: {EXPECTED_SG} (Pbcn), "
          f"Atoms: {len(conv_struct)}")

print(f"{'='*60}\n")

# ── Write fixture ─────────────────────────────────────────────────────────────

if not issues and not args.dry_run:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mo2c_pbcn_mp_1552.cif")

    # Add a header comment so the fixture is self-documenting
    header = textwrap.dedent(f"""\
        # Audited fixture for mp-1552 (Mo2C, Pbcn, SG 60)
        # Generated by generate_mp1552_fixture.py from fresh MP API fetch.
        # Conventional standard cell via SpacegroupAnalyzer(symprec={used_symprec}).
        # Reference: Epicier et al. Acta Cryst. B44 (1988) 78-84
        #   a~4.732  b~6.037  c~5.204 Å, SG Pbcn (#60), Z=4
        # Validated: formula={EXPECTED_FORMULA}, SG={EXPECTED_SG}, atoms={len(conv_struct)}
        #
    """)
    with open(out_path, "w") as f:
        f.write(header + cif_text)

    print(f"  ✓ Fixture written to: {out_path}")
    print(f"\n  Next steps:")
    print(f"  1. Add to _LOCAL_FIXTURES in mp_api.py:")
    print(f"       'mp-1552': 'mo2c_pbcn_mp_1552.cif',")
    print(f"  2. Clear the disk cache for this entry:")
    print(f"       from modules.xrd.cif_cache import get_cache")
    print(f"       c = get_cache()")
    print(f"       c._index.pop('mp:mp-1552:conv', None)  # new key")
    print(f"       c._index.pop('mp:mp-1552', None)       # old primitive key")
    print(f"       c._save_index()")
    print(f"  3. Restart the app server so modules reload.")

elif args.dry_run:
    print("  (--dry-run: fixture file not written)")
elif issues:
    print("  Fixture NOT written due to validation issues above.")
    sys.exit(1)
