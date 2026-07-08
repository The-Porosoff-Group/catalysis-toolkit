#!/usr/bin/env python
"""
Run GC processing from the terminal using the same backend as the GUI.

Examples:
    python scripts/gc_batch.py --file sample.xlsx --config rwgs --interactive
    python scripts/gc_batch.py --files "data/*.xlsx" --config co2_hydrogenation --inlet CO2=10 H2=40 Ar=15
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules import gc_processor


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^\w.\-]+", "_", str(value or "").strip())
    return safe.strip("_") or "sample"


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _prompt(label: str, default: Any = "") -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    value = input(f"{label}{suffix}: ").strip()
    return value if value else ("" if default is None else str(default))


def _repo_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _config_path(value: str) -> Path:
    raw = value.strip()
    candidate = Path(raw)
    if candidate.suffix.lower() == ".yaml":
        path = candidate if candidate.is_absolute() else ROOT / candidate
        if path.exists():
            return path.resolve()
    name = raw
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"
    path = ROOT / "modules" / "reaction_configs" / name
    if path.exists():
        return path.resolve()
    available = ", ".join(
        cfg["file"].replace(".yaml", "")
        for cfg in gc_processor.list_reaction_configs(ROOT / "modules" / "reaction_configs")
    )
    raise FileNotFoundError(f"Reaction config not found: {value}. Available: {available}")


def _parse_key_values(items: Optional[Iterable[str]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Blank key in {item!r}")
        out[key] = float(value)
    return out


def _default_inlet_flows(reaction_config: Dict[str, Any]) -> Dict[str, float]:
    flows: Dict[str, float] = {}
    for item in reaction_config.get("inlet_species", []):
        label = item.get("label")
        if label:
            flows[str(label)] = float(item.get("default_sccm") or 0.0)
    return flows


def _expand_files(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for item in patterns:
        raw = Path(item).expanduser()
        pattern = str(raw if raw.is_absolute() else ROOT / raw)
        matches = sorted(Path(p).resolve() for p in glob.glob(pattern))
        if matches:
            paths.extend(matches)
        else:
            paths.append(_repo_path(item))
    seen = set()
    unique = []
    for path in paths:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _interactive_metadata(args: argparse.Namespace, reaction_config: Dict[str, Any],
                          input_path: Path) -> tuple[Dict[str, Any], Dict[str, float], Optional[Path], int, int, Path]:
    catalyst_default = args.catalyst_id or input_path.stem
    metadata = {
        "catalyst_id": _prompt("Catalyst ID", catalyst_default),
        "temperature": _prompt("Temperature", args.temperature or ""),
        "pressure": _prompt("Pressure", args.pressure or ""),
        "ghsv": _prompt("GHSV", args.ghsv or ""),
        "run_duration_h": _prompt("Run duration h", args.run_duration_h or ""),
        "injection_interval_min": _prompt("Injection interval min", args.injection_interval_min or ""),
        "rejected_initial_injections": _prompt("Rejected initial reaction injections", args.rejected_initial_injections or 0),
        "registered_reaction_injections": _prompt("Registered reaction injections", args.registered_reaction_injections or ""),
        "bypass_omit_initial": _prompt("Bypass rows to omit first", args.bypass_omit_initial or 0),
        "bypass_points_used": _prompt("Bypass rows to average", args.bypass_points_used or 3),
        "same_file_bypass_mode": _prompt("Same-file bypass mode (auto/first/last/none)", args.same_file_bypass_mode or "auto"),
        "same_file_bypass_rows": _prompt("Same-file bypass rows checked", args.same_file_bypass_rows or 3),
        "c5_unknown_response_factor": _prompt("C5 unknown response factor", args.c5_unknown_response_factor or ""),
        "c6_unknown_response_factor": _prompt("C6 unknown response factor", args.c6_unknown_response_factor or ""),
        "plot_style": _prompt("Plot style (auto/time_on_stream/single_axis_stacked)", args.plot_style or "auto"),
        "notes": _prompt("Notes", args.notes or ""),
    }

    flows = _default_inlet_flows(reaction_config)
    flows.update(_parse_key_values(args.inlet))
    for key in list(flows):
        flows[key] = float(_prompt(f"Inlet {key} sccm", flows[key]))

    bypass = Path(args.bypass_file).resolve() if args.bypass_file else None
    bypass_answer = _prompt("Separate bypass workbook path, blank for same-file/manual", str(bypass or ""))
    bypass = _repo_path(bypass_answer) if bypass_answer else None

    ss_start = _as_int(_prompt("Steady-state start injection #", args.ss_start), 1) or 1
    ss_end = _as_int(_prompt("Steady-state end injection #", args.ss_end), 999) or 999
    out = _repo_path(_prompt("Output parent folder", str(args.out or ROOT / "results" / "gc_cli")))
    return metadata, flows, bypass, ss_start, ss_end, out


def _metadata_from_args(args: argparse.Namespace, input_path: Path) -> Dict[str, Any]:
    return {
        "catalyst_id": args.catalyst_id or input_path.stem,
        "temperature": args.temperature or "",
        "pressure": args.pressure or "",
        "ghsv": args.ghsv or "",
        "run_duration_h": args.run_duration_h or "",
        "injection_interval_min": args.injection_interval_min or "",
        "rejected_initial_injections": args.rejected_initial_injections or "",
        "registered_reaction_injections": args.registered_reaction_injections or "",
        "bypass_omit_initial": args.bypass_omit_initial or "",
        "bypass_points_used": args.bypass_points_used or "",
        "same_file_bypass_mode": args.same_file_bypass_mode or "auto",
        "same_file_bypass_rows": args.same_file_bypass_rows or "",
        "c5_unknown_response_factor": args.c5_unknown_response_factor or "",
        "c6_unknown_response_factor": args.c6_unknown_response_factor or "",
        "plot_style": args.plot_style or "auto",
        "notes": args.notes or "",
    }


def _run_one(input_path: Path, reaction_config: Dict[str, Any],
             metadata: Dict[str, Any], inlet_flows: Dict[str, float],
             ss_start: int, ss_end: int, out_parent: Path,
             bypass_path: Optional[Path]) -> Dict[str, Any]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id = _safe_name(str(metadata.get("catalyst_id") or input_path.stem))
    output_dir = out_parent / f"{safe_id}_{ts}"
    metadata = dict(metadata)
    metadata.update({
        "reactant": reaction_config["reactant"],
        "source_file": input_path.name,
        "bypass_file": bypass_path.name if bypass_path else "",
        "reaction": reaction_config.get("name", ""),
        "output_date": ts[:8],
        "output_prefix": f"{ts}_{safe_id}",
    })
    return gc_processor.run(
        filepath=str(input_path),
        output_dir=str(output_dir),
        reaction_config=reaction_config,
        metadata=metadata,
        inlet_flows=inlet_flows,
        ss_start=ss_start,
        ss_end=ss_end,
        bypass_filepath=str(bypass_path) if bypass_path else None,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run GC analysis from the terminal using the same backend as the GUI.")
    files = p.add_mutually_exclusive_group(required=True)
    files.add_argument("--file", help="One GC .xlsx workbook.")
    files.add_argument("--files", nargs="+", help="One or more GC .xlsx paths or glob patterns.")
    p.add_argument("--config", required=True,
                   help="Reaction config name or YAML path, e.g. rwgs, co2_hydrogenation, fts.")
    p.add_argument("--bypass-file", help="Optional separate bypass .xlsx workbook.")
    p.add_argument("--out", default=str(ROOT / "results" / "gc_cli"),
                   help="Output parent folder. Default: results/gc_cli.")
    p.add_argument("--interactive", action="store_true",
                   help="Prompt for metadata, inlet flows, bypass settings, and output folder.")
    p.add_argument("--inlet", nargs="*", metavar="SPECIES=SCCM",
                   help="Override inlet flows, e.g. --inlet CO2=10 H2=40 Ar=15.")
    p.add_argument("--catalyst-id")
    p.add_argument("--temperature")
    p.add_argument("--pressure")
    p.add_argument("--ghsv")
    p.add_argument("--run-duration-h")
    p.add_argument("--injection-interval-min")
    p.add_argument("--rejected-initial-injections")
    p.add_argument("--registered-reaction-injections")
    p.add_argument("--bypass-omit-initial")
    p.add_argument("--bypass-points-used")
    p.add_argument("--same-file-bypass-mode", choices=["auto", "first", "last", "none"])
    p.add_argument("--same-file-bypass-rows")
    p.add_argument("--c5-unknown-response-factor")
    p.add_argument("--c6-unknown-response-factor")
    p.add_argument("--plot-style", choices=["auto", "time_on_stream", "single_axis_stacked", "stacked_preview"])
    p.add_argument("--notes")
    p.add_argument("--ss-start", type=int, default=1)
    p.add_argument("--ss-end", type=int, default=999)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = _config_path(args.config)
    reaction_config = gc_processor.load_reaction_config(str(config_path))
    input_paths = _expand_files([args.file] if args.file else args.files)
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input workbook not found: {path}")
        if path.suffix.lower() != ".xlsx":
            raise ValueError(f"GC input must be .xlsx: {path}")

    if args.interactive and len(input_paths) != 1:
        raise ValueError("--interactive is for one file at a time. Use flags for batch runs.")

    if args.interactive:
        metadata, inlet_flows, bypass_path, ss_start, ss_end, out_parent = _interactive_metadata(
            args, reaction_config, input_paths[0])
    else:
        metadata = _metadata_from_args(args, input_paths[0])
        inlet_flows = _default_inlet_flows(reaction_config)
        inlet_flows.update(_parse_key_values(args.inlet))
        bypass_path = _repo_path(args.bypass_file) if args.bypass_file else None
        ss_start, ss_end = args.ss_start, args.ss_end
        out_parent = _repo_path(args.out)

    out_parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for path in input_paths:
        run_metadata = dict(metadata)
        if not args.catalyst_id and not args.interactive:
            run_metadata["catalyst_id"] = path.stem
        print(f"Processing {path.name} with {config_path.name}...")
        result = _run_one(path, reaction_config, run_metadata, inlet_flows,
                          ss_start, ss_end, out_parent, bypass_path)
        rows.append({
            "input_file": str(path),
            "catalyst_id": run_metadata.get("catalyst_id"),
            "reaction": reaction_config.get("name"),
            "conversion_%": result.get("conversion"),
            "conversion_std_%": result.get("conversion_std"),
            "carbon_balance_%": result.get("carbon_balance"),
            "n_reaction": result.get("n_reaction"),
            "n_ss": result.get("n_ss"),
            "bypass_source": result.get("bypass_source"),
            "output_dir": result.get("output_dir"),
            "summary_path": result.get("summary_path"),
            "flows_path": result.get("flows_path"),
            "plot_path": result.get("plot_path"),
            "selectivities": json.dumps(result.get("selectivities", {}), sort_keys=True),
        })
        print(f"  output: {result.get('output_dir')}")
        print(f"  conversion: {result.get('conversion')}%, carbon balance: {result.get('carbon_balance')}%")

    manifest = out_parent / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_gc_batch_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Batch manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
