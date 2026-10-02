#!/usr/bin/env python
"""Quality-control review of XRD refinement output.

Reads the summary.json / *_results.xlsx pair written by xrd_batch.py (or by
the GUI) and reports findings across four families:

  stats     counting-statistics validity, convergence, correlation
  cell      lattice parameters against the reference CIF
  size      crystallite size and broadening plausibility
  residual  structure in the difference curve

Findings carry a severity so an operator (or an agent driving a refine/review
loop) can tell "this number is wrong" from "this number is unverifiable".

    python scripts/xrd_review.py results/xrd_batch
    python scripts/xrd_review.py results/xrd_batch --json review.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.xrd.counting_statistics import estimate_counting_time  # noqa: E402

# Thresholds. Deliberately conservative: these flag "look at this", not "wrong".
VOLUME_DEVIATION_PCT = 1.0        # |dV/V0| above this is worth explaining
AXIS_ANISOTROPY_PCT = 0.5         # spread between per-axis deviations
SIZE_MIN_NM = 2.0
SIZE_MAX_NM = 200.0               # beyond lab-XRD resolving power
OUTLIER_SIGMA = 3.0
OUTLIER_FRACTION = 0.05           # >5% of points beyond 3 sigma is structural
DECILE_BIAS_SIGMA = 1.0           # coherent regional bias
PEAK_WINDOW_DEG = 0.35            # "on peak" half-width for residual split
INTENSITY_IMBALANCE_FRAC = 0.02   # residual extremum as fraction of max peak
FRACTION_GAP_PCT = 10.0           # wt% vs diffracted-area% disagreement
TICK_MATCH_DEG = 0.30             # residual counts as "on" a reflection within this

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


class Finding:
    def __init__(self, check: str, severity: str, message: str,
                 evidence: Optional[Dict[str, Any]] = None,
                 suggestion: str = ""):
        self.check = check
        self.severity = severity
        self.message = message
        self.evidence = evidence or {}
        self.suggestion = suggestion

    def as_dict(self) -> Dict[str, Any]:
        return {"check": self.check, "severity": self.severity,
                "message": self.message, "evidence": self.evidence,
                "suggestion": self.suggestion}


def _f(value: Any) -> Optional[float]:
    try:
        out = float(value)
        return None if math.isnan(out) else out
    except (TypeError, ValueError):
        return None


def check_statistics(summary: Dict[str, Any],
                     pattern: Optional[Dict[str, np.ndarray]]) -> List[Finding]:
    out: List[Finding] = []
    stats = summary.get("statistics", {})
    rwp, gof = _f(stats.get("Rwp")), _f(stats.get("GoF"))

    poisson_valid = True
    if pattern is not None:
        estimate = estimate_counting_time(pattern["tt"], pattern["y_obs"])
        if not estimate.get("is_counts"):
            poisson_valid = False
            scale = estimate.get("gof_scale")
            if scale is not None and gof is not None:
                out.append(Finding(
                    "stats", "warning",
                    f"Intensities are cps, not counts. Counting noise implies "
                    f"{estimate['seconds_per_step']:.2f} s/step, so the "
                    f"reported GoF {gof:.2f} corresponds to about "
                    f"{gof * scale:.2f}, and chi-squared {gof**2:.1f} to "
                    f"about {(gof * scale) ** 2:.1f}.",
                    {"GoF_reported": gof,
                     "GoF_corrected": round(gof * scale, 2),
                     "chi2_reported": round(gof ** 2, 2),
                     "chi2_corrected": round((gof * scale) ** 2, 2),
                     "seconds_per_step": round(estimate["seconds_per_step"], 3),
                     "n_windows": estimate["n_windows"],
                     "spread_pct": round(estimate["spread_pct"], 1)},
                    "Use the corrected value. Rwp and Rp are unaffected."))
            else:
                out.append(Finding(
                    "stats", "warning",
                    "Intensities are not whole counts, so sigma = sqrt(I) "
                    "does not apply, and the counting time could not be "
                    "recovered from the background.",
                    {"GoF_reported": gof},
                    "Judge this fit on Rwp and residual shape."))

    if rwp is not None and gof is not None and poisson_valid and gof > 3 and rwp < 8:
        out.append(Finding(
            "stats", "warning",
            f"Rwp {rwp:.2f}% is good but GoF {gof:.2f} is high on data whose "
            "counting statistics appear valid, so the misfit is real.",
            {"Rwp": rwp, "GoF": gof},
            "Inspect the residual findings below for the responsible term."))

    diagnostics = summary.get("refinement_diagnostics", {}) or {}
    if diagnostics.get("converged") is False:
        out.append(Finding(
            "stats", "critical",
            "GSAS-II did not converge before the cycle limit.",
            {"final_stage": (diagnostics.get("final_stage") or {}).get("stage")},
            "Treat every refined value as provisional; reduce the active "
            "parameter set and rerun."))

    for corr in (diagnostics.get("high_correlations") or [])[:3]:
        out.append(Finding(
            "stats", "warning",
            f"Strong correlation {corr.get('parameter_1')} vs "
            f"{corr.get('parameter_2')} = {corr.get('correlation_pct')}%.",
            dict(corr),
            "Correlated parameters trade off against each other; fix one and "
            "rerun to see which is actually determined."))

    for warning in summary.get("fit_warnings", []) or []:
        out.append(Finding("stats", "info", str(warning), {}, ""))
    return out


def check_cell(phase: Dict[str, Any]) -> List[Finding]:
    out: List[Finding] = []
    name = phase.get("phase") or phase.get("formula") or "phase"
    dv = _f(phase.get("delta_volume_pct"))
    axes = {k: _f(phase.get(f"delta_{k}_pct")) for k in ("a", "b", "c")}
    present = {k: v for k, v in axes.items() if v is not None}

    if dv is not None and abs(dv) >= VOLUME_DEVIATION_PCT:
        # Never critical: dV/V0 is measured against whichever reference CIF
        # was supplied, so a large value can simply mean the starting
        # composition differs from the sample. The refined lattice parameter
        # is the quantity that compares across references, not this.
        out.append(Finding(
            "cell", "warning",
            f"{name}: unit-cell volume differs from its reference CIF by "
            f"{dv:+.2f}%.",
            {"delta_volume_pct": dv, "reference": phase.get("cell_reference")},
            "Check the refined lattice parameter against literature rather "
            "than this percentage, which depends on the reference chosen. A "
            "large value often just means a different starting composition; "
            "Zero and sample displacement also push the cell."))

    if len(present) >= 2:
        spread = max(present.values()) - min(present.values())
        if spread >= AXIS_ANISOTROPY_PCT:
            worst = max(present, key=lambda k: abs(present[k]))
            out.append(Finding(
                "cell", "warning",
                f"{name}: anisotropic cell change, {spread:.2f} percentage "
                f"points between axes (largest {worst} {present[worst]:+.2f}%).",
                {f"delta_{k}_pct": v for k, v in present.items()},
                "Check whether the axis ratio change is physically expected "
                "for this phase; if not, suspect an unmodelled peak-position "
                "error rather than genuine strain."))
    return out


def check_size(phase: Dict[str, Any]) -> List[Finding]:
    out: List[Finding] = []
    name = phase.get("phase") or phase.get("formula") or "phase"
    size = _f(phase.get("crystallite_size_nm"))
    strain = _f(phase.get("microstrain_microstrain"))

    if size is not None and not (SIZE_MIN_NM <= size <= SIZE_MAX_NM):
        out.append(Finding(
            "size", "warning",
            f"{name}: crystallite size {size:.1f} nm is outside the range lab "
            f"XRD can meaningfully resolve ({SIZE_MIN_NM:g}-{SIZE_MAX_NM:g} nm).",
            {"crystallite_size_nm": size},
            "Above the upper bound the peak width is instrument-limited, so "
            "the value is a lower bound, not a measurement."))

    if size is not None and strain is not None:
        out.append(Finding(
            "size", "warning",
            f"{name}: size and microstrain were both refined; they correlate "
            "strongly and are rarely separable from one pattern.",
            {"crystallite_size_nm": size, "microstrain": strain},
            "Refine one at a time and keep whichever actually lowers Rwp."))
    return out


def check_fit_sanity(summary: Dict[str, Any]) -> List[Finding]:
    """Catch fits that failed rather than merely fitting badly."""
    out: List[Finding] = []
    stats = summary.get("statistics", {})
    rwp, gof = _f(stats.get("Rwp")), _f(stats.get("GoF"))
    # Rwp can stay plausible while a refinement diverges: the scale runs away
    # and the calculated pattern ends up orders of magnitude above the data.
    # Rp saturating at 100% is the reliable tell.
    rp = _f(stats.get("Rp"))
    if rp is not None and rp >= 99.0:
        out.append(Finding(
            "sanity", "critical",
            f"Rp is {rp:.1f}%, so the calculated pattern bears no relation to "
            f"the data even though Rwp reads {rwp if rwp is None else f'{rwp:.2f}'}%.",
            {"Rp": rp, "Rwp": rwp},
            "The refinement diverged. Discard it; do not report any fraction "
            "from this fit."))

    for label, value in (("Rwp", rwp), ("GoF", gof)):
        if value is not None and (not math.isfinite(value) or value > 1e3):
            out.append(Finding(
                "sanity", "critical",
                f"{label} is {value:.3g}: the refinement diverged rather than "
                "converging to a poor fit.",
                {label: value},
                "Discard this result. Reduce the active parameters and check "
                "that the instrument geometry matches the instrument file."))

    for phase in summary.get("phase_results", []) or []:
        name = phase.get("phase") or phase.get("formula") or "phase"
        wt = _f(phase.get("weight_fraction_pct"))
        sig = _f(phase.get("scale_sigma_rel_pct"))
        if sig is not None and sig > 100:
            out.append(Finding(
                "sanity", "critical",
                f"{name}: scale uncertainty is {sig:.3g}%, so its amount is "
                "not determined by the data at all.",
                {"weight_fraction_pct": wt, "scale_sigma_rel_pct": sig},
                "The phase has collapsed. Remove it, or fix its scale, and "
                "do not report a fraction for it."))
        elif wt is not None and wt <= 0.5 and sig is not None and sig > 20:
            out.append(Finding(
                "sanity", "warning",
                f"{name}: refined to {wt:.1f} wt% with a {sig:.0f}% scale "
                "uncertainty, so the data does not support including it.",
                {"weight_fraction_pct": wt, "scale_sigma_rel_pct": sig},
                "Drop this phase unless independent evidence requires it."))
    return out


def check_fraction_consistency(summary: Dict[str, Any]) -> List[Finding]:
    """Mass fraction and diffracted-intensity share are computed by different
    routes, so disagreement between them is evidence, not rounding."""
    out: List[Finding] = []
    for phase in summary.get("phase_results", []) or []:
        wt = _f(phase.get("weight_fraction_pct"))
        area = _f(phase.get("diffraction_area_fraction_pct"))
        if wt is None or area is None:
            continue
        gap = wt - area
        if abs(gap) >= FRACTION_GAP_PCT:
            name = phase.get("phase") or phase.get("formula") or "phase"
            out.append(Finding(
                "fraction", "warning",
                f"{name}: {wt:.1f} wt% but only {area:.1f}% of the diffracted "
                f"intensity ({gap:+.1f} points apart).",
                {"weight_fraction_pct": wt,
                 "diffraction_area_fraction_pct": area,
                 "gap_points": round(gap, 1)},
                "A phase booking far more mass than diffraction is usually "
                "modelling diffuse scattering. Treat the mass fraction as "
                "unreliable, and consider Debye background terms instead."))
    return out


def classify_unmodelled_peaks(pattern: Dict[str, np.ndarray]) -> List[Finding]:
    """Distinguish a missing phase from wrong intensity on a modelled one.

    Residual peaks sitting on existing reflections mean the intensities of
    phases already in the model are wrong (occupancy, texture, Uiso).
    Residual peaks with no reflection nearby mean something is absent.
    """
    out: List[Finding] = []
    tt, res = pattern["tt"], pattern["y_obs"] - pattern["y_calc"]
    ticks = pattern.get("ticks")
    scale = float(np.abs(res).max())
    if scale <= 0 or ticks is None or not len(ticks):
        return out

    window = max(1, int(round(0.15 / max(np.median(np.diff(tt)), 1e-6))))
    smooth = np.convolve(res, np.ones(window) / window, mode="same")
    threshold = 0.3 * float(smooth.max())
    if threshold <= 0:
        return out

    on_peak, off_peak, i = [], [], 0
    while i < len(tt):
        if smooth[i] > threshold:
            j = i
            while j + 1 < len(tt) and smooth[j + 1] > threshold:
                j += 1
            k = i + int(np.argmax(smooth[i:j + 1]))
            pos = float(tt[k])
            nearest = float(min(ticks, key=lambda t: abs(t - pos)))
            (on_peak if abs(nearest - pos) <= TICK_MATCH_DEG
             else off_peak).append((pos, round(nearest - pos, 2)))
            i = j + 1
        else:
            i += 1

    if off_peak:
        out.append(Finding(
            "residual", "warning",
            f"{len(off_peak)} residual peak(s) have no reflection within "
            f"{TICK_MATCH_DEG}deg: "
            + ", ".join(f"{p:.2f}deg" for p, _ in off_peak[:5]) + ".",
            {"unindexed_two_theta": [p for p, _ in off_peak]},
            "Intensity where no modelled phase has a reflection points to a "
            "missing phase. Identify it before adding free parameters."))
    if on_peak and not off_peak:
        out.append(Finding(
            "residual", "warning",
            f"{len(on_peak)} residual peak(s) sit on reflections that are "
            "already modelled: "
            + ", ".join(f"{p:.2f}deg" for p, _ in on_peak[:5]) + ".",
            {"misfit_two_theta": [p for p, _ in on_peak]},
            "The phases present have the wrong relative intensities rather "
            "than something being absent. Preferred orientation, site "
            "occupancy or Uiso are the candidates, in that order."))
    return out


def check_residuals(pattern: Dict[str, np.ndarray]) -> List[Finding]:
    out: List[Finding] = []
    tt, yo, yc = pattern["tt"], pattern["y_obs"], pattern["y_calc"]
    bg, ticks = pattern.get("background"), pattern.get("ticks")
    res = yo - yc
    sigma = np.sqrt(np.maximum(yo, 1.0))
    norm = res / sigma

    frac = float(np.mean(np.abs(norm) > OUTLIER_SIGMA))
    if frac > OUTLIER_FRACTION:
        out.append(Finding(
            "residual", "warning",
            f"{100*frac:.1f}% of points lie beyond {OUTLIER_SIGMA:g} sigma "
            "(noise alone would give well under 1%), so the misfit is "
            "systematic rather than statistical.",
            {"fraction_beyond_3sigma": round(frac, 4)},
            "A constant sigma error cannot produce this; look for a model term."))

    edges = np.percentile(tt, np.arange(0, 101, 10))
    biased = []
    for i in range(10):
        mask = (tt >= edges[i]) & (tt <= edges[i + 1])
        if mask.sum() < 10:
            continue
        bias = float(np.mean(norm[mask]))
        if abs(bias) >= DECILE_BIAS_SIGMA:
            biased.append({"two_theta_range": [round(edges[i], 1),
                                               round(edges[i + 1], 1)],
                           "mean_res_over_sigma": round(bias, 2)})
    if biased:
        out.append(Finding(
            "residual", "warning",
            f"Residuals are coherently biased in {len(biased)} angular "
            "region(s) rather than scattering about zero.",
            {"regions": biased},
            "Regional sign runs point at background shape or peak position, "
            "not at random noise."))

    if bg is not None and ticks is not None and len(ticks):
        on_peak = np.zeros_like(tt, dtype=bool)
        for t in ticks:
            on_peak |= np.abs(tt - t) <= PEAK_WINDOW_DEG
        w = 1.0 / sigma**2
        chi = w * res**2
        total = float(chi.sum())
        if total > 0 and on_peak.any() and (~on_peak).any():
            share = float(chi[on_peak].sum() / total)
            # Peak windows cover only part of the range, so compare the
            # chi-squared share against the share of points they contain.
            coverage = float(on_peak.mean())
            enrichment = share / coverage if coverage > 0 else float("inf")
            if enrichment >= 1.5:
                where, blame = ("concentrated at peaks",
                                "profile shape, peak position or relative "
                                "intensity")
            elif enrichment <= 0.7:
                where, blame = ("concentrated between peaks",
                                "the background model")
            else:
                where, blame = ("spread evenly across the pattern",
                                "a broad scale or weighting issue rather than "
                                "one localised term")
            out.append(Finding(
                "residual", "info",
                f"Misfit is {where}: peak windows hold {100*coverage:.0f}% of "
                f"the points but {100*share:.0f}% of chi-squared "
                f"({enrichment:.1f}x enrichment).",
                {"chi2_share_on_peak": round(share, 3),
                 "point_share_on_peak": round(coverage, 3),
                 "enrichment": round(enrichment, 2)},
                f"That implicates {blame}."))

        scale = float((yo - bg).max()) if bg is not None else float(yo.max())
        if scale > 0:
            pos = neg = None
            for t in ticks:
                m = np.abs(tt - t) <= PEAK_WINDOW_DEG
                if not m.any():
                    continue
                r = res[m]
                hi, lo = float(r.max()), float(r.min())
                if hi / scale >= INTENSITY_IMBALANCE_FRAC and (pos is None or hi > pos[1]):
                    pos = (float(t), hi)
                if -lo / scale >= INTENSITY_IMBALANCE_FRAC and (neg is None or lo < neg[1]):
                    neg = (float(t), lo)
            if pos and neg:
                out.append(Finding(
                    "residual", "warning",
                    "Reflections err in opposite directions: "
                    f"{pos[0]:.2f}deg is under-calculated by {pos[1]:+.0f} while "
                    f"{neg[0]:.2f}deg is over-calculated by {neg[1]:+.0f}. That is "
                    "an hkl-dependent intensity error, not a peak-width error.",
                    {"under_calculated_two_theta": round(pos[0], 2),
                     "over_calculated_two_theta": round(neg[0], 2)},
                    "Preferred orientation is the usual cause. Enable PO "
                    "(refined) and compare; if PO does not help, check site "
                    "occupancies in the CIF."))
    return out


def load_pattern(xlsx_path: Path) -> Optional[Dict[str, np.ndarray]]:
    try:
        import openpyxl
    except ImportError:
        return None
    if not xlsx_path.exists():
        return None
    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    if "Plot Data" not in wb.sheetnames:
        return None
    rows = list(wb["Plot Data"].iter_rows(values_only=True))
    if len(rows) < 3:
        return None
    header = [str(h) if h is not None else "" for h in rows[0]]

    def column(predicate) -> Optional[int]:
        for i, name in enumerate(header):
            if predicate(name.lower()):
                return i
        return None

    i_tt = column(lambda h: "2theta" in h or "2th" in h)
    i_yo = column(lambda h: "y_obs" in h)
    i_yc = column(lambda h: "y_calc" in h)
    i_bg = column(lambda h: "background" in h)
    # One tick column per phase; all of them count as modelled
    i_tks = [i for i, name in enumerate(header)
             if "peak 2" in name.lower()]
    if None in (i_tt, i_yo, i_yc):
        return None

    def grab(idx: Optional[int]) -> Optional[np.ndarray]:
        if idx is None:
            return None
        vals = [_f(r[idx]) if idx < len(r) else None for r in rows[1:]]
        return np.array([v for v in vals if v is not None], dtype=float)

    data = {"tt": grab(i_tt), "y_obs": grab(i_yo), "y_calc": grab(i_yc)}
    if any(v is None or not len(v) for v in data.values()):
        return None
    n = min(len(data["tt"]), len(data["y_obs"]), len(data["y_calc"]))
    data = {k: v[:n] for k, v in data.items()}
    bg = grab(i_bg)
    data["background"] = bg[:n] if bg is not None and len(bg) >= n else None
    all_ticks = []
    for idx in i_tks:
        col = grab(idx)
        if col is not None and len(col):
            all_ticks.append(col)
    data["ticks"] = (np.unique(np.concatenate(all_ticks))
                     if all_ticks else None)
    return data


def review_sample(summary_path: Path) -> Dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sample = summary.get("sample") or summary_path.parent.name

    xlsx = summary.get("summary_path")
    xlsx_path = Path(xlsx) if xlsx else None
    if xlsx_path is None or not xlsx_path.exists():
        found = sorted(summary_path.parent.glob("*_results.xlsx"))
        xlsx_path = found[0] if found else None
    pattern = load_pattern(xlsx_path) if xlsx_path else None

    findings = check_statistics(summary, pattern)
    findings.extend(check_fit_sanity(summary))
    findings.extend(check_fraction_consistency(summary))
    for phase in summary.get("phase_results", []) or []:
        findings.extend(check_cell(phase))
        findings.extend(check_size(phase))
    if pattern is not None:
        findings.extend(check_residuals(pattern))
        findings.extend(classify_unmodelled_peaks(pattern))
    else:
        findings.append(Finding(
            "residual", "info",
            "Pattern data unavailable, so residual structure was not checked.",
            {"looked_for": str(xlsx_path) if xlsx_path else None},
            "Keep the *_results.xlsx beside summary.json to enable this."))

    findings.sort(key=lambda f: SEVERITY_ORDER.get(f.severity, 9))
    stats = summary.get("statistics", {})
    return {"sample": sample,
            "Rwp": _f(stats.get("Rwp")),
            "GoF": _f(stats.get("GoF")),
            "findings": [f.as_dict() for f in findings]}


def find_summaries(target: Path) -> List[Path]:
    if target.is_file():
        return [target]
    direct = target / "summary.json"
    if direct.exists():
        return [direct]
    return sorted(target.glob("*/summary.json"))


def render(reviews: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for review in reviews:
        rwp = f"{review['Rwp']:.2f}%" if review["Rwp"] is not None else "n/a"
        gof = f"{review['GoF']:.2f}" if review["GoF"] is not None else "n/a"
        lines.append("")
        lines.append(f"=== {review['sample']}   Rwp {rwp}   GoF {gof} ===")
        if not review["findings"]:
            lines.append("  no findings")
        for f in review["findings"]:
            lines.append(f"  [{f['severity'].upper():8s}] ({f['check']}) {f['message']}")
            if f["suggestion"]:
                lines.append(f"             -> {f['suggestion']}")
    counts: Dict[str, int] = {}
    for review in reviews:
        for f in review["findings"]:
            counts[f["severity"]] = counts.get(f["severity"], 0) + 1
    lines.append("")
    lines.append(f"{len(reviews)} sample(s); " + ", ".join(
        f"{n} {sev}" for sev, n in sorted(counts.items(),
                                          key=lambda kv: SEVERITY_ORDER.get(kv[0], 9))
    ) if counts else f"{len(reviews)} sample(s); no findings")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", help="Batch output directory, sample directory, or summary.json")
    p.add_argument("--json", help="Also write the findings to this JSON file")
    args = p.parse_args()

    target = Path(args.target).resolve()
    if not target.exists():
        print(f"Not found: {target}", file=sys.stderr)
        return 2
    summaries = find_summaries(target)
    if not summaries:
        print(f"No summary.json under {target}", file=sys.stderr)
        return 2

    reviews = [review_sample(s) for s in summaries]
    print(render(reviews))
    if args.json:
        Path(args.json).write_text(
            json.dumps({"reviews": reviews}, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")

    worst = min((SEVERITY_ORDER.get(f["severity"], 9)
                 for r in reviews for f in r["findings"]), default=9)
    return 1 if worst == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
