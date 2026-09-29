"""Recover Poisson weighting for patterns exported as counts per second.

Rietveld weights assume sigma = sqrt(I), which holds only for whole counts.
Data exported as cps is I = N/t for a Poisson N, so Var(I) = I/t and the true
sigma is sqrt(I/t). chi-squared is then wrong by a factor t and GoF by sqrt(t).

t is rarely recorded in an exported pattern, but it is measurable from the
data: in a region with no peak, point-to-point scatter is pure counting noise,
so t = mean(I) / Var(I). Successive differences estimate that variance without
being fooled by a sloping background.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

MIN_WINDOW_POINTS = 40
N_WINDOWS = 40
# Windows containing a peak have their variance inflated by signal, which
# biases t low. Rank windows by median intensity and keep the quietest
# fraction rather than demanding strict flatness, so patterns with broad
# peaks still yield usable background.
QUIET_FRACTION = 0.35
MIN_WINDOWS = 3


def _windows(tt: np.ndarray, y: np.ndarray, n: int = N_WINDOWS) -> List[slice]:
    edges = np.linspace(0, len(tt), n + 1, dtype=int)
    return [slice(edges[i], edges[i + 1]) for i in range(n)
            if edges[i + 1] - edges[i] >= MIN_WINDOW_POINTS]


def estimate_counting_time(tt: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    """Estimate seconds per step from counting noise in peak-free regions.

    Returns a dict with `seconds_per_step` (None when undeterminable),
    `gof_scale` (multiply a Poisson-weighted GoF by this), `n_windows` and
    `spread_pct`. Whole-count data yields is_counts=True and no correction.
    """
    tt = np.asarray(tt, dtype=float)
    y = np.asarray(y, dtype=float)
    result: Dict[str, object] = {
        "is_counts": bool(np.nanmax(np.abs(y - np.round(y))) <= 1e-6),
        "seconds_per_step": None,
        "gof_scale": None,
        "n_windows": 0,
        "spread_pct": None,
    }
    if result["is_counts"] or len(y) < MIN_WINDOW_POINTS * MIN_WINDOWS:
        return result

    scored = []
    for window in _windows(tt, y):
        chunk = y[window]
        variance = float(np.mean(np.diff(chunk) ** 2) / 2.0)
        mean = float(chunk.mean())
        if variance > 0 and mean > 0:
            scored.append((float(np.median(chunk)), mean / variance))
    if not scored:
        return result

    scored.sort(key=lambda pair: pair[0])
    keep = max(MIN_WINDOWS, int(round(QUIET_FRACTION * len(scored))))
    estimates = [t for _, t in scored[:keep]]

    if len(estimates) < MIN_WINDOWS:
        return result

    seconds = float(np.median(estimates))
    if not (1e-4 < seconds < 1e4):
        return result

    spread = (float(np.percentile(estimates, 75) - np.percentile(estimates, 25))
              / seconds * 100.0)
    result.update({
        "seconds_per_step": seconds,
        "gof_scale": float(np.sqrt(seconds)),
        "n_windows": len(estimates),
        "spread_pct": spread,
    })
    return result


def describe(estimate: Dict[str, object], gof: Optional[float],
             chi2: Optional[float] = None) -> Optional[str]:
    """One-line human summary, or None when no correction applies."""
    if estimate.get("is_counts") or estimate.get("gof_scale") is None:
        return None
    scale = float(estimate["gof_scale"])
    seconds = float(estimate["seconds_per_step"])
    text = (f"Intensities are not whole counts, so sigma = sqrt(I) does not "
            f"apply. Counting noise implies about {seconds:.2f} s/step "
            f"({estimate['n_windows']} background windows, "
            f"{estimate['spread_pct']:.0f}% spread).")
    parts = []
    if gof is not None:
        parts.append(f"GoF {gof:.2f} corresponds to roughly {gof*scale:.2f}")
    if chi2 is not None:
        # chi-squared carries the full factor; GoF is its square root.
        parts.append(f"chi-squared {chi2:.2f} to roughly "
                     f"{chi2*seconds:.2f}")
    if parts:
        text += " Reported " + ", and ".join(parts) + "."
    return text
