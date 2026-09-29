# Batch refine-and-review

Drop-in workspace for refining several patterns against a fixed set of CIFs
and having the results checked automatically.

## Layout

```
batch_input/
├── patterns/    your scans (.txt .xy .xye .dat .csv .xlsx) - up to ~5 per run
└── cifs/        the candidate phases for this batch (.cif)
```

Both folders are git-ignored, so the data stays local.

## Run

```bash
conda activate catalysis
export PYTHONPATH="$HOME/g2full/GSAS-II/GSASII:$PYTHONPATH"   # source GSAS-II only

python scripts/xrd_batch.py \
    --patterns "batch_input/patterns/*" \
    --cif-dir batch_input/cifs \
    --instprm benchtop_Cu_Si640g.instprm \
    --out results/xrd_batch

python scripts/xrd_review.py results/xrd_batch --json results/xrd_batch/review.json
```

The first command refines every pattern and writes a `summary.json` plus a
`*_results.xlsx` per sample. The second reads those and reports findings.

## What the review checks

| Family | Looks for |
|---|---|
| `stats` | counting-statistics validity, convergence, parameter correlation |
| `cell` | lattice parameters and volume against the reference CIF |
| `size` | crystallite size plausibility, size/mustrain separability |
| `residual` | structure in the difference curve |

Severities are `critical` (the numbers are wrong), `warning` (explain this
before trusting it) and `info` (context). The script exits non-zero if
anything is `critical`, so it can gate a script.

### The residual checks are the diagnostic ones

They separate failure modes that Rwp alone cannot:

- **Coherent regional bias** - sign runs across 2theta mean a model term is
  wrong, not that the data is noisy.
- **Peak vs background enrichment** - compares the chi-squared share inside
  peak windows against the share of points those windows hold. Enrichment
  above ~1.5x implicates profile, position or intensity; below ~0.7x
  implicates the background.
- **Opposite-signed reflections** - one reflection under-calculated while
  another is over-calculated is an hkl-dependent intensity error, which
  usually means preferred orientation rather than a peak-width problem.

## A note on counting statistics

Rietveld weights assume `sigma = sqrt(I)`, which holds only for raw counts.
Data exported as **cps** scales chi-squared by an unknown factor, so **GoF
becomes uninterpretable while Rwp stays valid**. The review flags this
rather than silently reporting a bad-looking GoF. Re-export as raw counts if
you need GoF to mean something.
