"""DiD strengthening: parallel-trends visualisation + placebo event test.

Reviewer concern (Hi-Mehmet, Helin): the difference-in-differences estimates assume
parallel pre-treatment trends, which is asserted but never shown, and there is no
placebo check. This script:

  1. Builds monthly mean frame scores for the treatment group (local-government /
     municipal sources, `source_type == 'belediye'`) vs control (all other
     sources), and plots them around the 2024 local election (treatment date
     2024-03-31) so the reader can eyeball pre-trend parallelism.
  2. Runs a PLACEBO DiD at a fake treatment date in the stable inter-election
     window (default 2023-10-01), where no real treatment occurred. A near-zero,
     non-significant placebo coefficient supports the design; a large one warns
     that the main DiD may be picking up a pre-existing divergence.

Treatment-group definition matches the paper's local-government DiD. Adjust
--treat-date / --placebo-date / --treat-types as needed.

Run (from analysis/):
    python did_parallel_trends_placebo.py \
        --scored ../outputs/analytic_with_scores.csv \
        --outdir ../outputs
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

FRAMES = ["technical", "political", "development", "sustainability"]
SCORE = "distilbert_score_{}_0to3"


def load(scored):
    df = pd.read_csv(scored, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["source_type"] = df["source_type"].str.strip().str.lower()
    return df


def did_at(df, treat_date, treat_types, frame, window_months=6):
    """Two-period DiD around treat_date within +/- window_months."""
    d = df.copy()
    lo = pd.Timestamp(treat_date) - pd.DateOffset(months=window_months)
    hi = pd.Timestamp(treat_date) + pd.DateOffset(months=window_months)
    d = d[(d["date"] >= lo) & (d["date"] < hi)]
    d["treat"] = d["source_type"].isin(treat_types).astype(int)
    d["post"] = (d["date"] >= pd.Timestamp(treat_date)).astype(int)
    d["y"] = d[SCORE.format(frame)]
    if d["treat"].nunique() < 2 or d["post"].nunique() < 2:
        return None
    m = smf.ols("y ~ treat * post", data=d).fit()
    coef = m.params.get("treat:post", np.nan)
    se = m.bse.get("treat:post", np.nan)
    p = m.pvalues.get("treat:post", np.nan)
    return {"frame": frame, "DiD_coef": round(coef, 4), "se": round(se, 4),
            "p": round(p, 4), "n": int(len(d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="../outputs/analytic_with_scores.csv")
    ap.add_argument("--outdir", default="../outputs")
    ap.add_argument("--treat-date", default="2024-03-31")
    ap.add_argument("--placebo-date", default="2023-10-01")
    ap.add_argument("--treat-types", default="belediye")
    args = ap.parse_args()

    treat_types = [t.strip().lower() for t in args.treat_types.split(",")]
    df = load(args.scored)

    # ---- 1. Parallel-trends plot (monthly means, treat vs control) ----
    df["grp"] = np.where(df["source_type"].isin(treat_types), "Local gov (treat)", "Other (control)")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, frame in zip(axes.ravel(), FRAMES):
        g = (df.groupby(["month", "grp"])[SCORE.format(frame)]
               .mean().reset_index())
        for grp, sub in g.groupby("grp"):
            ax.plot(sub["month"], sub[SCORE.format(frame)], marker="o", ms=3, label=grp)
        ax.axvline(pd.Timestamp(args.treat_date), color="red", ls="--", lw=1)
        ax.axvline(pd.Timestamp(args.placebo_date), color="gray", ls=":", lw=1)
        ax.set_title(frame.capitalize())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Monthly mean frame score: treatment vs control "
                 "(red = local election, dotted = placebo date)")
    fig.tight_layout()
    fig_path = f"{args.outdir}/figures/V_did_parallel_trends.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved parallel-trends figure -> {fig_path}")

    # ---- 2. Real vs placebo DiD ----
    real_rows, plac_rows = [], []
    for frame in FRAMES:
        r = did_at(df, args.treat_date, treat_types, frame)
        p = did_at(df, args.placebo_date, treat_types, frame)
        if r: real_rows.append({**r, "event": f"REAL local election {args.treat_date}"})
        if p: plac_rows.append({**p, "event": f"PLACEBO {args.placebo_date}"})

    out = pd.DataFrame(real_rows + plac_rows)
    out_path = f"{args.outdir}/did_placebo_results.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved DiD + placebo table -> {out_path}\n")
    print(out.to_string(index=False))
    print("\nInterpretation: REAL coefficients should be sizeable/significant where "
          "the paper reports effects; PLACEBO coefficients should be small and "
          "non-significant. Large placebo effects would undercut the design.")


if __name__ == "__main__":
    main()
