"""Politicization Index with within-month bootstrap confidence intervals.

Reproducible from the scored corpus alone. Computes, for each month, the ratio of
mean political to mean technical frame score, with a 95% CI from resampling
documents within the month.

Usage:
    python politicization_bootstrap.py --scored analytic_with_scores.csv --out ci.csv

Index definition:
    index(m) = mean(political_m) / (mean(technical_m) + EPS)
where EPS=0.05 floors the denominator so a near-zero-technical month cannot explode.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

EPS = 0.05
N_BOOT = 2000
SEED = 42


def bootstrap_index(df: pd.DataFrame, pol_col: str, tech_col: str, model: str,
                    rng: np.random.Generator) -> pd.DataFrame:
    recs = []
    for month, g in df.groupby("month"):
        pol = g[pol_col].to_numpy(float)
        tech = g[tech_col].to_numpy(float)
        n = len(pol)
        if n < 5:
            continue
        point = pol.mean() / (tech.mean() + EPS)
        boots = np.empty(N_BOOT)
        for b in range(N_BOOT):
            s = rng.integers(0, n, n)
            boots[b] = pol[s].mean() / (tech[s].mean() + EPS)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        recs.append({
            "model": model, "month": month, "n_docs": n,
            "politicization_index": round(point, 4),
            "ci_lower": round(lo, 4), "ci_upper": round(hi, 4),
            "ci_width": round(hi - lo, 4),
        })
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="analytic_with_scores.csv")
    ap.add_argument("--out", default="politicization_bootstrap_ci.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.scored, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    rng = np.random.default_rng(SEED)
    db = bootstrap_index(df, "distilbert_score_political_0to3",
                         "distilbert_score_technical_0to3", "distilbert", rng)
    tf = bootstrap_index(df, "tfidf_score_political",
                         "tfidf_score_technical", "tfidf", rng)
    out = pd.concat([db, tf]).sort_values(["model", "month"])
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows -> {args.out}")
    print(out[out.model == "distilbert"].to_string(index=False))


if __name__ == "__main__":
    main()
