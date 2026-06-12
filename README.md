# Measuring the Discursive Landscape of Turkey's 2023 Earthquake

Computational text-analysis pipeline for the paper *Measuring the Discursive
Landscape of Turkey's 2023 Earthquake: A Computational Text Analysis of Official
Discourse Across Two Electoral Cycles* (CSSM 530).

The project scrapes official Turkish post-earthquake communications, builds a
2,139-document analytic corpus, annotates a 300-document training subset on four
discourse frames, trains two parallel supervised models, scores the full corpus,
and runs the downstream period/province/event analyses.



---

## Discourse frames (0 = absent, 1 = implicit, 2 = salient, 3 = dominant)

| Frame | Gloss |
|---|---|
| `technical` | Engineering, building codes, inspection, damage assessment |
| `political` | Election/party rhetoric, leadership, national mobilisation |
| `development` | Investment, infrastructure, economic transformation |
| `sustainability` | Green building, climate resilience (rare; ~94 % zeros) |

---

## Repository layout

```
.

├── corpus/                   # corpus construction
│   ├── filtrele.py                  # filtering pipeline
│   ├── corpus_creation_final.ipynb  # build + dedup + keyword + geo + period
│   ├── master_corpus_final.csv      # merged raw harvest
│   └── corpus_final.csv             # 2,139-doc analytic corpus
├── annotation/               # labels + provenance
│   ├── codebook.md, annotation_methodology.md
│   ├── annotation_set_cemre.csv, annotation_set_mehmet.csv  # human double-coding
│   ├── disagreements_*.csv          # reconciliation records
│   ├── llm_annotation.ipynb         # LLM-assisted annotation (Haiku trial -> Sonnet final)
│   ├── llm_annotation_output/       # gold_99, few-shot set, llm_on_201, etc.
│   └── final_300.csv                # gold training set w/ label_source columns
├── nlp_pipeline/             # training + scoring (the .py files)
│   ├── config.py                    # paths/constants (edit DATA_ROOT here)
│   ├── pipeline_tfidf_svr.py        # TF-IDF baseline (word 1-2 grams)
│   ├── pipeline_tfidf_v2.py         # TF-IDF v2 (word + char n-grams, weighting)
│   ├── pipeline_berturk.py          # BERTurk-base fine-tune
│   ├── pipeline_berturk_cv.py       # BERTurk K-fold CV
│   ├── pipeline_berturk_cv_weighted.py  # weighted-MSE CV variant
│   ├── train_final_models.py        # fit final TF-IDF + DistilBERTurk on all 300
│   ├── predict_corpus.py            # score corpus (single model)
│   ├── predict_corpus_both.py       # score corpus (TF-IDF + DistilBERTurk)
│   ├── experiments_logger.py        # appends results to experiments.csv
│   └── colab_quickstart.ipynb       # one-click Colab run
├── analysis/                 # downstream statistics
│   └── politicization_bootstrap.py  # monthly index + 95% bootstrap CI
├── outputs/                  # results behind the figures/tables
│   ├── analytic_with_scores.csv     # corpus + per-doc TF-IDF & DistilBERT scores
│   ├── 0X_*.csv                     # mixed models, DiD, clusters, distinctive words
│   ├── politicization_bootstrap_ci.csv
│   ├── loglen_results.txt           # length-confound control output
│   └── figures/                     # all result figures (PNG)
├── appendix/
│   └── appendix.tex                 # keyword list, codebook, LLM prompt
├── paper/
│   ├── discoure_revised.tex, .pdf   # the manuscript
│   └── references.bib
├── requirements.txt
└── README.md
```

> **Not included in git** (too large; distributed via the Zenodo release): the
> trained model artifacts — `outputs/final_distilbert/` (~272 MB) and
> `outputs/final_tfidf/tfidf_bundle.joblib`. See the Zenodo DOI below.

---

## Pipeline order (end to end)


1. **Build corpus** — `corpus/corpus_creation_final.ipynb` + `corpus/filtrele.py`:
   merge, exact + near-duplicate removal (MinHash/LSH, Jaccard 0.85), 26-term
   keyword pass, geographic + electoral-period assignment → `corpus_final.csv`
   (2,139 docs; 20.9 % retention from the 10,218-row raw harvest).
2. **Annotate** — double human coding of 99 docs, codebook reconciliation on
   Development, LLM annotation of the remaining 201 (Sonnet 4.5, temperature 0,
   8-shot) with manual review → `annotation/final_300.csv`. Every label carries a
   `label_source` (`human_agreement`, `human_disagreement`, `llm_only`).
3. **Train + score** — `nlp_pipeline/train_final_models.py` fits the final TF-IDF
   and DistilBERTurk models on all 300 docs; `predict_corpus_both.py` scores the
   2,139-doc corpus → `outputs/analytic_with_scores.csv`.
4. **Analyse** — `analysis/` + the result CSVs in `outputs/`: period/province
   comparisons, mixed-effects models, DiD, k-means clustering, distinctive
   vocabulary, and the Politicization Index with bootstrap CIs.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. TF-IDF baseline, 5-fold CV (~1 min, CPU)
cd nlp_pipeline && python pipeline_tfidf_svr.py

# 2. Transformer CV (GPU strongly recommended) — see colab_quickstart.ipynb
python pipeline_berturk_cv.py

# 3. Fit final models, then score the corpus
python train_final_models.py
python predict_corpus_both.py

# 4. Politicization Index + bootstrap CIs (from the scored corpus)
cd ../analysis
python politicization_bootstrap.py --scored ../outputs/analytic_with_scores.csv \
       --out ../outputs/politicization_bootstrap_ci.csv
```

`nlp_pipeline/config.py` auto-detects Google Colab and switches to Drive paths; on a
local machine edit `DATA_ROOT`. All scripts fix `random_state = 42`.

---

## Reproducibility notes

- **Cross-validation:** 5-fold `KFold(shuffle=True, random_state=42)` — not
  province-stratified; province balance checked descriptively after splitting.
- **TF-IDF features:** word 1–2 grams **and** char\_wb 3–5 grams stacked,
  `min_df=3`, `max_features=20000` per analyzer, Turkish stop-words. Sample
  weighting applied only when a frame's class-imbalance ratio exceeds 5×.
- **Transformer:** `dbmdz/distilbert-base-turkish-cased`, 512-token head
  truncation, labels normalised 0–3 → 0–1, weighted-MSE loss. BERTurk-base is
  trained for comparison.
- **LLM annotation:** Anthropic API, temperature 0, 8-shot. Two models appear by
  design — an initial Claude Haiku 4.5 trial for model selection and Claude
  Sonnet 4.5, which produced all 201 deployment-subset labels.
- **Politicization Index:** monthly ratio of mean Political to mean Technical
  DistilBERTurk score (0–3); 95 % CIs from within-month document resampling
  (2,000 replicates, seed 42). April 2024 peak = 2.63 (CI [2.00, 3.54]).

The TF-IDF baseline is fully deterministic; transformer runs are mostly
deterministic on identical hardware, with small GPU-to-GPU variation.

---

## Library versions and URLs

All tools used, with versions and homepages (SP7 requirement). Pinned in `requirements.txt`.

| Library | Version | URL |
|---|---|---|
| Python | 3.10–3.12 | https://www.python.org |
| pandas | 2.2.2 | https://pandas.pydata.org |
| numpy | 1.26.4 | https://numpy.org |
| scipy | 1.13.1 | https://scipy.org |
| scikit-learn | 1.5.1 | https://scikit-learn.org |
| torch | 2.3.1 | https://pytorch.org |
| transformers | 4.43.3 | https://huggingface.co/docs/transformers |
| datasets | 2.20.0 | https://huggingface.co/docs/datasets |
| accelerate | 0.32.1 | https://huggingface.co/docs/accelerate |
| statsmodels | 0.14.2 | https://www.statsmodels.org |
| matplotlib | 3.9.1 | https://matplotlib.org |
| anthropic | 0.34.2 | https://github.com/anthropics/anthropic-sdk-python |
| numbers-parser | 4.18.5 | https://github.com/masaccio/numbers-parser |

Pretrained models: BERTurk (`dbmdz/bert-base-turkish-cased`) and DistilBERTurk
(`dbmdz/distilbert-base-turkish-cased`), https://huggingface.co/dbmdz . LLM
annotation via the Anthropic API (Claude Haiku 4.5 for model selection; Claude
Sonnet 4.5 for the 201 deployment labels), https://www.anthropic.com .

## Analysis scripts (reviewer revisions)

| Script | Purpose |
|---|---|
| `analysis/politicization_bootstrap.py` | Monthly Politicization Index + 95% bootstrap CI |
| `analysis/robustness_human_only.py` | Retrain on human-only labels; check LLM-label dependence |
| `analysis/robustness_name_redaction.py` | Redact party/leader names; test Political ≠ name-spotting |
| `analysis/did_parallel_trends_placebo.py` | DiD parallel-trends figure + placebo event test |
| `analysis/robustness_analyses_colab.ipynb` | Colab notebook running all of the above + the BERTurk W5 experiment |

## Data / model archive

Trained models and the full output tree are archived on Zenodo: **DOI _[ADD AFTER
DEPOSIT]_**.

## License

_[choose one — e.g. MIT for code, CC-BY-4.0 for data — add a LICENSE file]_
