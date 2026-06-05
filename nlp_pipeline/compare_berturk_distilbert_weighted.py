"""Helin W5: isolate model-size from loss-function.

The paper compares BERTurk-base (underperforming) against DistilBERTurk (leading),
but the two were trained under different objectives, so the comparison conflates
model size with the loss function. This script runs BOTH models under the SAME
weighted-MSE loss, the SAME 5-fold CV, the SAME seed and hyperparameters, so the
only thing that differs is the encoder. Either outcome is publishable:
  - DistilBERT still wins under matched loss -> the distillation/size effect is real.
  - BERTurk-base catches up -> the original gap was a loss-function artefact.

This reuses the project's weighted-MSE trainer. It is designed to be run on a GPU
(Colab T4 is fine). It logs to experiments.csv via the existing logger.

COLAB USAGE (after mounting Drive and adding nlp_pipeline to sys.path):
    !python compare_berturk_distilbert_weighted.py

Or run the two underlying CV calls directly with the existing module:
    from pipeline_berturk_cv_weighted import main as cv_weighted
    cv_weighted("berturk_base_weighted",  epochs=10, lr=1e-5,
                hf_model="dbmdz/bert-base-turkish-cased")
    cv_weighted("distilberturk_weighted", epochs=10, lr=1e-5,
                hf_model="dbmdz/distilbert-base-turkish-cased")

Place this file in nlp_pipeline/ (it imports the pipeline modules).
"""
from __future__ import annotations

import sys

# pipeline_berturk_cv_weighted.main(experiment_name, epochs, lr, n_folds, hf_model, notes)
from pipeline_berturk_cv_weighted import main as cv_weighted

MODELS = [
    ("berturk_base_weighted",  "dbmdz/bert-base-turkish-cased",
     "BERTurk-base under MATCHED weighted-MSE loss (W5 control)"),
    ("distilberturk_weighted", "dbmdz/distilbert-base-turkish-cased",
     "DistilBERTurk under the same weighted-MSE loss (W5 reference)"),
]

EPOCHS = 10
LR = 1e-5
N_FOLDS = 5


def main():
    print("=" * 70)
    print("W5 experiment: model size vs loss function")
    print("Both models share: weighted-MSE loss, 5-fold KFold(seed=42),")
    print(f"epochs={EPOCHS}, lr={LR}. Only the encoder differs.")
    print("=" * 70)
    for name, hf_model, notes in MODELS:
        print(f"\n\n########## {name}  ({hf_model}) ##########")
        cv_weighted(
            experiment_name=name,
            epochs=EPOCHS,
            lr=LR,
            n_folds=N_FOLDS,
            hf_model=hf_model,
            notes=notes,
        )
    print("\n\nDone. Compare the macro QWK / per-frame QWK rows for the two")
    print("experiments in outputs/experiments.csv. The interpretation:")
    print("  - if DistilBERT still leads under matched loss -> size/distillation effect")
    print("  - if BERTurk-base catches up -> original gap was a loss-function artefact")


if __name__ == "__main__":
    main()
