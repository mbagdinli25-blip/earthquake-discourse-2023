"""Robustness check 1: human-only vs full-label training.

Reviewer concern (all three): the LLM annotations carry a residual upward bias on
Technical (+0.36) and Political (+0.25), and this bias propagates into the deployed
model. This script retrains the TF-IDF pipeline using ONLY the human_agreement
labels for each frame, scores the full corpus, and correlates those predictions
with the predictions from the full (human + LLM) training set already in
analytic_with_scores.csv. High correlation => the deployment findings do not hinge
on the LLM-labelled portion.

Per-frame, because label_source is recorded per frame (technical_source, etc.).

Run (from analysis/):
    python robustness_human_only.py \
        --train ../annotation/final_300.csv \
        --corpus ../corpus/corpus_final.csv \
        --scored ../outputs/analytic_with_scores.csv \
        --out ../outputs/robustness_human_only.csv
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVR

FRAMES = ["technical", "political", "development", "sustainability"]
RANDOM_STATE = 42

# Compact Turkish stopword list (same as the pipeline).
TURKISH_STOPWORDS = [
    "acaba","ama","ancak","artık","asla","aslında","az","bana","bazen","bazı","belki",
    "ben","benden","beni","benim","beri","beş","bile","bir","biraz","birçok","biri",
    "birkaç","birşey","biz","bizden","bizi","bizim","böyle","böylece","bu","buna","bunda",
    "bundan","bunlar","bunları","bunların","bunu","bunun","burada","çok","çünkü","da","daha",
    "dahi","de","defa","değil","diğer","diye","doksan","dokuz","dolayı","dolayısıyla","dört",
    "elli","en","fakat","falan","filan","gene","gibi","hala","hangi","hatta","hem","henüz",
    "hep","hepsi","her","herhangi","herkes","hiç","hiçbir","için","iki","ile","ilgili","ise",
    "işte","itibaren","itibariyle","kadar","karşın","kendi","kendilerine","kendini","kendisi",
    "kendisine","kendisini","kez","ki","kim","kimden","kime","kimi","kimse","madem","mı","mi",
    "mu","mü","nasıl","ne","neden","nedenle","nerde","nerede","nereye","niçin","niye","o",
    "olan","olarak","oldu","olduğu","olduğunu","olduklarını","olmadı","olmadığı","olmak",
    "olması","olmayan","olmaz","olsa","olsun","olup","olur","olursa","oluyor","on","ona",
    "ondan","onlar","onlardan","onları","onların","onu","onun","otuz","oysa","öyle","pek",
    "rağmen","sana","sanki","sekiz","seksen","sen","senden","seni","senin","siz","sizden",
    "sizi","sizin","sonra","şayet","şey","şeyden","şeyi","şeyler","şimdi","şu","şuna","şunda",
    "şundan","şunları","şunu","tüm","üç","üzere","var","vardı","ve","veya","ya","yani","yedi",
    "yerine","yetmiş","yine","yirmi","yoksa","yüz","zaten",
]


def make_vectorizers():
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=20000,
                               min_df=3, stop_words=TURKISH_STOPWORDS, lowercase=True,
                               sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=20000,
                               min_df=3, lowercase=True, sublinear_tf=True)
    return word_vec, char_vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../annotation/final_300.csv")
    ap.add_argument("--corpus", default="../corpus/corpus_final.csv")
    ap.add_argument("--scored", default="../outputs/analytic_with_scores.csv")
    ap.add_argument("--out", default="../outputs/robustness_human_only.csv")
    args = ap.parse_args()

    train = pd.read_csv(args.train, encoding="utf-8-sig")
    corpus = pd.read_csv(args.corpus, encoding="utf-8-sig")
    scored = pd.read_csv(args.scored, encoding="utf-8-sig")
    corpus["text"] = corpus["text"].fillna("").astype(str)

    rows = []
    for frame in FRAMES:
        src_col = f"{frame}_source"
        # human_agreement rows only
        mask = train[src_col] == "human_agreement"
        sub = train[mask & train["text"].notna() & train[frame].notna()].copy()
        n_human = len(sub)
        if n_human < 20:
            print(f"[{frame}] only {n_human} human-agreement docs — skipping (too few)")
            continue

        wv, cv = make_vectorizers()
        Xtr = hstack([wv.fit_transform(sub["text"]), cv.fit_transform(sub["text"])]).tocsr()
        y = sub[frame].values.astype(float)
        m = LinearSVR(random_state=RANDOM_STATE, max_iter=10000)
        m.fit(Xtr, y)

        Xall = hstack([wv.transform(corpus["text"]), cv.transform(corpus["text"])]).tocsr()
        pred_human_only = np.clip(m.predict(Xall), 0, 3)

        # align to the full-training predictions already in the scored corpus
        full = scored[["doc_id", f"tfidf_score_{frame}"]].copy()
        comp = corpus[["doc_id"]].copy()
        comp["human_only"] = pred_human_only
        comp = comp.merge(full, on="doc_id", how="inner")
        r, _ = pearsonr(comp["human_only"], comp[f"tfidf_score_{frame}"])
        rho, _ = spearmanr(comp["human_only"], comp[f"tfidf_score_{frame}"])
        mean_shift = comp["human_only"].mean() - comp[f"tfidf_score_{frame}"].mean()
        rows.append({
            "frame": frame, "n_human_train": n_human,
            "pearson_r": round(r, 4), "spearman_rho": round(rho, 4),
            "mean_shift_humanonly_minus_full": round(mean_shift, 4),
        })
        print(f"[{frame}] n_human={n_human}  r={r:.3f}  rho={rho:.3f}  shift={mean_shift:+.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
