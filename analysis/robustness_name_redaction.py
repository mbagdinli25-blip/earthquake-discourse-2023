"""Robustness check 2: name-redaction for the Political frame.

Reviewer concern (Hi-Mehmet, Berke): in Section VI-G, 20 of the top 25 Political
tokens are proper nouns (party names AKP/CHP, leader names, institutions). This
leaves it ambiguous whether the model captures political *content* or just does
institutional/named-entity recognition. This script redacts a list of party,
leader, and institution names from the text, retrains the Political model on the
redacted text, rescores the corpus, and checks whether (a) predictions still
correlate with the original and (b) the monthly period pattern survives.

If the period pattern (esp. the local-campaign rise) holds after redaction, the
argument that the model captures genuine political mobilisation discourse is much
stronger.

Run (from analysis/):
    python robustness_name_redaction.py \
        --train ../annotation/final_300.csv \
        --corpus ../corpus/corpus_final.csv \
        --scored ../outputs/analytic_with_scores.csv \
        --out ../outputs/robustness_name_redaction.csv
"""
from __future__ import annotations

import argparse
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVR

RANDOM_STATE = 42

# Names / institutions to redact. Case-insensitive, whole-word where sensible.
# Extend this list if inspection of top tokens reveals more.
REDACT_TERMS = [
    # parties
    "akp", "ak parti", "ak parti", "chp", "mhp", "iyi parti", "iyi̇ parti", "deva",
    "cumhur ittifak", "millet ittifak",
    # leaders / figures
    "erdoğan", "erdogan", "recep tayyip", "kılıçdaroğlu", "kilicdaroglu",
    "imamoğlu", "imamoglu", "yavaş", "yavas", "özel", "bahçeli", "bahceli",
    "soyer", "böcek", "bocek", "savaş", "savas",
    # institutions frequently named
    "cumhurbaşkanı", "cumhurbaskani", "cumhurbaşkanlığı", "akpli", "chpli",
    "belediye başkanı", "büyükşehir belediye başkanı",
]

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


def redact(text: str, patterns) -> str:
    t = text
    for pat in patterns:
        t = pat.sub(" ", t)
    return t


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
    ap.add_argument("--out", default="../outputs/robustness_name_redaction.csv")
    ap.add_argument("--frame", default="political")
    args = ap.parse_args()

    frame = args.frame
    patterns = [re.compile(re.escape(t), flags=re.IGNORECASE) for t in REDACT_TERMS]

    train = pd.read_csv(args.train, encoding="utf-8-sig")
    corpus = pd.read_csv(args.corpus, encoding="utf-8-sig")
    scored = pd.read_csv(args.scored, encoding="utf-8-sig")
    for df in (train, corpus):
        df["text"] = df["text"].fillna("").astype(str)

    # redact both training and deployment text
    train["text_red"] = train["text"].map(lambda s: redact(s, patterns))
    corpus["text_red"] = corpus["text"].map(lambda s: redact(s, patterns))

    sub = train[train[frame].notna()].copy()
    wv, cv = make_vectorizers()
    Xtr = hstack([wv.fit_transform(sub["text_red"]), cv.fit_transform(sub["text_red"])]).tocsr()
    y = sub[frame].values.astype(float)
    m = LinearSVR(random_state=RANDOM_STATE, max_iter=10000)
    m.fit(Xtr, y)

    Xall = hstack([wv.transform(corpus["text_red"]), cv.transform(corpus["text_red"])]).tocsr()
    corpus[f"{frame}_redacted"] = np.clip(m.predict(Xall), 0, 3)

    # 1) correlation with the original (non-redacted) prediction
    comp = corpus[["doc_id", "date", f"{frame}_redacted"]].merge(
        scored[["doc_id", f"tfidf_score_{frame}", "distilbert_score_%s_0to3" % frame]],
        on="doc_id", how="inner")
    r_tf, _ = pearsonr(comp[f"{frame}_redacted"], comp[f"tfidf_score_{frame}"])
    r_db, _ = pearsonr(comp[f"{frame}_redacted"], comp["distilbert_score_%s_0to3" % frame])

    # 2) does the monthly period pattern survive?
    comp["date"] = pd.to_datetime(comp["date"], errors="coerce")
    comp = comp.dropna(subset=["date"])
    comp["month"] = comp["date"].dt.to_period("M").dt.to_timestamp()
    monthly = comp.groupby("month").agg(
        redacted_mean=(f"{frame}_redacted", "mean"),
        original_mean=(f"tfidf_score_{frame}", "mean"),
        n=("doc_id", "count"),
    ).reset_index()
    month_corr = monthly["redacted_mean"].corr(monthly["original_mean"])

    print(f"=== Name-redaction robustness ({frame}) ===")
    print(f"  redacted vs original TF-IDF:     r = {r_tf:.3f}")
    print(f"  redacted vs original DistilBERT: r = {r_db:.3f}")
    print(f"  monthly-mean pattern preserved:  r = {month_corr:.3f}")
    print(f"\n  April 2024 redacted mean: "
          f"{monthly.loc[monthly.month=='2024-04-01','redacted_mean'].values}")

    monthly.to_csv(args.out, index=False)
    # also write a one-row summary alongside
    pd.DataFrame([{
        "frame": frame,
        "r_vs_tfidf": round(r_tf, 4),
        "r_vs_distilbert": round(r_db, 4),
        "monthly_pattern_corr": round(month_corr, 4),
        "n_redact_terms": len(REDACT_TERMS),
    }]).to_csv(args.out.replace(".csv", "_summary.csv"), index=False)
    print(f"\nWrote {args.out} (+ _summary.csv)")


if __name__ == "__main__":
    main()
