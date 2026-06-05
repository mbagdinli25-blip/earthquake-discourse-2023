import pandas as pd

BASE = "/Users/mehmetbagdinli/Desktop/deprem"

df = pd.read_csv(f"{BASE}/master_corpus.csv", encoding="utf-8-sig",
                 engine="python", on_bad_lines="skip")
print(f"Ham kayıt: {len(df)}")

# ── 1. Exact duplicate temizle ────────────────────────────────────────────────
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
print(f"Duplicate sonrası: {len(df)}")

# ── 2. Çok kısa metinleri at ──────────────────────────────────────────────────
df["kelime"] = df["text"].astype(str).str.split().str.len()
df = df[df["kelime"] >= 50].reset_index(drop=True)
print(f"Kısa metin sonrası: {len(df)}")

# ── 3. Tarih filtresi: Şubat 2023 – Aralık 2024 ──────────────────────────────
df["date"] = pd.to_datetime(df["date"], errors="coerce")
onceki = len(df)
df = df[
    (df["date"] >= "2023-02-01") &
    (df["date"] <= "2024-12-31")
].reset_index(drop=True)
print(f"Tarih filtresi sonrası: {len(df)}  ({onceki - len(df)} metin aralık dışı)")

# ── 4. Deprem anahtar kelime filtresi ─────────────────────────────────────────
ANAHTAR = [
    "deprem", "enkaz", "afet", "konut", "yıkım", "yıkılan", "hasar",
    "kurtarma", "tahliye", "konteyner", "prefabrik", "kalıcı konut",
    "altyapı", "yeniden yapım", "yeniden yapılanma", "dönüşüm",
    "inşaat", "TOKİ", "AFAD", "arama kurtarma", "depremzede",
    "6 şubat", "kahramanmaraş depremi", "hatay depremi",
    "acil durum", "barınak", "çadır kent"
]
pattern = "|".join(ANAHTAR)
df["depremle_ilgili"] = df["text"].str.contains(pattern, case=False, na=False)
print(f"Anahtar kelime eşleşen : {df['depremle_ilgili'].sum()}")
print(f"Eşleşmeyen (atılacak)  : {(~df['depremle_ilgili']).sum()}")
df = df[df["depremle_ilgili"]].copy()

# ── 5. Kaynak başına 500 üst sınır (tarih sırasıyla) ─────────────────────────
df = df.sort_values("date")
df = (
    df.groupby("source", group_keys=False)
    .apply(lambda x: x.head(500))
    .reset_index(drop=True)
)

# ── Özet ─────────────────────────────────────────────────────────────────────
print(f"\n── Filtrelenmiş corpus ──────────────────────────────────")
print(f"Toplam kayıt : {len(df)}")
print(f"\nKaynak dağılımı:")
print(df.groupby("source")["text"].count().rename("n").to_string())
print(f"\nTarih aralığı: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Tarihsiz     : {df['date'].isna().sum()}")

# ── 6. Kaydet ─────────────────────────────────────────────────────────────────
df[["source", "date", "text"]].to_csv(
    f"{BASE}/corpus_filtre.csv", index=False, encoding="utf-8-sig"
)
print(f"\n✅ Kaydedildi → corpus_filtre.csv")
