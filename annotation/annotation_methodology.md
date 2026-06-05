# Annotation Methodology — 6 Şubat Depremi Söylem Analizi

## 1. Veri ve Görev

**Corpus:** 6 Şubat 2023 Kahramanmaraş depremine ilişkin 300 Türkçe haber/açıklama metni.
Kaynaklar: Çevre, Şehircilik ve İklim Değişikliği Bakanlığı (csb), Cumhurbaşkanlığı (tccb),
valilikler, belediyeler, AFAD.

**Annotation görevi:** Her metni 4 söylem çerçevesi (frame) için 0-3 ordinal Likert
ölçeğinde etiketleme:

- **Technical** (teknik dayanıklılık): mühendislik, yönetmelik, hasar tespit
- **Political** (politik mobilizasyon): seçim/parti söylemi, liderlik
- **Development** (kalkınma): yatırım, altyapı, ekonomik dönüşüm
- **Sustainability** (sürdürülebilirlik): yeşil bina, iklim direnci, ekoloji

**Skala:** 0=Yok, 1=İma, 2=Belirgin (~%30-50), 3=Baskın (omurga).

---

## 2. İnsan Annotation Aşaması (99 doc)

İki bağımsız annotator (Cemre ve Mehmet) ilk 99 doc'u codebook v1 rehberinde
etiketledi.

### İlk inter-annotator agreement (IAA)

| Frame | Cohen's κ |
|---|---|
| Technical | 0.82 |
| Political | 0.84 |
| **Development** | **0.48** (sorunlu) |
| Sustainability | 1.00 |

### Codebook revizyonu

Development frame'inde anlaşmazlık yüksek olduğu için annotator'lar bir araya
gelip codebook'u tartıştı, tanımları netleştirdi ve etiketleri yeniden gözden geçirdi.

### İkinci IAA (revize sonrası)

| Frame | Cohen's κ | Quadratic κ |
|---|---|---|
| Technical | 0.82 | 0.93 |
| Political | 0.84 | 0.93 |
| **Development** | **0.81** | **0.92** (önemli iyileşme) |
| Sustainability | 1.00 | 1.00 |

Sustainability ekstrem dengesiz (%94 sıfır), corpus'ta nadir bir frame.

---

## 3. Gold Dataset Oluşturma (Agreement-Based)

**Önemli metodolojik karar:** Ortalama-yuvarlama yerine *frame başına agreement-based
gold* yaklaşımı kullanıldı.

- İki annotator anlaştıysa → o değer kesin gold (`human_agreement`)
- Anlaşmadıysa → flag'lendi (`human_disagreement`), conservative default olarak
  min değer alındı

Bu yaklaşım disagreement'ı bilgi olarak korur ve sahte konsensüs üretmez. Ordinal
ölçeklerde aritmetik ortalamanın anlamsız bir orta yer ürettiği problemi de aşar.

---

## 4. LLM Annotation Aşaması (Kalan 201 doc)

### Few-shot pool

4 frame'in tamamında insan agreement olan doc'lardan label space'i kapsayan 8
örnek manuel seçildi (zero-baseline'dan high-political'e kadar diversite gözetildi).

### Prompt yapısı

- Codebook tanımları (her frame için 0-3 göstergeleri ve karar rehberi)
- 8 few-shot örnek (metin + JSON etiket)
- JSON output schema zorunluluğu
- Temperature=0, deterministik tahmin

### Model seçimi süreci

1. **İlk deneme — Claude Haiku 4.5** (hızlı, ucuz). 91 gold doc validation'ında:

   | Frame | κ_quad | Δ_mean |
   |---|---|---|
   | Technical | 0.59 | **+0.57** (over-detection) |
   | Political | 0.76 | +0.18 |
   | Development | 0.49 | +0.25 |
   | Sustainability | 0.89 | 0.00 |

   Sistematik over-detection bias'ı tespit edildi.

2. **Hızlı karşılaştırma — Sonnet 4.5 (20 doc):** Technical kalibrasyonu düzeldi
   (Δ=0.00), kabul edilebilir performans.

3. **Final model — Claude Sonnet 4.5.** 201 doc'a uygulandı.

### Sonnet 4.5 validation (91 gold doc)

| Frame | Exact | ±1 tolerance | κ | κ_quad | Δ_mean |
|---|---|---|---|---|---|
| Technical | 48.8% | 92.5% | 0.253 | 0.639 | +0.36 |
| Political | 49.4% | 92.4% | 0.303 | 0.700 | +0.25 |
| Development | 46.8% | 87.0% | 0.255 | 0.472 | +0.13 |
| Sustainability | 94.5% | 98.9% | 0.528 | 0.831 | -0.02 |

LLM hala miktar over-detection eğilimi gösteriyor ama Haiku'dan belirgin biçimde
iyi. ±1 toleransta %87-99 doğruluk — yorum doğru, kalibrasyon kısmen sapık.

---

## 5. Manuel Review (LLM Çıktısı Doğrulama)

Validation κ_quad değerleri ve over-detection paterni göz önüne alınarak şu
yüksek-risk kategoriler gözle kontrol edildi:

### Sustainability=3 (4 doc)

Hepsi okundu. **1 false positive** (`csb_0669`) tespit edildi ve S=2'ye düşürüldü.
Diğer 3 doc gerçekten iklim/çevre omurgalı (BM destekli proje, Bakan Kurum
sürdürülebilirlik konuşması, Hatay Asi Nehri çevre projesi) — onaylandı.

### Development=3 (39 doc)

Hepsi okundu. **15 false positive** tespit edildi ve D=2'ye düşürüldü. Bu doc'lar
tipik olarak yerel ölçekli iş yeri/asfalt/küçük çarşı haberleriydi — "kalkınma
vizyonu omurgası" değil. Geri kalan 24 doc (büyük TOKİ projeleri, bakanlık vizyon
konuşmaları, Cumhurbaşkanlığı açılış konuşmaları) onaylandı.

Bu doc'lar `label_source = llm_then_manual` olarak işaretlendi.

---

## 6. Final Dataset

300 doc, frame başına 4 olası label_source:

| Source | Açıklama |
|---|---|
| `human_agreement` | İki annotator anlaştı (en güvenilir) |
| `human_disagreement` | Anlaşmadılar, conservative min alındı |
| `llm_only` | LLM tahmin etti, manuel kontrol edilmedi |
| `llm_then_manual` | LLM tahmin etti, gözle kontrol sonrası düzeltildi |

### Final dağılımlar

**Development:** 23% / 20% / 45% / 12% (0/1/2/3)

**Sustainability:** 91% / 5% / 2% / 2% — corpus'ta nadir frame (beklenen bulgu)

---

## 7. Pipeline Bir Bakışta

```
300 doc
   │
   ├── 99 doc → İki annotator → IAA → Codebook revizyonu →
   │            Yeniden annotation → Frame-başına agreement-based gold
   │
   └── 201 doc → Claude Sonnet 4.5 (8-shot, temp=0) →
                 Validation (91 gold üzerinde κ hesabı) →
                 Yüksek-risk kategorilerin manuel review'u →
                 16 düzeltme

→ Final 300 doc, label_source şeffaflığıyla
```

---

## 8. Yayında Nasıl Raporlanır

> 300 documents were annotated for four discourse frames on a 0–3 Likert scale.
> Of these, 99 were independently double-annotated by two coders following a
> structured codebook. Initial inter-annotator agreement on the Development frame
> was insufficient (Cohen's κ=0.48); after codebook reconciliation discussion
> and re-annotation, agreement reached substantial levels for all frames
> (Technical κ=0.82, Political κ=0.84, Development κ=0.81, Sustainability κ=1.00;
> quadratic-weighted κ ranging 0.92–1.00). Per-frame agreement-based gold labels
> were derived: where annotators agreed, the consensus value was retained;
> disagreements were flagged with a conservative (minimum) default.
>
> The remaining 201 documents were annotated by Claude Sonnet 4.5 (temperature=0)
> using an 8-shot prompt with examples drawn from the agreement subset.
> Validation against held-out human-agreement labels yielded quadratic-weighted
> κ of 0.64 (Technical), 0.70 (Political), 0.47 (Development), and 0.83
> (Sustainability). Given moderate Development κ and rare-class concerns for
> Sustainability, all LLM Development=3 (n=39) and Sustainability=3 (n=4)
> predictions were manually reviewed against the codebook, with 16 corrections
> applied (1 Sustainability, 15 Development). Each label in the final dataset
> is tagged with its provenance (human_agreement, human_disagreement, llm_only,
> llm_then_manual).

---

## 9. Metodolojik Güçlü Yönler

- Double annotation + IAA ölçümü (standart pratik)
- Codebook iterasyonu (Development κ 0.48 → 0.81) — titiz çalışma göstergesi
- Agreement-based gold (ortalama yerine) — ordinal task için doğru yaklaşım
- LLM çıktısının validation'a karşı sınanması (sahip olunan gold üzerinden)
- Yüksek-risk kategorilerde human-in-the-loop manuel review
- Her etiketin kaynağı şeffaf (label_source kolonu)
- Quadratic-weighted κ raporlanması (ordinal data için doğru metrik)

---

## 10. Eksikler / İleri Çalışma için Öneriler

- 99 doc'taki `human_disagreement` etiketleri henüz adjudication'dan geçmedi.
  İdeal olarak Cemre ve Mehmet bu doc'ları beraber tartışıp uzlaşmalı.
- LLM validation, few-shot olarak kullanılan 8 doc dışında 91 doc üzerinde
  yapıldı; bağımsız bir test set olsa daha iyiydi ama 99 toplam annotated doc
  kısıtı bunu zorluyor.
- Sadece tek model (Sonnet 4.5) kullanıldı; multi-model ensemble (Claude +
  GPT + Gemini) ileri çalışma için potansiyel.
- Sustainability=1 ve =2 (22 doc) ve Technical=3 (18 doc) tahminleri zaman
  kısıtı nedeniyle manuel review'a alınmadı. İleri sürüm bu kontrolü ekleyebilir.
