"""
landmark_quality_report.py — Mevcut Landmark Kalite Raporu
============================================================

Mevcut landmark JSON dosyalarini analiz edip kalite skoru verir.
Sorunlu kelimeleri tespit eder.

Kullanim:
    python landmark_quality_report.py <landmarks_klasoru>
    python landmark_quality_report.py dataset/landmarks/
"""

import os, sys, json
import numpy as np


def analyze_landmark(filepath):
    """Tek bir landmark dosyasinin kalite analizini yap."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    n = len(frames)
    if n == 0:
        return None

    word = data.get("word_tr", os.path.basename(filepath)[:-5])

    # Null frame sayilari
    null_lh = sum(1 for f in frames if not f.get("left_hand"))
    null_rh = sum(1 for f in frames if not f.get("right_hand"))
    null_both = sum(1 for f in frames if not f.get("left_hand") and not f.get("right_hand"))

    # En uzun null gap
    max_lh_gap = _max_gap(frames, "left_hand")
    max_rh_gap = _max_gap(frames, "right_hand")
    max_gap = max(max_lh_gap, max_rh_gap)

    # Z jitter
    z_diffs = []
    for i in range(1, n):
        for key in ["11", "12", "15", "16"]:
            p = frames[i].get("pose", {}).get(key)
            pp = frames[i-1].get("pose", {}).get(key)
            if p and pp:
                z_diffs.append(abs(p[2] - pp[2]))
    avg_z = np.mean(z_diffs) if z_diffs else 0
    max_z = np.max(z_diffs) if z_diffs else 0

    # Hareket miktari
    xy_diffs = []
    for i in range(1, n):
        for key in ["15", "16"]:
            p = frames[i].get("pose", {}).get(key)
            pp = frames[i-1].get("pose", {}).get(key)
            if p and pp:
                xy_diffs.append(abs(p[0]-pp[0]) + abs(p[1]-pp[1]))
    avg_motion = np.mean(xy_diffs) if xy_diffs else 0

    # Skor hesapla
    score = 100.0
    null_ratio = (null_lh + null_rh) / (2 * n)
    score -= null_ratio * 60
    score -= min(max_gap / n, 1.0) * 25
    score -= min(avg_z / 0.3, 1.0) * 10
    if avg_motion < 0.005:
        score -= 5
    score = max(0, min(100, score))

    # Sorun tespiti
    problems = []
    if null_lh > n * 0.3:
        problems.append(f"sol el %{null_lh/n*100:.0f} kayip")
    if null_rh > n * 0.3:
        problems.append(f"sag el %{null_rh/n*100:.0f} kayip")
    if max_gap > n * 0.3:
        problems.append(f"max bosluk {max_gap} frame")
    if max_z > 0.5:
        problems.append(f"Z spike {max_z:.2f}")
    if avg_z > 0.1:
        problems.append(f"Z jitter yuksek {avg_z:.3f}")

    return {
        "word": word,
        "score": round(score, 1),
        "frames": n,
        "null_lh": null_lh,
        "null_rh": null_rh,
        "null_both": null_both,
        "max_gap": max_gap,
        "avg_z": round(avg_z, 4),
        "max_z": round(max_z, 4),
        "avg_motion": round(avg_motion, 4),
        "problems": problems,
        "smoothed": data.get("smoothed", False),
    }


def _max_gap(frames, key):
    max_g = 0
    cur = 0
    for f in frames:
        if not f.get(key):
            cur += 1
            max_g = max(max_g, cur)
        else:
            cur = 0
    return max_g


def main():
    if len(sys.argv) < 2:
        print("Kullanım: python landmark_quality_report.py <landmarks_klasoru>")
        sys.exit(1)

    lm_dir = sys.argv[1]
    files = sorted(f for f in os.listdir(lm_dir) if f.endswith(".json"))

    results = []
    for fname in files:
        try:
            r = analyze_landmark(os.path.join(lm_dir, fname))
            if r:
                results.append(r)
        except Exception as e:
            print(f"HATA: {fname} — {e}")

    # Skora gore sirala
    results.sort(key=lambda x: x["score"])

    print("=" * 75)
    print("TSL Landmark Kalite Raporu")
    print("=" * 75)
    print(f"Toplam: {len(results)} kelime")

    # Smooth edilmis veri uyarisi
    smoothed_count = sum(1 for r in results if r.get("smoothed"))
    if smoothed_count > 0:
        print(f"\n⚠ UYARI: {smoothed_count}/{len(results)} dosya smooth edilmis!")
        print("  Smooth edilmis veride null frame'ler doldurulmus, gercek kalite daha dusuk olabilir.")
        print("  Gercek kaliteyi gormek icin: landmarks_backup/ klasorunu analiz edin.")
        print("  veya: python landmark_quality_report.py landmarks_backup/")
    print()

    scores = [r["score"] for r in results]
    print(f"Skor dagilimi:")
    print(f"  Ortalama:  {np.mean(scores):.1f}")
    print(f"  Medyan:    {np.median(scores):.1f}")
    print(f"  Min:       {np.min(scores):.1f}")
    print(f"  Max:       {np.max(scores):.1f}")
    print()

    # Kategorilere ayir
    excellent = [r for r in results if r["score"] >= 80]
    good = [r for r in results if 60 <= r["score"] < 80]
    poor = [r for r in results if 40 <= r["score"] < 60]
    bad = [r for r in results if r["score"] < 40]

    print(f"Kategoriler:")
    print(f"  Mukemmel (80-100): {len(excellent)} kelime")
    print(f"  Iyi (60-79):       {len(good)} kelime")
    print(f"  Zayif (40-59):     {len(poor)} kelime")
    print(f"  Kotu (0-39):       {len(bad)} kelime")
    print()

    # Sorunlu kelimeleri detayli goster
    problematic = [r for r in results if r["score"] < 70]
    if problematic:
        print(f"{'─' * 75}")
        print(f"SORUNLU KELIMELER (skor < 70) — Yeniden video secimi onerilen:")
        print(f"{'─' * 75}")
        for r in problematic:
            prob_str = ", ".join(r["problems"]) if r["problems"] else "genel dusuk kalite"
            print(f"  {r['word']:25s} skor:{r['score']:5.1f}  null_L:{r['null_lh']:2d} null_R:{r['null_rh']:2d}  gap:{r['max_gap']:2d}  | {prob_str}")
        print()
        print(f"Bu {len(problematic)} kelime icin best_landmark_picker.py calistirin:")
        print(f"  python best_landmark_picker.py dataset/videos/ {lm_dir}")
    else:
        print("Tum kelimeler iyi durumda!")

    print()

    # Iyi kelimeleri de ozet goster
    if excellent:
        print(f"{'─' * 75}")
        print(f"EN IYI KELIMELER (skor >= 80):")
        print(f"{'─' * 75}")
        for r in sorted(excellent, key=lambda x: -x["score"])[:15]:
            print(f"  {r['word']:25s} skor:{r['score']:5.1f}")


if __name__ == "__main__":
    main()
