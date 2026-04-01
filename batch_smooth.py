"""
batch_smooth.py — Tüm Landmark JSON Dosyalarını Toplu Smooth Et
================================================================

Kullanım:
    python batch_smooth.py <landmarks_klasoru>
    python batch_smooth.py dataset/landmarks/

Ne yapar:
    1. Orijinal dosyaları 'landmarks_backup/' klasörüne yedekler
    2. Her JSON'a smoothing pipeline uygular
    3. Smooth edilmiş versiyonu orijinal dosyanın üzerine yazar
    4. Detaylı rapor ve istatistik verir

Notlar:
    - Zaten smooth edilmiş dosyalar atlanır ("smoothed": true)
    - --force flag'i ile tekrar smooth edilebilir
    - Yedek her zaman alınır (güvenlik)
"""

import os, sys, json, shutil, time
from landmark_smoother import smooth_landmark_data

def main():
    # ── Argümanları oku ──
    if len(sys.argv) < 2:
        print("Kullanım: python batch_smooth.py <landmarks_klasoru> [--force]")
        print("Örnek:    python batch_smooth.py dataset/landmarks/")
        print("          python batch_smooth.py dataset/landmarks/ --force")
        sys.exit(1)

    landmarks_dir = sys.argv[1]
    force = "--force" in sys.argv

    if not os.path.isdir(landmarks_dir):
        print(f"HATA: '{landmarks_dir}' klasörü bulunamadı!")
        sys.exit(1)

    # ── JSON dosyalarını bul ──
    json_files = sorted([
        f for f in os.listdir(landmarks_dir)
        if f.endswith(".json") and not f.startswith(".")
    ])

    if not json_files:
        print(f"HATA: '{landmarks_dir}' içinde JSON dosyası bulunamadı!")
        sys.exit(1)

    print("=" * 60)
    print(f"TSL Landmark Batch Smoother")
    print(f"=" * 60)
    print(f"Klasör:      {os.path.abspath(landmarks_dir)}")
    print(f"Dosya sayısı: {len(json_files)}")
    print(f"Force modu:   {'EVET' if force else 'Hayır'}")
    print()

    # ── Yedek al ──
    backup_dir = os.path.join(os.path.dirname(landmarks_dir.rstrip("/")), "landmarks_backup")
    if not os.path.exists(backup_dir):
        print(f"Yedek alınıyor → {backup_dir}")
        shutil.copytree(landmarks_dir, backup_dir)
        print(f"Yedek tamamlandı: {len(json_files)} dosya")
    else:
        print(f"Yedek zaten mevcut: {backup_dir}")
    print()

    # ── İşle ──
    stats = {
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "total_null_fixed": 0,
        "total_jitter_reduction": 0.0,
    }

    results = []

    t0 = time.time()

    for fname in json_files:
        fpath = os.path.join(landmarks_dir, fname)
        word = fname[:-5]  # .json uzantısını kaldır

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            # frames yoksa atla
            if "frames" not in raw_data:
                print(f"  ⚠ {fname:30s} — frames anahtarı yok, atlanıyor")
                stats["skipped"] += 1
                continue

            # Zaten smooth edilmiş mi?
            if raw_data.get("smoothed") and not force:
                print(f"  ✓ {fname:30s} — zaten smooth, atlanıyor")
                stats["skipped"] += 1
                continue

            raw_frames = raw_data["frames"]

            # Smoothing öncesi istatistik
            pre_null_lh = sum(1 for fr in raw_frames if not fr.get("left_hand"))
            pre_null_rh = sum(1 for fr in raw_frames if not fr.get("right_hand"))
            pre_z_jitter = _avg_z_jitter(raw_frames)

            # Smooth et
            smoothed_data = smooth_landmark_data(raw_data)
            smooth_frames = smoothed_data["frames"]

            # Smoothing sonrası istatistik
            post_null_lh = sum(1 for fr in smooth_frames if not fr.get("left_hand"))
            post_null_rh = sum(1 for fr in smooth_frames if not fr.get("right_hand"))
            post_z_jitter = _avg_z_jitter(smooth_frames)

            null_fixed = (pre_null_lh - post_null_lh) + (pre_null_rh - post_null_rh)
            jitter_pct = (1 - post_z_jitter / pre_z_jitter) * 100 if pre_z_jitter > 0 else 0

            # Dosyaya yaz
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(smoothed_data, f, ensure_ascii=False)

            status_parts = []
            if null_fixed > 0:
                status_parts.append(f"null:{pre_null_lh+pre_null_rh}→{post_null_lh+post_null_rh}")
            if jitter_pct > 5:
                status_parts.append(f"jitter:-%{jitter_pct:.0f}")
            status_str = " | ".join(status_parts) if status_parts else "minimal değişiklik"

            print(f"  ✓ {fname:30s} — {status_str}")

            stats["processed"] += 1
            stats["total_null_fixed"] += null_fixed
            stats["total_jitter_reduction"] += jitter_pct

            results.append({
                "word": word,
                "null_fixed": null_fixed,
                "jitter_reduction_pct": round(jitter_pct, 1),
            })

        except Exception as e:
            print(f"  ✗ {fname:30s} — HATA: {e}")
            stats["errors"] += 1

    elapsed = time.time() - t0

    # ── Rapor ──
    print()
    print("=" * 60)
    print("RAPOR")
    print("=" * 60)
    print(f"Toplam dosya:       {len(json_files)}")
    print(f"İşlenen:            {stats['processed']}")
    print(f"Atlanan:            {stats['skipped']}")
    print(f"Hata:               {stats['errors']}")
    print(f"Düzeltilen null:    {stats['total_null_fixed']} frame")
    if stats["processed"] > 0:
        avg_jitter_red = stats["total_jitter_reduction"] / stats["processed"]
        print(f"Ort. jitter azalma: %{avg_jitter_red:.1f}")
    print(f"Süre:               {elapsed:.1f} saniye")
    print(f"Yedek klasörü:      {os.path.abspath(backup_dir)}")
    print()

    # En çok iyileşen kelimeler
    if results:
        top_improved = sorted(results, key=lambda r: r["null_fixed"] + r["jitter_reduction_pct"], reverse=True)[:10]
        print("En çok iyileşen kelimeler:")
        for r in top_improved:
            print(f"  {r['word']:25s} — null düzeltme: {r['null_fixed']:2d}, jitter azalma: %{r['jitter_reduction_pct']}")

    print()
    print("Tamamlandı! Backend'i yeniden başlatın:")
    print("  python backend.py")


def _avg_z_jitter(frames):
    """Pose landmark Z jitter ortalaması."""
    diffs = []
    for i in range(1, len(frames)):
        for key in ["11", "12", "15", "16"]:
            p = frames[i].get("pose", {}).get(key)
            pp = frames[i-1].get("pose", {}).get(key)
            if p and pp:
                diffs.append(abs(p[2] - pp[2]))
    return sum(diffs) / len(diffs) if diffs else 0


if __name__ == "__main__":
    main()
