"""
best_landmark_picker.py — TSL Multi-Video En Iyi Landmark Secici
=================================================================

Dataset yapisi:
  - Video klasoru: signerX_sampleY_color.mp4 + signerX_sampleY_depth.mp4
  - Label dosyasi: train_labels.csv (signerX_sampleY,class_id)
  - Label map:     label_map.json veya mevcut landmark JSON'larindan class_id->kelime

Her kelime icin:
  1. train_labels.csv'den o kelimeye ait tum sample'lari bulur
  2. Her sample'in _color.mp4 videosundan landmark cikarir
  3. Kalite skorlar, en iyi 1 tanesini secer
  4. O kelimenin landmark JSON'u olarak kaydeder

Kullanim:
    # Tum kelimeleri isle (uzun surer ~1-2 saat):
    python best_landmark_picker.py --all

    # Sadece sorunlu kelimeleri isle (hizli):
    python best_landmark_picker.py --bad-only

    # Tek kelime isle:
    python best_landmark_picker.py --word tamam

    # Dry-run (dosya yazmadan analiz):
    python best_landmark_picker.py --bad-only --dry-run

    # Kelime basina max N video dene (hizlandirmak icin):
    python best_landmark_picker.py --all --max-videos 10

Konfigürasyon: asagidaki PATHS bolumunu kendi sisteminize gore duzenleyin.
"""

import os, sys, json, csv, shutil, time
import cv2
import numpy as np

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions,
    HandLandmarker, HandLandmarkerOptions,
)
from mediapipe import Image as MpImage
import mediapipe as mp

# ═══════════════════════════════════════════════════════════════
# KONFIGÜRASYON — Kendi sisteminize gore duzenleyin
# ═══════════════════════════════════════════════════════════════

PATHS = {
    "video_dir":    r"C:\Users\ASUS\Desktop\trainset\train_set_vfbha39\train",
    "labels_csv":   r"train_labels.csv",
    "landmarks_dir": r"dataset\landmarks",
    "label_map":    r"model_assets\label_map.json",
}

# Landmark extraction ayarlari
POSE_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24]
TARGET_FRAMES = 30
BAD_SCORE_THRESHOLD = 75   # Bu skorun altindaki kelimeler "sorunlu"
MAX_VIDEOS_DEFAULT = 20    # Kelime basina max video (hiz icin)

# Model dosyalari
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mp_models")
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
POSE_MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_full.task")
HAND_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")

import urllib.request

def ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for url, path, name in [
        (POSE_MODEL_URL, POSE_MODEL_PATH, "Pose"),
        (HAND_MODEL_URL, HAND_MODEL_PATH, "Hand"),
    ]:
        if not os.path.exists(path):
            print(f"  {name} modeli indiriliyor...")
            urllib.request.urlretrieve(url, path)
            print(f"  OK")


def create_landmarkers():
    return (
        PoseLandmarker.create_from_options(PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
            num_poses=1,
        )),
        HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        ))
    )


# ═══════════════════════════════════════════════════════════════
# Label / Mapping Yukleyiciler
# ═══════════════════════════════════════════════════════════════

def load_labels_csv(csv_path):
    """train_labels.csv -> {sample_name: class_id}"""
    mapping = {}
    with open(csv_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2:
                mapping[parts[0]] = int(parts[1])
    return mapping


def load_class_to_word(label_map_path, landmarks_dir):
    """
    class_id -> kelime eslesmesi.
    Oncelik: label_map.json, yoksa mevcut landmark JSON'larindan cikarir.
    """
    class_to_word = {}

    # 1. label_map.json'dan dene
    if os.path.exists(label_map_path):
        with open(label_map_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for k, v in raw.items():
            cid = int(k)
            if isinstance(v, dict):
                # {"TR": "tamam", "EN": "ok"} formatı
                word = v.get("TR", v.get("tr", v.get("EN", str(cid))))
                word = str(word).lower().replace(" ", "_")
                # Turkce karakter donusumu
                for old, new in [("ç","c"),("ğ","g"),("ı","i"),("ö","o"),("ş","s"),("ü","u")]:
                    word = word.replace(old, new)
                class_to_word[cid] = word
            elif isinstance(v, str):
                class_to_word[cid] = v.lower().replace(" ", "_")

    # 2. Mevcut landmark JSON'larindan tamamla
    if os.path.isdir(landmarks_dir):
        for fname in os.listdir(landmarks_dir):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(landmarks_dir, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                cid = data.get("class_id")
                word = data.get("word_tr", fname[:-5])
                if cid is not None and cid not in class_to_word:
                    class_to_word[int(cid)] = word.lower()
            except:
                pass

    return class_to_word


def group_samples_by_class(labels, class_to_word):
    """
    {class_id: [sample_name, ...]} ve
    {word: [sample_name, ...]} olustur.
    """
    by_class = {}
    for sample, cid in labels.items():
        by_class.setdefault(cid, []).append(sample)

    by_word = {}
    for cid, samples in by_class.items():
        word = class_to_word.get(cid, f"class_{cid}")
        by_word[word] = {
            "class_id": cid,
            "samples": sorted(samples),
        }
    return by_word


# ═══════════════════════════════════════════════════════════════
# Landmark Extraction
# ═══════════════════════════════════════════════════════════════

def extract_from_video(video_path, pose_lm, hand_lm, target_frames=TARGET_FRAMES):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 5:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, min(target_frames, total), dtype=int).tolist()
    frames_data = []
    fi = si = 0

    while cap.isOpened() and si < len(indices):
        ret, frame = cap.read()
        if not ret:
            break
        if fi == indices[si]:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            fd = {"pose": {}, "left_hand": None, "right_hand": None}

            try:
                pr = pose_lm.detect(mp_img)
                if pr.pose_landmarks and len(pr.pose_landmarks) > 0:
                    for idx in POSE_INDICES:
                        lm = pr.pose_landmarks[0][idx]
                        fd["pose"][str(idx)] = [lm.x, lm.y, lm.z]
            except:
                pass

            try:
                hr = hand_lm.detect(mp_img)
                if hr.hand_landmarks:
                    for hi, hlms in enumerate(hr.hand_landmarks):
                        side = "Left"
                        if hr.handedness and hi < len(hr.handedness):
                            side = hr.handedness[hi][0].category_name
                        pts = [[l.x, l.y, l.z] for l in hlms]
                        if side == "Left":
                            fd["left_hand"] = pts
                        else:
                            fd["right_hand"] = pts
            except:
                pass

            frames_data.append(fd)
            si += 1
        fi += 1

    cap.release()
    while len(frames_data) < target_frames and len(frames_data) > 0:
        frames_data.append(frames_data[-1].copy())

    stats = compute_quality(frames_data[:target_frames])
    return {"frames": frames_data[:target_frames], "stats": stats}


def compute_quality(frames):
    n = len(frames)
    if n == 0:
        return {"quality_score": 0}
    null_lh = sum(1 for f in frames if not f.get("left_hand"))
    null_rh = sum(1 for f in frames if not f.get("right_hand"))
    any_hand = sum(1 for f in frames if f.get("left_hand") or f.get("right_hand"))
    max_gap = max(_mg(frames, "left_hand"), _mg(frames, "right_hand"))
    zd = []
    for i in range(1, n):
        for k in ["11","12","15","16"]:
            p = frames[i].get("pose",{}).get(k)
            pp = frames[i-1].get("pose",{}).get(k)
            if p and pp:
                zd.append(abs(p[2]-pp[2]))
    az = np.mean(zd) if zd else 0
    sc = 100 - (null_lh+null_rh)/(2*n)*60 - min(max_gap/n,1)*25 - min(az/0.3,1)*10
    return {
        "quality_score": round(max(0,min(100,sc)),1),
        "null_left_hand": null_lh, "null_right_hand": null_rh,
        "any_hand_frames": any_hand, "max_null_gap": max_gap,
        "avg_z_jitter": round(az,5), "total_frames": n,
    }

def _mg(frames, key):
    mg=c=0
    for f in frames:
        if not f.get(key): c+=1; mg=max(mg,c)
        else: c=0
    return mg


# ═══════════════════════════════════════════════════════════════
# Mevcut Landmark Kalite Kontrol
# ═══════════════════════════════════════════════════════════════

def get_existing_score(landmarks_dir, word):
    """Mevcut landmark'in kalite skorunu dondur (yoksa 0)."""
    path = os.path.join(landmarks_dir, word + ".json")
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Smooth edilmis veri her zaman iyi cikar
        # Backup'tan orijinal skoru kontrol et
        backup_path = os.path.join(landmarks_dir + "_backup", word + ".json")
        check_path = backup_path if os.path.exists(backup_path) else path
        with open(check_path, "r", encoding="utf-8") as f:
            check_data = json.load(f)
        stats = compute_quality(check_data.get("frames", []))
        return stats["quality_score"]
    except:
        return 0


def find_bad_words(landmarks_dir):
    """Sorunlu kelimeleri bul (skor < threshold)."""
    bad = []
    backup_dir = landmarks_dir.rstrip("/\\") + "_backup"

    for fname in sorted(os.listdir(landmarks_dir)):
        if not fname.endswith(".json"):
            continue
        word = fname[:-5]

        # Orijinal (smooth oncesi) veriyi kontrol et
        check_path = os.path.join(backup_dir, fname) if os.path.isdir(backup_dir) else os.path.join(landmarks_dir, fname)
        if not os.path.exists(check_path):
            check_path = os.path.join(landmarks_dir, fname)

        try:
            with open(check_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            stats = compute_quality(data.get("frames", []))
            if stats["quality_score"] < BAD_SCORE_THRESHOLD:
                bad.append((word, stats["quality_score"]))
        except:
            bad.append((word, 0))

    return bad


# ═══════════════════════════════════════════════════════════════
# Ana Program
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="TSL Best Landmark Picker")
    parser.add_argument("--all", action="store_true", help="Tum kelimeleri isle")
    parser.add_argument("--bad-only", action="store_true", help="Sadece sorunlu kelimeleri isle")
    parser.add_argument("--word", type=str, help="Tek kelime isle")
    parser.add_argument("--dry-run", action="store_true", help="Dosya yazmadan analiz")
    parser.add_argument("--max-videos", type=int, default=MAX_VIDEOS_DEFAULT, help="Kelime basina max video")
    parser.add_argument("--video-dir", type=str, default=PATHS["video_dir"])
    parser.add_argument("--labels-csv", type=str, default=PATHS["labels_csv"])
    parser.add_argument("--landmarks-dir", type=str, default=PATHS["landmarks_dir"])
    parser.add_argument("--label-map", type=str, default=PATHS["label_map"])
    args = parser.parse_args()

    if not args.all and not args.bad_only and not args.word:
        parser.print_help()
        print("\nOrnek:")
        print("  python best_landmark_picker.py --bad-only")
        print("  python best_landmark_picker.py --all --max-videos 10")
        print("  python best_landmark_picker.py --word tamam")
        sys.exit(1)

    # Yukle
    print("Veriler yukleniyor...")
    labels = load_labels_csv(args.labels_csv)
    class_to_word = load_class_to_word(args.label_map, args.landmarks_dir)
    word_data = group_samples_by_class(labels, class_to_word)

    print(f"  {len(labels)} sample, {len(class_to_word)} kelime eslesmesi")

    # Hangi kelimeleri isleyecegiz?
    if args.word:
        target_words = [args.word.lower()]
    elif args.bad_only:
        bad = find_bad_words(args.landmarks_dir)
        target_words = [w for w, s in bad]
        print(f"\n  {len(bad)} sorunlu kelime bulundu (skor < {BAD_SCORE_THRESHOLD}):")
        for w, s in sorted(bad, key=lambda x: x[1]):
            print(f"    {w:25s} skor: {s:.1f}")
    else:
        target_words = sorted(word_data.keys())

    if not target_words:
        print("\nIslenecek kelime yok!")
        sys.exit(0)

    print(f"\n{len(target_words)} kelime islenecek")
    print()

    # Modeller
    ensure_models()
    print("Landmarker hazirlaniyor...")
    pose_lm, hand_lm = create_landmarkers()
    print("Hazir.\n")

    # Yedek
    if not args.dry_run and os.path.isdir(args.landmarks_dir):
        backup = args.landmarks_dir.rstrip("/\\") + "_backup_before_picker"
        if not os.path.exists(backup):
            print(f"Yedek aliniyor -> {backup}")
            shutil.copytree(args.landmarks_dir, backup)
    if not args.dry_run:
        os.makedirs(args.landmarks_dir, exist_ok=True)

    t0 = time.time()
    results = []
    improved = 0

    for wi, word in enumerate(sorted(target_words)):
        if word not in word_data:
            print(f"[{wi+1}/{len(target_words)}] {word} — kelime eslemesi bulunamadi, atlaniyor")
            continue

        info = word_data[word]
        samples = info["samples"]
        class_id = info["class_id"]

        # Max video limiti — rastgele sec
        if len(samples) > args.max_videos:
            # Farkli signer'lardan secmeye calis
            np.random.seed(42)  # Tekrarlanabilir
            samples = list(np.random.choice(samples, args.max_videos, replace=False))

        print(f"[{wi+1}/{len(target_words)}] {word} (class:{class_id}, {len(samples)} video)", end="", flush=True)

        existing_score = get_existing_score(args.landmarks_dir, word)
        best_score = -1
        best_data = None
        best_sample = None

        for sample in samples:
            video_path = os.path.join(args.video_dir, sample + "_color.mp4")
            if not os.path.exists(video_path):
                continue
            r = extract_from_video(video_path, pose_lm, hand_lm)
            if r and r["stats"]["quality_score"] > best_score:
                best_score = r["stats"]["quality_score"]
                best_data = r
                best_sample = sample

        if not best_data:
            print(" — video bulunamadi")
            continue

        ns = best_data["stats"]["null_left_hand"] + best_data["stats"]["null_right_hand"]
        change = ""
        if existing_score > 0:
            diff = best_score - existing_score
            change = f" (mevcut:{existing_score:.1f} -> {best_score:.1f}, {'+'if diff>0 else ''}{diff:.1f})"
            if best_score > existing_score:
                improved += 1

        print(f" -> {best_sample} skor:{best_score:.1f} null:{ns}{change}")

        # Kaydet (sadece iyilesse veya mevcut yoksa)
        if not args.dry_run and (best_score >= existing_score or existing_score == 0):
            out = {
                "class_id": class_id,
                "word_tr": word,
                "frame_count": len(best_data["frames"]),
                "hand_detected": best_data["stats"]["any_hand_frames"],
                "pose_indices": POSE_INDICES,
                "source_video": best_sample + "_color.mp4",
                "quality_score": best_score,
                "frames": best_data["frames"],
            }
            with open(os.path.join(args.landmarks_dir, word + ".json"), "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False)

        results.append({"word": word, "score": best_score, "existing": existing_score})

    elapsed = time.time() - t0
    pose_lm.close()
    hand_lm.close()

    # Rapor
    print(f"\n{'='*65}")
    print("RAPOR")
    print(f"{'='*65}")
    print(f"Islenen:      {len(results)} kelime")
    print(f"Iyilesen:     {improved}")
    print(f"Sure:         {elapsed:.0f}s ({elapsed/60:.1f} dk)")
    if results:
        sc = [r["score"] for r in results]
        print(f"Yeni skor:    ort={np.mean(sc):.1f} min={np.min(sc):.1f} max={np.max(sc):.1f}")
    print(f"\nSonraki adim: python batch_smooth.py {args.landmarks_dir} --force")


if __name__ == "__main__":
    main()
