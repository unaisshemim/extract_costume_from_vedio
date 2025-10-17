import cv2
import os
import sys
from ultralytics import YOLO
from PIL import Image
import imagehash
import shutil
from tqdm import tqdm
import json
import numpy as np


# Configuration
TEMP_DIR = "temp_frames"
FINAL_OUTPUT_DIR = "final_output"
METADATA_FILE = "frame_metadata.json"
FRAME_INTERVAL = 5
MIN_CONFIDENCE = 0.5
HASH_THRESHOLD = 15
COLOR_SIMILARITY_THRESHOLD = 0.85


# ===================== Utility Functions =====================

def extract_frames(video_path, output_dir, every_n_frames=FRAME_INTERVAL):
    """Extract frames from video at intervals."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count, saved = 0, 0
    print(f"🎞 Extracting every {every_n_frames}th frame...")

    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % every_n_frames == 0:
                path = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
                cv2.imwrite(path, frame)
                saved += 1
            frame_count += 1
            pbar.update(1)
    cap.release()
    print(f"✅ Extracted {saved} frames.\n")
    return saved


def image_sharpness(image_path):
    """Sharpness via Laplacian variance."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    return cv2.Laplacian(img, cv2.CV_64F).var()


def calculate_color_histogram(image_path):
    """Color histogram (for dress similarity)."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    return np.concatenate([hist_h, hist_s])


def compare_color_histograms(hist1, hist2):
    if hist1 is None or hist2 is None:
        return 0.0
    return cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)


# ===================== Pose-based Visibility Scoring =====================

def calculate_body_visibility(keypoints):
    """
    Calculate how much of the body is visible using YOLOv8 pose keypoints.
    Returns a normalized score 0–1.
    """
    if keypoints is None or len(keypoints) == 0:
        return 0.0

    # Extract the numpy array from keypoints object
    if hasattr(keypoints, 'data'):
        kps = keypoints.data.cpu().numpy()  # (1, 17, 3) or (17, 3)
        if len(kps.shape) == 3:  # (1, 17, 3)
            kps = kps[0]  # Take first (and only) detection
    elif hasattr(keypoints, 'xy'):
        # Alternative access pattern for different YOLO versions
        kps = keypoints.xy.cpu().numpy()[0]  # (17, 3)
    else:
        # Fallback - try direct conversion
        kps = keypoints.cpu().numpy()
        if len(kps.shape) == 3:  # (1, 17, 3)
            kps = kps[0]  # Take first detection
    visible = kps[:, 2] > 0.2
    visible_count = np.sum(visible)
    total_count = len(kps)

    # Define indices for upper and lower body
    upper_idx = [0, 1, 2, 3, 4, 5, 6]       # head + shoulders + elbows + wrists
    lower_idx = [11, 12, 13, 14, 15, 16]    # hips + knees + ankles

    upper_visible = np.sum(visible[upper_idx])
    lower_visible = np.sum(visible[lower_idx])

    if total_count == 0:
        return 0.0

    visibility_ratio = visible_count / total_count
    top_ratio = upper_visible / len(upper_idx)
    bottom_ratio = lower_visible / len(lower_idx)
    full_body_balance = 1 - abs(top_ratio - bottom_ratio)

    return (visibility_ratio * 0.6) + (full_body_balance * 0.4)


# ===================== Person Filtering & Scoring =====================

def filter_people_with_pose(input_dir, output_dir):
    """Detects people using YOLO Pose and scores by body visibility."""
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO("yolov8n-pose.pt")

    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])
    metadata = {}
    count = 0

    print("🔍 Detecting people and estimating body visibility...")
    for file in tqdm(files, desc="Analyzing frames"):
        path = os.path.join(input_dir, file)
        img = cv2.imread(path)
        if img is None:
            continue

        results = model(path, verbose=False)
        if len(results) == 0:
            continue

        best_score = -1
        best_box = None
        best_conf = 0
        best_vis = 0

        for r in results:
            for box, keypoints in zip(r.boxes, r.keypoints):
                if int(box.cls) != 0:  # class 0 = person
                    continue
                conf = float(box.conf)
                if conf < MIN_CONFIDENCE:
                    continue

                body_visibility = calculate_body_visibility(keypoints)
                if body_visibility > best_score:
                    best_score = body_visibility
                    best_conf = conf
                    best_box = box
                    best_vis = body_visibility

        if best_box is not None:
            shutil.copy(path, os.path.join(output_dir, file))
            metadata[file] = {
                "confidence": best_conf,
                "body_visibility": best_vis,
                "sharpness": 0
            }
            count += 1

    with open(os.path.join(output_dir, METADATA_FILE), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Found {count} valid frames with pose metadata.\n")
    return count


# ===================== Duplicate Removal =====================

def remove_duplicates(folder):
    print("🧩 Removing similar frames and selecting best full-body images...")
    metadata_path = os.path.join(folder, METADATA_FILE)
    if not os.path.exists(metadata_path):
        print("❌ Metadata not found.")
        return 0

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Pre-compute all image features in one pass
    files = sorted([f for f in os.listdir(folder) if f.endswith(".jpg") and f in metadata])
    image_hashes = {}
    color_hist = {}
    
    print("📊 Pre-computing image features...")
    for filename in tqdm(files, desc="Computing features"):
        path = os.path.join(folder, filename)
        # Compute all features in one pass to minimize I/O
        metadata[filename]["sharpness"] = image_sharpness(path)
        color_hist[filename] = calculate_color_histogram(path)
        image_hashes[filename] = imagehash.phash(Image.open(path))

    # Fast grouping using pre-computed hashes
    processed, image_groups = set(), []
    
    for i, f1 in enumerate(tqdm(files, desc="Grouping")):
        if f1 in processed:
            continue
        
        h1 = image_hashes[f1]
        hist1 = color_hist[f1]
        group = [f1]
        processed.add(f1)

        # Only check remaining files
        for f2 in files[i+1:]:
            if f2 in processed:
                continue
            
            h2 = image_hashes[f2]
            hist2 = color_hist[f2]
            
            # Quick hash comparison first (faster)
            if abs(h1 - h2) < HASH_THRESHOLD:
                group.append(f2)
                processed.add(f2)
            # Only do expensive color comparison if hash didn't match
            elif compare_color_histograms(hist1, hist2) > COLOR_SIMILARITY_THRESHOLD:
                group.append(f2)
                processed.add(f2)

        if group:  # Only add non-empty groups
            image_groups.append(group)

    to_remove = set()
    kept = []
    # ==================== MODIFIED SECTION START ====================
    for group in tqdm(image_groups, desc="Selecting best"):
        if not group:
            continue

        # Select the best file using a hierarchical key:
        # 1. Prioritize the highest body_visibility score.
        # 2. Use sharpness as a tie-breaker.
        # 3. Use confidence as a final tie-breaker.
        best_file = max(group, key=lambda f: (
            metadata[f]['body_visibility'],
            metadata[f]['sharpness'],
            metadata[f]['confidence']
        ))

        kept.append(best_file)
        for f in group:
            if f != best_file:
                to_remove.add(os.path.join(folder, f))
    # ===================== MODIFIED SECTION END =====================

    for path in to_remove:
        if os.path.exists(path):
            os.remove(path)

    os.remove(metadata_path)
    print(f"✅ Kept {len(kept)} best images, removed {len(to_remove)} similar ones.")
    return len(kept)


# ===================== Organization & Pipeline =====================

def process_video(video_path):
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)

    # Clean up existing directories
    for folder in [TEMP_DIR, FINAL_OUTPUT_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    extracted = extract_frames(video_path, TEMP_DIR)
    if extracted == 0:
        print("❌ No frames extracted.")
        sys.exit(1)

    detected = filter_people_with_pose(TEMP_DIR, FINAL_OUTPUT_DIR)
    if detected == 0:
        print("❌ No people detected.")
        sys.exit(1)

    final_count = remove_duplicates(FINAL_OUTPUT_DIR)
    shutil.rmtree(TEMP_DIR)

    print(f"🎉 Done! {final_count} final high-quality full-body frames saved in {FINAL_OUTPUT_DIR}.\n")


# ===================== Main =====================

def main():
    # Make sure to place your video file and name it "input.mp4"
    # in the same directory as this script.
    video_path = "input.mp4"
    process_video(video_path)


if __name__ == "__main__":
    main()