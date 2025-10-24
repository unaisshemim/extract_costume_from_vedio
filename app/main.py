from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import os
import sys
import tempfile
import shutil
from ultralytics import YOLO
from PIL import Image
import imagehash
from tqdm import tqdm
import json
import numpy as np
import io
import zipfile
from typing import List
import uuid
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="Video Frame Extractor API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
FRAME_INTERVAL = 5
MIN_CONFIDENCE = 0.5
HASH_THRESHOLD = 10
COLOR_SIMILARITY_THRESHOLD = 0.85
MAX_FRAMES = 300

# Load YOLO model globally once for better performance
model_pose = YOLO("yolov8n-pose.pt")

# ThreadPool for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=4)


# ===================== Utility Functions =====================

def extract_frames_fast(video_path, every_n_frames=FRAME_INTERVAL, max_frames=MAX_FRAMES):
    """Extract frames efficiently using OpenCV with skipping."""
    print("Starting extract_frames_fast function")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video file")
        raise HTTPException(status_code=400, detail=f"Cannot open video file")
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total}")
    frames = []
    count = 0

    while count < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += every_n_frames
        if len(frames) >= max_frames:
            print(f"Reached max_frames limit: {max_frames}")
            break
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    return frames


def image_sharpness(frame):
    """Sharpness via Laplacian variance."""
    if isinstance(frame, str):
        print(f"Calculating sharpness for {frame}")
        img = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if img is None:
        print("Image could not be loaded")
        return 0
    sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
    print(f"Sharpness calculated: {sharpness}")
    return sharpness


def calculate_color_histogram(frame):
    """Color histogram (for dress similarity)."""
    if isinstance(frame, str):
        print(f"Calculating color histogram for {frame}")
        img = cv2.imread(frame)
    else:
        img = frame
    
    if img is None:
        print("Image could not be loaded")
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print("Converted to HSV")
    hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist = np.concatenate([hist_h, hist_s])
    print("Color histogram calculated")
    return hist


def compare_color_histograms(hist1, hist2):
    print("Comparing color histograms")
    if hist1 is None or hist2 is None:
        print("One or both histograms are None")
        return 0.0
    similarity = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)
    print(f"Histogram similarity: {similarity}")
    return similarity


# ===================== Pose-based Visibility Scoring =====================

def calculate_body_visibility(keypoints):
    """
    Calculate how much of the body is visible using YOLOv8 pose keypoints.
    Returns a normalized score 0–1.
    """
    print("Calculating body visibility")
    if keypoints is None or len(keypoints) == 0:
        print("No keypoints provided")
        return 0.0

    # Extract the numpy array from keypoints object
    if hasattr(keypoints, 'data'):
        kps = keypoints.data.cpu().numpy()
        if len(kps.shape) == 3:
            kps = kps[0]
    elif hasattr(keypoints, 'xy'):
        kps = keypoints.xy.cpu().numpy()[0]
    else:
        kps = keypoints.cpu().numpy()
        if len(kps.shape) == 3:
            kps = kps[0]
    
    print("Keypoints extracted")
    visible = kps[:, 2] > 0.2
    visible_count = np.sum(visible)
    total_count = len(kps)

    # Define indices for upper and lower body
    upper_idx = [0, 1, 2, 3, 4, 5, 6]
    lower_idx = [11, 12, 13, 14, 15, 16]

    upper_visible = np.sum(visible[upper_idx])
    lower_visible = np.sum(visible[lower_idx])

    if total_count == 0:
        print("No keypoints")
        return 0.0

    visibility_ratio = visible_count / total_count
    top_ratio = upper_visible / len(upper_idx)
    bottom_ratio = lower_visible / len(lower_idx)
    full_body_balance = 1 - abs(top_ratio - bottom_ratio)

    score = (visibility_ratio * 0.6) + (full_body_balance * 0.4)
    print(f"Body visibility score: {score}")
    return score


# ===================== Person Filtering & Scoring =====================

def filter_people_with_pose_fast(frames):
    """Batch process frames with YOLO Pose."""
    print("Starting filter_people_with_pose_fast")
    results = model_pose(frames, verbose=False)
    output_frames = []
    metadata = {}

    for idx, r in enumerate(results):
        print(f"Processing frame {idx}")
        # Check if any person (class 0) is detected
        if len(r.boxes) == 0:
            print("No persons detected, skipping")
            continue
        
        persons = r.boxes.cls == 0
        if not persons.any():
            print("No persons detected, skipping")
            continue

        best_box = None
        best_score = -1
        best_vis = 0
        best_conf = 0

        for box, keypoints in zip(r.boxes, r.keypoints):
            if int(box.cls) != 0:
                print("Not a person, skipping")
                continue
            conf = float(box.conf)
            if conf < MIN_CONFIDENCE:
                print(f"Confidence too low: {conf}, skipping")
                continue

            vis = calculate_body_visibility(keypoints)
            if vis > best_score:
                best_score = vis
                best_box = box
                best_vis = vis
                best_conf = conf
                print(f"New best score: {best_score}")

        if best_box is not None:
            output_frames.append(frames[idx])
            metadata[idx] = {
                "confidence": best_conf,
                "body_visibility": best_vis,
                "sharpness": image_sharpness(frames[idx])
            }
            print(f"Added frame {idx} to output")

    print(f"Filtered {len(output_frames)} people frames")
    return output_frames, metadata


# ===================== Duplicate Removal =====================

def remove_duplicates_fast(frames, metadata):
    """Fast duplicate removal using hash clustering (no pairwise O(n²))."""
    print("Starting remove_duplicates_fast")
    print(f"Processing {len(frames)} frames")
    
    # Convert frames to PIL images and compute hashes
    hashes = []
    for f in frames:
        pil_img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        hashes.append(imagehash.phash(pil_img))
    
    uniq_indices = []
    seen = set()

    for i, h in enumerate(hashes):
        found_similar = False
        for j in seen:
            if abs(h - hashes[j]) < HASH_THRESHOLD:
                found_similar = True
                print(f"Frame {i} is similar to frame {j}")
                # Compare with existing similar frame to keep the better one
                if i in metadata and j in metadata:
                    if (metadata[i]['body_visibility'], metadata[i]['sharpness'], metadata[i]['confidence']) > \
                       (metadata[j]['body_visibility'], metadata[j]['sharpness'], metadata[j]['confidence']):
                        # Replace j with i
                        seen.discard(j)
                        uniq_indices.remove(j)
                        found_similar = False
                break
        
        if not found_similar:
            uniq_indices.append(i)
            seen.add(i)
            print(f"Kept frame {i} as unique")
    
    unique_frames = [frames[i] for i in uniq_indices]
    print(f"Kept {len(unique_frames)} unique frames")
    return unique_frames


# ===================== Video Processing Pipeline =====================

def process_video_fast(video_path):
    """Process video efficiently in-memory without temp files."""
    print("Starting process_video_fast")
    
    # Extract frames to memory
    frames = extract_frames_fast(video_path, every_n_frames=FRAME_INTERVAL)
    if len(frames) == 0:
        print("No frames extracted")
        raise HTTPException(status_code=400, detail="No frames extracted from video")
    print(f"Extracted {len(frames)} frames")

    # Detect people in frames
    people_frames, metadata = filter_people_with_pose_fast(frames)
    if len(people_frames) == 0:
        print("No people detected")
        raise HTTPException(status_code=400, detail="No people detected in frames")
    print(f"Detected {len(people_frames)} people frames")

    # Remove duplicates
    unique_frames = remove_duplicates_fast(people_frames, metadata)
    print(f"Kept {len(unique_frames)} unique frames")

    # Encode to JPEG bytes
    imgs = []
    for i, frame in enumerate(unique_frames):
        _, buffer = cv2.imencode(".jpg", frame)
        imgs.append({
            "filename": f"frame_{i:04d}.jpg",
            "data": buffer.tobytes()
        })
    
    return imgs, len(unique_frames)


# ===================== API Endpoints =====================

@app.get("/")
async def root():
    print("Root endpoint called")
    return {
        "message": "Video Frame Extractor API",
        "endpoints": {
            "/process-video": "POST - Upload video, returns ZIP of extracted frames",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health():
    print("Health endpoint called")
    return {"status": "healthy"}


@app.post("/process-video")
async def process_video_endpoint(file: UploadFile = File(...)):
    """
    Accept video file as blob, process it, and return images as a ZIP blob.
    """
    print("Process video endpoint called")
    if not file.content_type.startswith('video/'):
        print("Invalid file type")
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Create temporary working directory
    work_dir = tempfile.mkdtemp()
    print(f"Created work dir: {work_dir}")
    
    try:
        # Save uploaded video
        video_path = os.path.join(work_dir, f"input_{uuid.uuid4()}.mp4")
        with open(video_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        print(f"Saved video to {video_path}")
        
        # Process video (fast in-memory version)
        processed_images, count = process_video_fast(video_path)
        
        if count == 0:
            print("No valid frames found")
            raise HTTPException(status_code=400, detail="No valid frames found")
        
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for img_data in processed_images:
                # Add to ZIP with sequential naming
                zip_file.writestr(img_data["filename"], img_data["data"])
                print(f"Added {img_data['filename']} to ZIP")
        
        # Reset buffer position to beginning
        zip_buffer.seek(0)
        print("ZIP created successfully")
        
        # Return ZIP as streaming response
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=extracted_frames.zip",
                "X-Frame-Count": str(count)
            }
        )
    
    except Exception as e:
        print(f"Exception occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temporary directory
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            print("Cleaned up work dir")


@app.post("/process-video-json")
async def process_video_json_endpoint(file: UploadFile = File(...)):
    """
    Alternative endpoint that returns image blobs as base64 encoded JSON array.
    """
    print("Process video JSON endpoint called")
    if not file.content_type.startswith('video/'):
        print("Invalid file type")
        raise HTTPException(status_code=400, detail="File must be a video")
    
    work_dir = tempfile.mkdtemp()
    print(f"Created work dir: {work_dir}")
    
    try:
        # Save uploaded video
        video_path = os.path.join(work_dir, f"input_{uuid.uuid4()}.mp4")
        with open(video_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        print(f"Saved video to {video_path}")
        
        # Process video (fast in-memory version)
        processed_images, count = process_video_fast(video_path)
        
        if count == 0:
            print("No valid frames found")
            raise HTTPException(status_code=400, detail="No valid frames found")
        
        # Encode all images as base64
        import base64
        images_data = []
        for img_data in processed_images:
            print(f"Encoding {img_data['filename']}")
            img_base64 = base64.b64encode(img_data["data"]).decode('utf-8')
            images_data.append({
                "filename": img_data["filename"],
                "data": img_base64,
                "mime_type": "image/jpeg"
            })
        
        print("Returning JSON response")
        return {
            "count": count,
            "images": images_data
        }
    
    except Exception as e:
        print(f"Exception occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            print("Cleaned up work dir")


if __name__ == "__main__":
    print("Starting FastAPI server")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)