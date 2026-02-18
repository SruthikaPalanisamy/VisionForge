import cv2
import numpy as np
import base64
import time
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import random
import io
import datetime
import os
import json
import uuid

app = FastAPI()

# Setup templates
templates = Jinja2Templates(directory=".")

# Dataset directory setup
DATASET_DIR = "dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

for d in [IMAGES_DIR, LABELS_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

def generate_synthetic_image():
    # Create 500x500 grayscale image
    img = np.ones((500, 500), dtype=np.uint8) * 50
    
    # Metal plate (rectangle) coordinates
    p_w = random.randint(350, 420)
    p_h = random.randint(350, 420)
    x1 = (500 - p_w) // 2
    y1 = (500 - p_h) // 2
    x2 = x1 + p_w
    y2 = y1 + p_h
    
    # Draw metal plate
    cv2.rectangle(img, (x1, y1), (x2, y2), (200), -1)
    
    # Generate 4 holes
    holes = []
    margin = 50
    quads = [
        (x1 + margin, y1 + margin, x1 + p_w//2 - margin, y1 + p_h//2 - margin),
        (x1 + p_w//2 + margin, y1 + margin, x2 - margin, y1 + p_h//2 - margin),
        (x1 + margin, y1 + p_h//2 + margin, x1 + p_w//2 - margin, y2 - margin),
        (x1 + p_w//2 + margin, y1 + p_h//2 + margin, x2 - margin, y2 - margin)
    ]
    
    for q in quads:
        hx = random.randint(q[0], q[2])
        hy = random.randint(q[1], q[3])
        hr = 20 # Target 20
        holes.append({"x": hx, "y": hy, "r": hr})
        cv2.circle(img, (hx, hy), hr, (50), -1)
        
    # Add Gaussian noise
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    ground_truth = {
        "plate_width": p_w,
        "plate_height": p_h,
        "holes": holes
    }
    
    return img, ground_truth

def process_image(img, targets=None):
    start_time = time.time()
    pixel_to_mm = 0.5 # 1mm = 2px -> 0.5 mm/px
    tolerance = 0.02 # 2% as requested
    
    # CV Processing
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_img = img.copy()
    else:
        gray = img
        processed_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Denoising and morphological ops for cleaner edges
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    # Morphological closing to join edges and remove noise
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    summary = {"plate_w": 0, "plate_h": 0, "hole_count": 0, "status": "PASS", "latency": 0, "errors": []}
    
    if contours:
        plate_cnt = max(contours, key=cv2.contourArea)
        px_x, px_y, px_w, px_h = cv2.boundingRect(plate_cnt)
        
        # Dimensions in mm
        m_w = px_w * pixel_to_mm
        m_h = px_h * pixel_to_mm
        
        # Overlay Box & Dimensions
        cv2.rectangle(processed_img, (px_x, px_y), (px_x + px_w, px_y + px_h), (0, 255, 0), 2)
        cv2.putText(processed_img, f"W: {round(m_w,1)}mm", (px_x, px_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(processed_img, f"H: {round(m_h,1)}mm", (px_x - 70, px_y + px_h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Comparison if targets provided
        if targets:
            t_w = float(targets.get("width", m_w))
            t_h = float(targets.get("height", m_h))
            t_hd = float(targets.get("hole_d", 0))

            err_w = abs(m_w - t_w) / max(t_w, 1)
            err_h = abs(m_h - t_h) / max(t_h, 1)

            summary["plate_w"] = {"measured": round(m_w, 2), "expected": round(t_w, 2), "error": round(err_w*100, 2)}
            summary["plate_h"] = {"measured": round(m_h, 2), "expected": round(t_h, 2), "error": round(err_h*100, 2)}

            if err_w > tolerance:
                summary["status"] = "FAIL"
                summary["errors"].append(f"Width error: {'+' if m_w > t_w else ''}{round(m_w - t_w, 1)}mm")
            if err_h > tolerance:
                summary["status"] = "FAIL"
                summary["errors"].append(f"Height error: {'+' if m_h > t_h else ''}{round(m_h - t_h, 1)}mm")

            results.append({"feature": "Plate Width", "measured": f"{round(m_w, 2)} mm", "expected": f"{round(t_w, 2)} mm", "status": "PASS" if err_w <= tolerance else "FAIL"})
            results.append({"feature": "Plate Height", "measured": f"{round(m_h, 2)} mm", "expected": f"{round(t_h, 2)} mm", "status": "PASS" if err_h <= tolerance else "FAIL"})

        # Holes
        detected_holes = []
        for cnt in contours:
            if cnt is plate_cnt: continue
            area = cv2.contourArea(cnt)
            if 300 < area < 10000:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                detected_holes.append({"x": int(x), "y": int(y), "r": int(radius)})
                m_rd = radius * 2 * pixel_to_mm # diameter in mm
                
                color = (255, 0, 0)
                if targets and abs(m_rd - float(targets.get("hole_d", m_rd))) / max(float(targets.get("hole_d", m_rd)), 1) > tolerance:
                    color = (0, 0, 255)
                    summary["status"] = "FAIL"
                
                cv2.circle(processed_img, (int(x), int(y)), int(radius), color, 2)
                cv2.putText(processed_img, f"D:{round(m_rd,1)}", (int(x)-15, int(y)-int(radius+5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        summary["hole_count"] = {"measured": len(detected_holes), "expected": 4}
        if len(detected_holes) != 4: 
            summary["status"] = "FAIL"
            summary["errors"].append(f"Hole Count error: {len(detected_holes)} / 4")

    summary["latency"] = round((time.time() - start_time) * 1000, 2)
    return processed_img, results, summary

def img_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate")
async def generate():
    id_str = f"INS-{random.randint(1000, 9999)}"
    img, ground_truth = generate_synthetic_image()
    orig_b64 = img_to_base64(img)
    
    proc_img, results, summary = process_image(img, ground_truth)
    proc_b64 = img_to_base64(proc_img)
    
    return JSONResponse({
        "id": id_str,
        "timestamp": datetime.datetime.now().strftime("%I:%M:%S %p"),
        "original": orig_b64,
        "processed": proc_b64,
        "results": results,
        "summary": summary
    })

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    id_str = f"INS-{random.randint(1000, 9999)}"
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse({"error": "Invalid image format"}, status_code=400)
    
    orig_b64 = img_to_base64(img)
    proc_img, results, summary = process_image(img)
    proc_b64 = img_to_base64(proc_img)
    
    return JSONResponse({
        "id": id_str,
        "timestamp": datetime.datetime.now().strftime("%I:%M:%S %p"),
        "original": orig_b64,
        "processed": proc_b64,
        "results": results,
        "summary": summary
    })

@app.post("/inspect_live")
async def inspect_live(
    file: UploadFile = File(...),
    target_w: float = 0,
    target_h: float = 0,
    target_hole_d: float = 0
):
    id_str = f"LVE-{random.randint(1000, 9999)}"
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse({"error": "Invalid image format"}, status_code=400)
    
    targets = {"width": target_w, "height": target_h, "hole_d": target_hole_d}
    orig_b64 = img_to_base64(img)
    proc_img, results, summary = process_image(img, targets)
    proc_b64 = img_to_base64(proc_img)
    
    return JSONResponse({
        "id": id_str,
        "timestamp": datetime.datetime.now().strftime("%I:%M:%S %p"),
        "original": orig_b64,
        "processed": proc_b64,
        "results": results,
        "summary": summary
    })

@app.post("/save_to_dataset")
async def save_to_dataset(request: Request):
    try:
        data = await request.json()
        image_b64 = data.get("image")
        metadata = data.get("metadata")
        
        if not image_b64 or not metadata:
            return JSONResponse({"error": "Missing image or metadata"}, status_code=400)
        
        # Decode and save image
        image_data = base64.b64decode(image_b64)
        file_id = str(uuid.uuid4())
        image_filename = f"{file_id}.jpg"
        image_path = os.path.join(IMAGES_DIR, image_filename)
        
        with open(image_path, "wb") as f:
            f.write(image_data)
            
        # Save labels (metadata)
        label_filename = f"{file_id}.json"
        label_path = os.path.join(LABELS_DIR, label_filename)
        
        with open(label_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        return JSONResponse({"status": "success", "id": file_id})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
