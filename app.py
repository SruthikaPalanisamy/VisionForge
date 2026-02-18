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

# Mount static directory for dataset image serving
app.mount("/static", StaticFiles(directory="static"), name="static")

# Dataset directory setup
DATASET_DIR = "static/dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

for d in [IMAGES_DIR, LABELS_DIR]:
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def generate_synthetic_image(t_w=200, t_h=200, t_hd=10, t_hc=4):
    # 1mm = 2px calibration
    # Target dimensions in pixels
    p_w = int(t_w * 2) 
    p_h = int(t_h * 2)
    hr = int(t_hd) # 1mm = 2px, so radius in px = diameter in mm
    
    # Add slight random variation (+/- 2px) to simulated target
    p_w += random.randint(-2, 2)
    p_h += random.randint(-2, 2)
    
    # Create background based on plate size with margins
    canvas_w = max(p_w + 100, 500)
    canvas_h = max(p_h + 100, 500)
    img = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 50
    
    x1 = (canvas_w - p_w) // 2
    y1 = (canvas_h - p_h) // 2
    x2 = x1 + p_w
    y2 = y1 + p_h
    
    # Draw metal plate
    cv2.rectangle(img, (x1, y1), (x2, y2), (200), -1)
    
    # Generate holes
    holes = []
    margin = 40
    
    # Distribute holes in a grid-like pattern based on target count
    cols = int(np.ceil(np.sqrt(t_hc)))
    rows = int(np.ceil(t_hc / cols))
    
    for i in range(t_hc):
        row = i // cols
        col = i % cols
        
        # Calculate cell coordinates
        cell_w = (p_w - 2*margin) // cols
        cell_h = (p_h - 2*margin) // rows
        
        cx_base = x1 + margin + col * cell_w + cell_w // 2
        cy_base = y1 + margin + row * cell_h + cell_h // 2
        
        # Jitter hole position
        hx = cx_base + random.randint(-margin//2, margin//2)
        hy = cy_base + random.randint(-margin//2, margin//2)
        
        # Jitter hole radius slightly
        h_hr = hr + random.choice([-1, 0, 1])
        
        holes.append({"x": hx, "y": hy, "r": h_hr})
        cv2.circle(img, (hx, hy), h_hr, (50), -1)
        
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
    PPM = 2.0 # Pixels Per Millimeter
    pixel_to_mm = 1.0 / PPM 
    tolerance = 0.02 # 2%
    
    # CV Preprocessing
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_img = img.copy()
    else:
        gray = img
        processed_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 1. Preprocessing: Strong blur to smooth surface scratches/grain
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # 2. Binary Thresholding: Inverse + Otsu to isolate features
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Clean Up: Morphological opening for small noise
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find potential features
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    summary = {"plate_w": 0, "plate_h": 0, "hole_count": 0, "status": "PASS", "latency": 0, "errors": []}
    
    # Plate Detection (Canny boundary)
    edges = cv2.Canny(blurred, 30, 100)
    plate_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if plate_contours:
        plate_cnt = max(plate_contours, key=cv2.contourArea)
        px_x, px_y, px_w, px_h = cv2.boundingRect(plate_cnt)
        
        # Dimensions in mm
        m_w = px_w * pixel_to_mm
        m_h = px_h * pixel_to_mm
        
        # Overlay
        cv2.rectangle(processed_img, (px_x, px_y), (px_x + px_w, px_y + px_h), (0, 255, 0), 2)
        cv2.putText(processed_img, f"W: {round(m_w,1)}mm", (px_x, px_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(processed_img, f"H: {round(m_h,1)}mm", (px_x - 70, px_y + px_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if targets:
            t_w, t_h = float(targets.get("width", m_w)), float(targets.get("height", m_h))
            err_w, err_h = abs(m_w - t_w) / max(t_w, 1), abs(m_h - t_h) / max(t_h, 1)
            summary["plate_w"] = {"measured": round(m_w, 2), "expected": round(t_w, 2), "error": round(err_w*100, 2)}
            summary["plate_h"] = {"measured": round(m_h, 2), "expected": round(t_h, 2), "error": round(err_h*100, 2)}
            
            if err_w > tolerance:
                summary["status"] = "FAIL"
                summary["errors"].append(f"Width error: {'+' if m_w > t_w else ''}{round(m_w - t_w, 1)}mm")
            if err_h > tolerance:
                summary["status"] = "FAIL"
                summary["errors"].append(f"Height error: {'+' if m_h > t_h else ''}{round(m_h - t_h, 1)}mm")
            
            results.append({"feature": "Plate Width", "measured": f"{round(m_w,2)}mm", "expected": f"{round(t_w,2)}mm", "error": f"{round(err_w*100,2)}%"})
            results.append({"feature": "Plate Height", "measured": f"{round(m_h,2)}mm", "expected": f"{round(t_h,2)}mm", "error": f"{round(err_h*100,2)}%"})

    # 4. Areal Filtering (100 - 2000 pixels)
    detected_holes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 2000:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = float(bw)/bh
            
            if 0.7 < aspect_ratio < 1.3:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                detected_holes.append({"x": int(x), "y": int(y), "r": int(radius)})
                m_rd = radius * 2 * pixel_to_mm
                
                color = (255, 0, 0)
                if targets:
                    t_rd = float(targets.get("hole_d", m_rd))
                    if abs(m_rd - t_rd) / max(t_rd, 1) > tolerance:
                        color = (0, 0, 255)
                        summary["status"] = "FAIL"
                
                cv2.circle(processed_img, (int(x), int(y)), int(radius), color, 2)
                cv2.putText(processed_img, f"D:{round(m_rd,1)}mm", (int(x)-15, int(y)-int(radius+5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    t_hc = int(targets.get("hole_count", 4)) if targets else 4
    summary["hole_count"] = {"measured": len(detected_holes), "expected": t_hc}
    if len(detected_holes) != t_hc: 
        summary["status"] = "FAIL"
        summary["errors"].append(f"Hole Count error: {len(detected_holes)} / {t_hc}")

    summary["latency"] = round((time.time() - start_time) * 1000, 2)
    return processed_img, results, summary

def img_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate")
async def generate(
    target_w: float = 200,
    target_h: float = 200,
    target_hole_d: float = 10,
    target_hole_count: int = 4
):
    id_str = f"INS-{random.randint(1000, 9999)}"
    # Use user targets to guide the generation
    img, ground_truth = generate_synthetic_image(target_w, target_h, target_hole_d, target_hole_count)
    orig_b64 = img_to_base64(img)
    
    targets = {"width": target_w, "height": target_h, "hole_d": target_hole_d, "hole_count": target_hole_count}
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

@app.post("/process_upload")
async def process_upload(
    file: UploadFile = File(...),
    target_w: float = 0,
    target_h: float = 0,
    target_hole_d: float = 0,
    target_hole_count: int = 4
):
    id_str = f"INS-{random.randint(1000, 9999)}"
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse({"error": "Invalid image format"}, status_code=400)
    
    targets = {"width": target_w, "height": target_h, "hole_d": target_hole_d, "hole_count": target_hole_count}
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

@app.post("/inspect_live")
async def inspect_live(
    file: UploadFile = File(...),
    target_w: float = 0,
    target_h: float = 0,
    target_hole_d: float = 0,
    target_hole_count: int = 4
):
    id_str = f"LVE-{random.randint(1000, 9999)}"
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse({"error": "Invalid image format"}, status_code=400)
    
    targets = {"width": target_w, "height": target_h, "hole_d": target_hole_d, "hole_count": target_hole_count}
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
        
        status = metadata.get("summary", {}).get("status", "UNKNOWN")
        file_id = f"{status}_{uuid.uuid4()}"
        
        # Decode and save image
        image_data = base64.b64decode(image_b64)
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

@app.get("/get_dataset_images")
async def get_dataset_images():
    try:
        files = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')]
        # Sort by creation time (newest first)
        files.sort(key=lambda x: os.path.getmtime(os.path.join(IMAGES_DIR, x)), reverse=True)
        return JSONResponse({"images": files})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
