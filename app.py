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

app = FastAPI()

# Setup templates
templates = Jinja2Templates(directory=".")

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

def process_image(img, ground_truth=None):
    start_time = time.time()
    # Units in px for this specific UI request
    tolerance = 0.05 # 5% tolerance for px matching
    
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
    summary = {"plate_w": 0, "plate_h": 0, "hole_count": 0, "status": "PASS", "latency": 0}
    
    if contours:
        plate_cnt = max(contours, key=cv2.contourArea)
        px_x, px_y, px_w, px_h = cv2.boundingRect(plate_cnt)
        
        # Overlay Bounding Box (Green) - matches screenshot style
        cv2.rectangle(processed_img, (px_x, px_y), (px_x + px_w, px_y + px_h), (0, 255, 0), 2)
        
        m_w = px_w
        e_w = ground_truth["plate_width"] if ground_truth else 500 # Ref image shows 500px target
        error_w = abs(m_w - e_w) / max(e_w, 1)
        
        m_h = px_h
        e_h = ground_truth["plate_height"] if ground_truth else 500 # Ref image shows 500px target
        error_h = abs(m_h - e_h) / max(e_h, 1)

        summary["plate_w"] = {"measured": round(m_w, 2), "expected": round(e_w, 2), "error": round(error_w*100, 2)}
        summary["plate_h"] = {"measured": round(m_h, 2), "expected": round(e_h, 2), "error": round(error_h*100, 2)}

        results.append({"feature": "Plate Width", "measured": f"{round(m_w, 2)} px", "expected": f"{round(e_w, 2)} px", "error": f"{round(error_w*100, 2)}%", "status": "PASS" if error_w <= tolerance else "FAIL"})
        results.append({"feature": "Plate Height", "measured": f"{round(m_h, 2)} px", "expected": f"{round(e_h, 2)} px", "error": f"{round(error_h*100, 2)}%", "status": "PASS" if error_h <= tolerance else "FAIL"})

        if error_w > tolerance or error_h > tolerance: summary["status"] = "FAIL"

        # Holes
        detected_holes = []
        for cnt in contours:
            if cnt is plate_cnt: continue
            area = cv2.contourArea(cnt)
            # Adjust hole filter for pixel area
            if 300 < area < 10000:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                detected_holes.append({"x": int(x), "y": int(y), "r": int(radius)})
                cv2.circle(processed_img, (int(x), int(y)), int(radius), (255, 0, 0), 2)
                cv2.drawMarker(processed_img, (int(x), int(y)), (255, 0, 0), cv2.MARKER_CROSS, 8, 2)

        summary["hole_count"] = {"measured": len(detected_holes), "expected": 4}
        # In the screenshot it fails if count != 4
        if len(detected_holes) != 4: summary["status"] = "FAIL"

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
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original": orig_b64,
        "processed": proc_b64,
        "results": results,
        "summary": summary
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
