from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
import uvicorn
import joblib
import numpy as np
from PIL import Image, ImageOps
import io
import os
from scipy import ndimage
import math

app = FastAPI()

# Load model
model_path = os.path.join(os.path.dirname(__file__), "../model/mnist_model.pkl")
model = None

@app.on_event("startup")
async def load_model():
    global model
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    else:
        print("Model file not found. Please train the model first.")

import time

def preprocess_image(image: Image.Image) -> np.ndarray:
    # Convert to grayscale
    image = image.convert('L')
    
    # Invert colors (white digit on black background)
    image = ImageOps.invert(image)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Threshold to remove noise/artifacts
    img_array[img_array < 50] = 0
    
    # Check for empty input
    if np.sum(img_array) < 1000: # Arbitrary threshold, adjust as needed
        return None
    
    # Find bounding box
    rows = np.any(img_array, axis=1)
    cols = np.any(img_array, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
        
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Crop the digit
    digit = img_array[rmin:rmax+1, cmin:cmax+1]
    
    # Resize to fit in 20x20 box while preserving aspect ratio
    h, w = digit.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    digit_pil = Image.fromarray(digit.astype(np.uint8))
    digit_pil = digit_pil.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    digit_resized = np.array(digit_pil)
    
    # Create 28x28 black canvas
    final_img = np.zeros((28, 28))
    
    # Calculate center of mass
    cy, cx = ndimage.center_of_mass(digit_resized)
    
    paste_y = int(14 - cy)
    paste_x = int(14 - cx)
    
    start_y = max(0, paste_y)
    start_x = max(0, paste_x)
    end_y = min(28, paste_y + new_h)
    end_x = min(28, paste_x + new_w)
    
    src_start_y = start_y - paste_y
    src_start_x = start_x - paste_x
    src_end_y = src_start_y + (end_y - start_y)
    src_end_x = src_start_x + (end_x - start_x)
    
    final_img[start_y:end_y, start_x:end_x] = digit_resized[src_start_y:src_end_y, src_start_x:src_end_x]
    
    # Debug: Save preprocessed image
    debug_dir = "debug_images"
    os.makedirs(debug_dir, exist_ok=True)
    timestamp = int(time.time() * 1000)
    Image.fromarray(final_img.astype(np.uint8)).save(f"{debug_dir}/proc_{timestamp}.png")
    
    return final_img.reshape(1, -1) / 255.0

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model
    if model is None:
        # Try loading again
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            return {"error": "Model not loaded"}
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Preprocess
    img_array = preprocess_image(image)
    
    if img_array is None:
        return {"error": "No digit detected"}
    
    prediction = model.predict(img_array)
    return {"digit": int(prediction[0])}

# Mount static files last to avoid overriding API routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
