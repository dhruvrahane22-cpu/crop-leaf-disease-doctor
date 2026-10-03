import io
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image

from src.models.custom_cnn import CustomLeafCNN
from src.chatbot.retriever import LeafDoctorChatbot

app = FastAPI(title="Crop Leaf-Disease Doctor API")
chatbot = LeafDoctorChatbot()

# Load class names
TRAIN_DIR = "data/processed/train"
if os.path.exists(TRAIN_DIR):
    CLASS_NAMES = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
else:
    CLASS_NAMES = [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
        "Blueberry___healthy", "Cherry_(including_sour)___healthy", "Cherry_(including_sour)___Powdery_mildew",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
        "Corn_(maize)___healthy", "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)", "Grape___healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
        "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
        "Potato___healthy", "Potato___Late_blight", "Raspberry___healthy", "Soybean___healthy",
        "Squash___Powdery_mildew", "Strawberry___healthy", "Strawberry___Leaf_scorch",
        "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy", "Tomato___Late_blight",
        "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
    ]

# Initialize Model
device = torch.device("cpu")
model = CustomLeafCNN(num_classes=len(CLASS_NAMES)).to(device)

WEIGHTS_PATH = "logs/custom_cnn.pth"
if os.path.exists(WEIGHTS_PATH):
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/")
def read_root():
    return {"message": "Crop Leaf-Disease Doctor API is Running"}

# Look for this section in app/api.py (around line 55 onwards)

@app.post("/predict")
async def predict(file: UploadFile = File(...), user_query: str = Form("")):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence_tensor, predicted_idx = torch.max(probabilities, 1)
        
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence = float(confidence_tensor.item())
    
    # --- THIS IS STEP 1 CODE ---
    # Retrieve chatbot response
    advisory_data = chatbot.get_advisory(predicted_class, user_query)
    
    # Format advisory response string cleanly
    if isinstance(advisory_data, dict):
        advisory_text = (
            advisory_data.get("message") or 
            advisory_data.get("answer") or 
            advisory_data.get("advisory") or 
            str(advisory_data)
        )
        citation_text = advisory_data.get("citation", "General Agricultural Extension Service")
    else:
        advisory_text = str(advisory_data)
        citation_text = "General Agricultural Extension Service"
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "advisory": advisory_text,
        "citation": citation_text
    }