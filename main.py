from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from transformers import ViTForImageClassification, ViTImageProcessor
from peft import PeftModel
from PIL import Image, UnidentifiedImageError
import io
import torch
import math

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    base_model_name = "google/vit-base-patch16-224"
    adapter_path = "./model_adapters"
    
    device = torch.device("mps") if torch.backends.mps.is_available() else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    try:
        processor = ViTImageProcessor.from_pretrained(base_model_name)

        base_model = ViTForImageClassification.from_pretrained(
            base_model_name,
            num_labels=15,
            ignore_mismatched_sizes=True
        )

        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.to(device)
        model.eval()

        ml_models["id2label"] = {
            0: "aerosol_cans",
            1: "aluminum_food_cans",
            2: "aluminum_soda_cans",
            3: "cardboard_boxes",
            4: "cardboard_packaging",
            5: "clothing",
            6: "coffee_grounds",
            7: "disposable_plastic_cutlery",
            8: "eggshells",
            9: "food_waste",
            10: "glass_beverage_bottles",
            11: "glass_cosmetic_containers",
            12: "glass_food_jars",
            13: "magazines",
            14: "newspaper"
        }

        ml_models["processor"] = processor
        ml_models["model"] = model
        ml_models["device"] = device

    except Exception:
        pass
    yield

    ml_models.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/predict")
async def predict(file: UploadFile = File()):
    if "model" not in ml_models or "processor" not in ml_models:
         raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Input must be an image/jpeg or image/png")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")

        processor = ml_models["processor"]
        model = ml_models["model"]
        device = ml_models["device"]
        id2label = ml_models["id2label"]

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            predicted_class_idx = logits.argmax(-1).item()

            probs = torch.softmax(logits, dim=1)
            confidence = probs[0][predicted_class_idx].item()

        return {
            "label": id2label[predicted_class_idx].replace("_", " ").title(),
            "confidence": confidence,
            "id": predicted_class_idx
        }
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file. Could not identify image format.")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
