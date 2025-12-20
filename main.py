from fastapi import FastAPI, UploadFile, HTTPException, File
from contextlib import asynccontextmanager
from transformers import ViTForImageClassification, ViTImageProcessor
from peft import PeftModel
import torch
from PIL import Image
import io

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")

    base_model_name = "google/vit-base-patch16-224"
    adapter_path = "./model_adapters"

    try:
        processor = ViTImageProcessor.from_pretrained(base_model_name)

        base_model = ViTForImageClassification.from_pretrained(
            base_model_name,
            num_labels=15,
            ignore_mismatched_sizes=True
        )

        model = PeftModel.from_pretrained(base_model, adapter_path)

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

        print("Model sucessfully loaded!")

    except Exception as e:
        print(f"Failed to load model:{e}")
    yield

    ml_models.clear()
    print("Shutting down.....")

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/predict")
async def predict(file: UploadFile = File()):

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Input must be an image")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")

        processor = ml_models["processor"]
        model = ml_models["model"]

        id2label = ml_models["id2label"]

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            predicted_class_idx = logits.argmax(-1).item()

            probs = torch.softmax(logits, dim=1)
            confidence = probs[0][predicted_class_idx].item()

        return {
            "prediction": id2label[predicted_class_idx],
            "confidence": f"{confidence:2%}",
            "id": predicted_class_idx
        }
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
