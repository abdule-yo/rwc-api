# Waste Classification API ♻️

A high-performance Computer Vision API built to identify recyclable waste types. This project uses FastAPI to serve a Google ViT (Vision Transformer) model that has been fine-tuned using LoRA (Low-Rank Adaptation) to recognize 15 specific categories of waste.🚀

## 🚀 Features

- ** State-of-the-art Vision **: Uses google/vit-base-patch16-224 as the backbone.
- ** Efficient Fine-Tuning ** : Uses PEFT & LoRA adapters (~10MB) - instead of full model weights (~350MB).
- ** Fast Inference: ** Optimized for local CPU execution.
- ** Async API ** : Built on FastAPI for non-blocking performance.
- ** 15 Classes ** : Specifically trained to detect common household recyclables.

## 🛠️ Tech Stack

- ** Framework ** : FastAPI (Python)
- ** ML Engine ** : PyTorch
- ** Model Library ** : Hugging Face Transformers & PEFT
- ** Image Processing ** : Pillow (PIL)
- ** Server ** : Uvicorn📂

## Project Structure.

├── main.py # The FastAPI application logic
├── requirements.txt # Python dependencies
├── lora_adapters/ # FOLDER containing your trained weights
│ ├── adapter_config.json
│ ├── adapter_model.safetensors
│ └── ...
└── venv/ # Virtual Environment (Git ignored)

## ⚡ Installation & Setup

1. Clone the repositorygit clone [https://github.com/YOUR_USERNAME/waste-classification-api.git](https://github.com/YOUR_USERNAME/waste-classification-api.git)

cd waste-classification-api

2. Create a Virtual Environment# Mac/Linux
   python3 -m venv venv
   source venv/bin/activate

# Windows

python -m venv venv
venv\Scripts\activate
Install Dependenciespip install -r requirements.txt
Setup Model WeightsEnsure your trained LoRA files are inside a folder named lora_adapters in the root directory.(Note: The base Google model will download automatically on the first run).🏃‍♂️ Running the ServerStart the development server:fastapi dev main.py

# OR

uvicorn main:app --reload
The server will start at http://127.0.0.1:8000.🔌 API DocumentationPOST /predictUpload an image file to get a classification.Request:Content-Type: multipart/form-dataBody: file (Binary Image Data - JPG/PNG)Example Request (Curl):curl -X POST "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)" \
 -H "accept: application/json" \
 -H "Content-Type: multipart/form-data" \
 -F "file=@/path/to/your/image.jpg"
Success Response (200 OK):{
"prediction": "aluminum_soda_cans",
"confidence": "98.50%",
"id": 2
}
