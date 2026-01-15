# Waste Classification API ♻️

A high-performance Computer Vision API built to identify recyclable waste types. This project uses FastAPI to serve a Google ViT (Vision Transformer) model that has been fine-tuned using LoRA (Low-Rank Adaptation) to recognize 15 specific categories of waste.🚀

## 🚀 Features

- ** State-of-the-art Vision **: Uses google/vit-base-patch16-224 as the backbone.
- ** Efficient Fine-Tuning ** : Uses PEFT & LoRA adapters (~10MB) - instead of full model weights (~350MB).
- ** Fast Inference: ** Optimized for local CPU execution.
- ** Async API ** : Built on FastAPI for non-blocking performance.
- ** 30 Classes ** : Specifically trained to detect common household recyclables.

## 🛠️ Tech Stack

- ** Framework ** : FastAPI (Python)
- ** ML Engine ** : PyTorch
- ** Model Library ** : Hugging Face Transformers & PEFT
- ** Image Processing ** : Pillow (PIL)
- ** Server ** : Uvicorn📂
