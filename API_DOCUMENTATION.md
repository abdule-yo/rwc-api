# RWC API Documentation

Base URL: `http://localhost:8000` (Default FastAPI port)

## Endpoints

### 1. Health Check
**GET** `/`

Returns a simple greeting to verify the server is running.

- **Response**: `200 OK`
```json
{
  "Hello": "World"
}
```

### 2. Predict Waste Class
**POST** `/predict`

Classifies an uploaded image into one of 15 recyclable waste categories.

- **Headers**:
  - `Content-Type`: `multipart/form-data`

- **Body (form-data)**:
  - `file`: The image file (binary). Must be a valid image format (JPEG, PNG).

- **Response**: `200 OK`
```json
{
  "label": "Cardsboard Boxes",
  "confidence": 0.985,
  "id": 3
}
```
  - `label`: The predicted class name (human readable, title case).
  - `confidence`: Float value between 0 and 1 indicating model certainty.
  - `id`: The internal integer ID of the class.

- **Errors**:
  - `400 Bad Request`: If the file is not an image or is corrupt.
  - `503 Service Unavailable`: If the ML models failed to load.

### Example Usage

#### cURL
```bash
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:8000/predict
```

#### JavaScript (Fetch)
```javascript
const formData = new FormData();
formData.append("file", imageFile); // imageFile is a File object

const response = await fetch("http://localhost:8000/predict", {
  method: "POST",
  body: formData,
});

if (response.ok) {
  const result = await response.json();
  console.log("Prediction:", result);
} else {
  console.error("Error:", response.statusText);
}
```

## Waste Categories (ID Mapping)
0. Aerosol Cans
1. Aluminum Food Cans
2. Aluminum Soda Cans
3. Cardboard Boxes
4. Cardboard Packaging
5. Clothing
6. Coffee Grounds
7. Disposable Plastic Cutlery
8. Eggshells
9. Food Waste
10. Glass Beverage Bottles
11. Glass Cosmetic Containers
12. Glass Food Jars
13. Magazines
14. Newspaper
