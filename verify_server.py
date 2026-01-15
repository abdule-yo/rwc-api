from fastapi.testclient import TestClient
from main import app
from PIL import Image
import io

client = TestClient(app)

def test_predict_valid_image():
    print("Testing valid image...")
    img = Image.new('RGB', (224, 224), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
    )
    
    if response.status_code == 200:
        print("✅ Valid image test passed")
        print(response.json())
    else:
        print(f"❌ Valid image test failed: {response.status_code}")
        print(response.text)

def test_predict_invalid_image():
    print("\nTesting invalid image (text file masquerading as image)...")
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"this is not an image", "image/jpeg")}
    )
    
    if response.status_code == 400:
        print("✅ Invalid image test passed (Got 400 as expected)")
        print(response.json())
    else:
        print(f"❌ Invalid image test failed: Expected 400, got {response.status_code}")
        print(response.text)

def test_cors():
    print("\nTesting CORS...")
    response = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        }
    )
    if response.status_code == 200 and response.headers.get("access-control-allow-origin") == "*":
        print("✅ CORS test passed")
    else:
        print(f"❌ CORS test failed: {response.status_code}")
        print(response.headers)


if __name__ == "__main__":
    print("Starting tests...")
    with TestClient(app) as client:
        def test_predict_valid_image(c):
            print("Testing valid image...")
            img = Image.new('RGB', (224, 224), color = 'red')
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()

            response = c.post(
                "/predict",
                files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
            )
            
            if response.status_code == 200:
                print("✅ Valid image test passed")
            else:
                print(f"❌ Valid image test failed: {response.status_code}")
                print(response.text)

        def test_predict_invalid_image(c):
            print("\nTesting invalid image (text file masquerading as image)...")
            response = c.post(
                "/predict",
                files={"file": ("test.txt", b"this is not an image", "image/jpeg")}
            )
            
            if response.status_code == 400:
                print("✅ Invalid image test passed (Got 400 as expected)")
            else:
                print(f"❌ Invalid image test failed: Expected 400, got {response.status_code}")
                print(response.text)

        def test_cors(c):
            print("\nTesting CORS...")
            response = c.options(
                "/predict",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                }
            )
            allow_origin = response.headers.get("access-control-allow-origin")
            if response.status_code == 200 and (allow_origin == "*" or allow_origin == "http://localhost:3000"):
                print("✅ CORS test passed")
            else:
                print(f"❌ CORS test failed: {response.status_code}")
                print(response.headers)

        try:
            test_predict_valid_image(client)
            test_predict_invalid_image(client)
            test_cors(client)
        except Exception as e:
            print(f"Tests failed with exception: {e}")
