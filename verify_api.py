import requests
import numpy as np
from PIL import Image
import io

def test_api():
    url = "http://localhost:8000/predict"
    
    # Test 1: Valid Digit
    print("Testing Valid Digit...")
    img_array = np.zeros((28, 28), dtype=np.uint8)
    img_array[5:25, 12:16] = 255 # Draw a line
    img_array_frontend = 255 - img_array # White background
    
    img = Image.fromarray(img_array_frontend.astype(np.uint8))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    files = {'file': ('digit.png', img_byte_arr, 'image/png')}
    response = requests.post(url, files=files)
    print(f"Response: {response.json()}")
    
    # Test 2: Empty Input
    print("\nTesting Empty Input...")
    img_array_empty = np.ones((28, 28), dtype=np.uint8) * 255 # All white
    
    img_empty = Image.fromarray(img_array_empty)
    img_byte_arr_empty = io.BytesIO()
    img_empty.save(img_byte_arr_empty, format='PNG')
    img_byte_arr_empty.seek(0)
    
    files_empty = {'file': ('empty.png', img_byte_arr_empty, 'image/png')}
    response_empty = requests.post(url, files=files_empty)
    print(f"Response: {response_empty.json()}")
    
    if 'digit' in response.json() and 'error' in response_empty.json():
        print("\nVerification PASSED")
    else:
        print("\nVerification FAILED")

if __name__ == "__main__":
    test_api()
