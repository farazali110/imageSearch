import requests

url = "https://imagesearch-sd80.onrender.com/"  # Replace with your Render URL
file_path = "test.jpg"  # Path to your local image

with open(file_path, "rb") as image_file:
    files = {"file": image_file}
    response = requests.post(url, files=files)

print(response.status_code)
print(response.json())
