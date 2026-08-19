import base64
import json
import requests

with open("test_images/dog.jpg", "rb") as f:
    image = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "body": json.dumps({
        "image": image
    })
}

response = requests.post(
    "http://localhost:9000/2015-03-31/functions/function/invocations",
    json=payload
)

print(response.status_code)
print(response.text)