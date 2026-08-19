import base64
import json

from lambda_func import lambda_handler


with open("test_images/dog.jpg", "rb") as image:
    encoded = base64.b64encode(
        image.read()
    ).decode()


event = {
    "body": json.dumps({
        "image": encoded
    })
}


response = lambda_handler(event, None)

print(response)