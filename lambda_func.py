# import os
#
# os.environ["TORCH_HOME"] = "/tmp/torch"
# import json
# import base64
# import torch
# from torchvision import models
# from torchvision.models import ResNet18_Weights
# from PIL import Image
# from io import BytesIO
# import time
#
# start_time = time.time()
# cold_start = True
#
# # Load model once when Lambda starts
# weights = ResNet18_Weights.DEFAULT
# model = models.resnet18(weights=weights)
# model.eval()
#
# preprocess = weights.transforms()
# categories = weights.meta["categories"]
#
# model_load_time = time.time() - start_time
# print(f"MODEL LOAD TIME: {model_load_time:.4f} seconds")
#
#
# def lambda_handler(event, context):
#     global cold_start
#
#     request_start = time.time()
#
#     try:
#         # Get image from request
#         body = json.loads(event["body"])
#
#         image_data = body["image"]
#
#         # Decode image
#         image_bytes = base64.b64decode(image_data)
#
#         image = Image.open(
#             BytesIO(image_bytes)
#         ).convert("RGB")
#
#
#         # Preprocess image
#         input_tensor = preprocess(image).unsqueeze(0)
#
#
#         # Run inference
#         with torch.no_grad():
#             output = model(input_tensor)
#
#
#         prediction_index = output.argmax(dim=1).item()
#
#         prediction = categories[prediction_index]
#
#         inference_time = time.time() - request_start
#
#         print(f"INFERENCE TIME: {inference_time:.4f} seconds")
#         print(f"COLD START: {cold_start}")
#
#         cold_start = False
#
#         return {
#             "statusCode": 200,
#             "body": json.dumps({
#                 "prediction": prediction,
#                 "inference_time": inference_time,
#                 "cold_start": cold_start
#             })
#         }
#
#
#         return {
#             "statusCode": 200,
#             "body": json.dumps({
#                 "prediction": prediction
#             })
#         }
#
#
#     except Exception as e:
#
#         return {
#             "statusCode": 500,
#             "body": json.dumps({
#                 "error": str(e)
#             })
#         }
#
import os

# Allow PyTorch to write cache files in Lambda
os.environ["TORCH_HOME"] = "/tmp/torch"

import json
import base64
import time
import torch

from torchvision import models
from torchvision.models import ResNet18_Weights
from PIL import Image
from io import BytesIO


# ============================
# Model initialization
# ============================

cold_start = True

model_start_time = time.time()

weights = ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)
model.eval()

preprocess = weights.transforms()
categories = weights.meta["categories"]

model_load_time = time.time() - model_start_time

print(f"MODEL LOAD TIME: {model_load_time:.4f} seconds")


# ============================
# Lambda Handler
# ============================

def lambda_handler(event, context):

    global cold_start

    request_start_time = time.time()

    current_cold_start = cold_start

    try:

        # Parse request body
        body = json.loads(event["body"])

        image_data = body["image"]


        # Decode Base64 image
        image_bytes = base64.b64decode(image_data)

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")


        # Preprocess image
        input_tensor = preprocess(image).unsqueeze(0)


        # Run inference
        inference_start_time = time.time()

        with torch.no_grad():
            output = model(input_tensor)

        inference_time = time.time() - inference_start_time


        # Get prediction
        prediction_index = output.argmax(dim=1).item()

        prediction = categories[prediction_index]


        # Total request latency
        total_time = time.time() - request_start_time


        print(f"COLD START: {current_cold_start}")
        print(f"INFERENCE TIME: {inference_time:.4f} seconds")
        print(f"TOTAL REQUEST TIME: {total_time:.4f} seconds")


        # After first request, container is warm
        cold_start = False


        return {
            "statusCode": 200,
            "body": json.dumps({
                "prediction": prediction,
                "cold_start": current_cold_start,
                "model_load_time": model_load_time,
                "inference_time": inference_time,
                "total_time": total_time
            })
        }


    except Exception as e:

        print(f"ERROR: {str(e)}")

        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e)
            })
        }