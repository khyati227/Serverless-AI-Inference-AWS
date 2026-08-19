import torch
from torchvision import models, transforms
from PIL import Image

# Load pretrained ResNet18
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()

# Image preprocessing
preprocess = weights.transforms()

# Load image
image = Image.open("test_images/dog.jpg").convert("RGB")
input_tensor = preprocess(image).unsqueeze(0)

# Perform inference
with torch.no_grad():
    output = model(input_tensor)

# Get prediction
prediction = output.argmax(dim=1).item()

# Get class labels
categories = weights.meta["categories"]

print("Prediction:", categories[prediction])