# core/ml_models/image_cnn.py

import torch
from torchvision import models, transforms
from PIL import Image

model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

def classify_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    output = model(input_tensor)
    _, predicted = output.max(1)
    class_names = models.EfficientNet_B7_Weights.DEFAULT.meta['categories']
    return class_names[predicted.item()]
