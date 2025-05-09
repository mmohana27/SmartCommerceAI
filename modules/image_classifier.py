import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load pre-trained ResNet model
resnet = models.resnet50(pretrained=True)
resnet.eval()

# Image preprocessing steps
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Path to your product image
image_path = "product_images/item_1.jpg"  # Make sure this image exists

try:
    # Open and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        outputs = resnet(image_tensor)
        _, predicted = torch.max(outputs, 1)

    print(f"Predicted class for image {image_path}: {predicted.item()}")

except Exception as e:
    print(f"Error: {e}")
