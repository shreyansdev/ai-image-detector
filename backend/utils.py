import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

def get_resnet50_model(num_classes=2, checkpoint_path=None):
    """
    Initializes a ResNet-50 model. 
    If checkpoint_path is provided, it loads the fine-tuned weights.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Modify the final fully connected layer for binary classification (AI vs Real)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    if checkpoint_path:
        # Map location 'cpu' for safety, can be changed to 'cuda' if GPU available
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

# Standard ResNet transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(model, image_bytes):
    """
    Performs inference on the uploaded image.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)
    
    # Simple softmax for probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    ai_prob = probabilities[1].item() # Assume index 1 is 'AI Generated'
    
    return ai_prob
