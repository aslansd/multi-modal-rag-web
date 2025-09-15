import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from sentence_transformers import SentenceTransformer

# Text embedding model
text_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_text_embedding(text):
    return text_model.encode(text, normalize_embeddings=True)

# Image embedding model (CLIP or ResNet as fallback)
resnet = models.resnet50(pretrained=True)
resnet.eval()
transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

def get_image_embedding(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = resnet(img_tensor)
    return features.squeeze().numpy()