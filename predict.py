import torch
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
from load_model import load_model

IMG_SIZE = 256

model, classes, device = load_model()

def smart_roi_crop(img):
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))

    if len(coords) > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        if y1 > y0 and x1 > x0:
            img_np = img_np[y0:y1, x0:x1]

    return Image.fromarray(img_np)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

def predict(image):
    img = smart_roi_crop(image)
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]

        idx = torch.argmax(probs).item()

    return classes[idx], round(probs[idx].item() * 100, 2)