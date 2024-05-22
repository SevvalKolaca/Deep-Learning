import torch
from flask import Flask, request
from PIL import Image
import os
import cv2
import numpy as np
from torchvision import transforms
from model import EfficientNet

app = Flask(__name__)

widthMult, depthMult, dropoutRate, numClasses = 1.0, 1.0, 0.2, 5

model = EfficientNet(widthMult, depthMult, dropoutRate, numClasses)
model.load_state_dict(torch.load('./model.pth', map_location=torch.device('cpu')))
model.eval()

device = torch.device("cpu")
model.to(device)

label_mapping = {0: 'Elephant', 1: 'Chicken', 2: 'Cat', 3: 'Sheep', 4: 'Chipmunk'}


def preprocess_image(image):
    # Convert PIL Image to numpy array
    image = np.array(image)

    # Convert from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image
    image = cv2.resize(image, (224, 224))

    # Convert the numpy array to a PyTorch tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part in the request'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    try:
        filename = os.path.basename(file.filename)
        save_path = os.path.join("./uploads", filename)
        file.save(save_path)
        image = Image.open(save_path)
        inputs = preprocess_image(image)

        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predicted_label = predicted.item()
            print(predicted_label)

        actual_label = label_mapping[predicted_label]
        response = f'Image class is {actual_label}'
        return response
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    os.makedirs('./uploads', exist_ok=True)
    app.run(debug=True, port=5000)
