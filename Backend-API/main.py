from fastapi import UploadFile, BackgroundTasks
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from starlette.requests import Request
from starlette.background import BackgroundTasks
from typing import List


import torch
from torchvision import transforms
from PIL import Image

import torch.nn as nn
from torchvision import models

from class_mapping import class_index_to_name # class_index_to_name is a dictionary

class CustomImageClassifier(nn.Module):
   def __init__(self, num_classes):
       super(CustomImageClassifier, self).__init__()
      
       # Load a pre-trained model (e.g., ResNet)
       self.model = models.resnet18(pretrained=True)
      
       # Freeze the parameters of the model
       for param in self.model.parameters():
           param.requires_grad = False


       # Assuming ResNet18 is used, the in_features for the first added linear layer
       in_features = self.model.fc.in_features


       # Replace the fully connected layer
       self.model.fc = nn.Sequential(
           nn.Linear(in_features, 512),
           nn.ReLU(),
           nn.BatchNorm1d(512),
           nn.Dropout(0.5),
           nn.Linear(512, 256),
           nn.ReLU(),
           nn.BatchNorm1d(256),
           nn.Dropout(0.5),
           nn.Linear(256, num_classes)
       )


   def forward(self, x):
       return self.model(x)

app = FastAPI()

model_path = "bestmodel.pt"

# Initialize your custom model
model = CustomImageClassifier(num_classes=101) 

# Load the trained model state
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Define the image preprocessing function
def preprocess_image(file: UploadFile):
    image = Image.open(file.file)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Define the prediction function
def predict_image(file: UploadFile):
    image = preprocess_image(file)

    # Perform prediction using your model
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        prediction = int(predicted)
    
    print("Prediction: ", prediction)
    print("Class name: ", class_index_to_name[prediction])
    return prediction

# FastAPI endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    '''
    This endpoint performs the following steps:
    1. Preprocess the input image
    2. Run the prediction
    3. Return the predicted class name
    '''

    # Run the prediction in the background
    prediction = predict_image(file)

    nameOfFood = class_index_to_name[prediction]

    # Return prediction in the response
    return JSONResponse(content={"message": "Image prediction completed.", "prediction": nameOfFood})

# Online Check - STATUS Check
@app.get("/status")
def read_status():
    '''
    This endpoint returns the status of the api
    '''
    return {"status": "online"}

@app.get("/")
def read_root():
    '''
    This endpoint returns the names of the authors of the API && Model
    '''
    return {"Food101 API": "Authors: Efe Şirin, Nihat Aksu"}

# Get the list of class names
@app.get("/classes/", response_model=List[str])
async def get_classes():
    '''
    This endpoint returns the list of class names
    '''
    # Return the list of class names
    return list(class_index_to_name.values())