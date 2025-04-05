import numpy as np 
import pandas as pd
import random
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Subset
from torchvision import models
import datetime
import os
import time
from tqdm import tqdm
import threading
from IPython import display
from IPython.display import clear_output
from sklearn.metrics import confusion_matrix
import seaborn as sns


best_model = None
best_val_accuracy = 0.0

# Nihat - 21 Ocak - 101 class 
class CustomImageClassifier2(nn.Module):
   def __init__(self, num_classes):
       super(CustomImageClassifier2, self).__init__()
      
       # Load a pre-trained model (e.g., ResNet)
    #    self.model = models.resnet50(weights=True)
       self.model = models.resnet50(pretrained=True)

      
       # Freeze the parameters of the model
       for param in self.model.parameters():
           param.requires_grad = False


       # Assuming ResNet18 is used, the in_features for the first added linear layer
       in_features = self.model.fc.in_features


       # Replace the fully connected layer
       self.model.fc = nn.Sequential(
           nn.Linear(in_features, 1024),
           nn.ReLU(),
           nn.BatchNorm1d(1024),
           nn.Dropout(0.5),
           nn.Linear(1024, 512),
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
   

# ======================= TRAINING ======================= #

def main():

    # ======================= PYTHON SETTINGS ======================= #
    # =======================   GPU or CPU    ======================= #

    device = torch.device("cpu")
    # torch.multiprocessing.set_start_method('file_system')

    # Check if GPU is available -> CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(0))

    # Apple Silicon GPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        out = torch.ones(1, device=device)
        print (out)
        print ("MPS device found. - Apple Silicon GPU")
    else:
        print ("MPS device not found.")


    print("Device:", device)

    # =======================   Ranom Seeds   ======================= #
    # Set random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # =============================================================== #


    # ======================= NORMALIZE PARAMS ======================= #
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to a consistent size
        transforms.ToTensor(),          # Convert images to tensors
        transforms.Normalize(           # Normalize the images
            # average values of the red, green, 
            # and blue channels across all images in the ImageNet dataset.
            mean=[0.485, 0.456, 0.406], 
            # standard deviation of the red, green, and blue 
            # channels across all images in the ImageNet dataset.
            std=[0.229, 0.224, 0.225]   
        )
    ])

    # ======================= DATA AUGMENTATION ======================= #
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),            # Resize images to a consistent size
        transforms.RandomHorizontalFlip(),        # Random horizontal flipping
        transforms.RandomRotation(15),            # Random rotation by 10 degrees
        transforms.ColorJitter(brightness=0.2,    # Adjust brightness, contrast, saturation, and hue
                            contrast=0.2,
                            saturation=0.2,
                            hue=0.1),
        transforms.ToTensor(),                    # Convert images to tensors
        transforms.Normalize(                     # Normalize the images
            mean=[0.485, 0.456, 0.406],           # Mean for ImageNet dataset
            std=[0.229, 0.224, 0.225]             # Standard deviation for ImageNet dataset
        )
    ])

    # ======================= LOAD MAIN DATASET ======================= #
    # Download and load the Food101 dataset
    train_dataset = torchvision.datasets.Food101(
        root='./data',                 # Directory to save the downloaded data
        split='train',
        download=True,                 # Download the data if not present
        transform=train_transform            # Apply the train_transform to the data
    )

    test_dataset = torchvision.datasets.Food101(
        root='./data',                 # Directory to save the downloaded data
        split='test',
        download=True,                 # Download the data if not present
        transform=transform            # Apply the transform to the data
    )


    # Get a subset with the first n classes
    n = 101  # For example, get the first 5 classes
    subset_train = train_dataset
    subset_test = test_dataset


    # Set other hyperparameters and initialize KFold
    num_epochs = 2
    batch_size = 32
    num_classes = 101
    learning_rate = 0.001
    num_folds = 5
    patience = 5  # Early stopping patience
    seed = 42
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)


    global best_model
    global best_val_accuracy


    # Split dataset into 80% training and 20% validation
    train_size = int(0.8 * len(subset_train))
    val_size = len(subset_train) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(subset_train, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)

    # Initialize model, loss function, and optimizer
    model = CustomImageClassifier2(num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_model = None
    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    early_stopping_counter = 0

    # Lists to keep track of losses and other metrics
    training_losses = []
    validation_losses = []

    # Training loop with Early Stopping
    for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast("mps"):
                outputs = model(images.to(device))
                loss = criterion(outputs, labels.to(device))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        average_training_loss = running_loss / num_batches
        training_losses.append(average_training_loss)

        # Validation loop
        model.eval()
        total_correct = 0
        total_samples = 0
        val_running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.cuda.amp.autocast("mps"), torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.view(-1).cpu().numpy())
                all_labels.extend(labels.view(-1).cpu().numpy())

        val_accuracy = total_correct / total_samples
        average_val_loss = val_running_loss / len(val_loader)
        validation_losses.append(average_val_loss)

        # Early Stopping Check
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            best_model = model.state_dict()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break

        # Plotting and printing
        clear_output(wait=True)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epoch + 2), training_losses, label='Training Loss')
        plt.plot(range(1, epoch + 2), validation_losses, label='Validation Loss')
        plt.title(f'Training and Validation Losses (Up to Epoch {epoch + 1})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {average_training_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {average_val_loss:.4f}")

    # Plot Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Load the best model for testing
    model.load_state_dict(best_model)
    model.eval()

    # Define the data loader for testing
    test_loader = DataLoader(subset_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)

    # Testing loop
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels.to(device)).sum().item()

    test_accuracy = total_correct / total_samples
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Plotting training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(training_losses)+1), training_losses, label='Training Loss')
    plt.plot(range(1, len(validation_losses)+1), validation_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Save the best model
    torch.save(best_model, 'bestmodel.pt')
    print("Done!")


if __name__ == "__main__":
    # Start time get
    start_time = time.time()
    # Print the start time
    print("Start time:", datetime.datetime.now())

    main()
    # End time get
    end_time = time.time()

    # Time elapsed
    elapsed_time_seconds = end_time - start_time
    elapsed_time = datetime.timedelta(seconds=elapsed_time_seconds)

    print("Time elapsed:", elapsed_time)
