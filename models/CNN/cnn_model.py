#Gage White
#COS 422 - Project: CNN Model 
#@Version: 14 April 2026

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from EMNISTDataset import EMNISTDataset
from torch.utils.data import DataLoader

class CNN(nn.Module):

    def __init__(self, f1, f2, hidden_fc, num_classes= 47):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, f1, kernel_size= 3) #layer 1
        self.pool = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(f1, f2, kernel_size= 3) #layer 2

        self.flattened_size = f2 * 5 * 5

        self.fc1 = nn.Linear(self.flattened_size, hidden_fc)
        self.fc2 = nn.Linear(hidden_fc, num_classes)

#forward pass
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        #flatten
        x = x.view(-1, self.flattened_size) 
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def train_func(model, train_loader, criterion, optimizer, device):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()               #clear gradients
            outputs = model(images)             #Forward pass
            loss = criterion(outputs, labels)   #Calc loss
            loss.backward()                     #backward pass
            optimizer.step()                    #update params

            #calculate batch accuracy
            total_loss += loss.item() * images.size(0) #loss for batch
            _, predicted = torch.max(outputs.data, 1) #prediction
            total += labels.size(0)
            correct += (predicted == labels).sum().item() #correct matches

        epoch_loss = total_loss  / total #average loss for dataset
        epoch_acc = correct / total      #accuracy for dataset
        return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device): #implement this next
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader: 
            images = images.float().to(device)
            labels = labels.to(device)

            #Forward pass
            outputs = model(images)
            #Compute loss
            loss = criterion(outputs, labels)
            #compute loss
            total_loss += loss.item() * images.size(0)
            #compute accuracy 
            _, predicted = torch.max(outputs, 1) 
            total += labels.size(0) 
            correct += (predicted == labels).sum().item()
            

    #calculate averages
    epoch_loss = total_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #set device
    print(f" Program is using {device.type.upper()}")
    #Hyperparameters
    f1 = 32            #num of Layer 1 filters
    f2 = 64            #num of Layer 2 filters
    hidden_dim = 128   #Hidden layer neurons
    learn_rate = 1e-3  #learn rate
    epochs = 45        #num of training loops
    batch_size = 64     #images per batch


    #dataset wrapping tensors
    train_dataset = EMNISTDataset('emnist-balanced-train.csv', split='train')
    val_dataset = EMNISTDataset('emnist-balanced-train.csv', split='validate')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size, shuffle=False)

  
  
    criterion = nn.CrossEntropyLoss() #loss function for multi-class
    

    model = CNN(f1,f2, hidden_dim).to(device)
    print(f" Model on GPU? {next(model.parameters()).is_cuda}")


    optimizer = optim.Adam(model.parameters(), learn_rate)



    train_accuracy = []
    val_accuracy = []

    for epoch in range(epochs):
        train_loss, train_acc = train_func(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_accuracy.append(train_acc)
        val_accuracy.append(val_acc)

        if epoch == 0 or (epoch +1) %10 == 0:
            print(
                f"Epoch {epoch + 1:03d} |" 
                f"Train Loss={train_loss:.4f} | Train Accuracy={train_acc:.4f} | Validation Loss={val_loss:.4f} | Validation Accuracy = {val_acc: .4f}")
            

    #plot training and validation accuracy
    plt.figure(figsize=(10,5))
    plt.plot(range(1, epochs + 1), train_accuracy, label='Training Accuracy', color= 'green')
    plt.plot(range(1, epochs + 1), val_accuracy, label='Validation Accuracy', color='red')
    plt.xlabel('Epoch') #     - x-axis: epochs
    plt.ylabel('Accuracy') #     - y-axis: accuracy
    plt.title("CNN Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()