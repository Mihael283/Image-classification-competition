# architecture.py
import torch
import torch.nn.functional as F
from dataset import ImagesDataset, cutmix
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from classes.simple_net import MyCNN
from logger import Logger
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.cuda.amp import autocast, GradScaler
import os
import matplotlib.pyplot as plt

torch.manual_seed(42)
torch.autograd.set_detect_anomaly(True)

os.makedirs('visualizations', exist_ok=True)

def visualize_batch(inputs, labels, epoch, batch_idx):
    plt.figure(figsize=(20, 10))
    for i in range(min(32, inputs.shape[0])):
        plt.subplot(4, 8, i+1)
        plt.imshow(inputs[i].squeeze().cpu().numpy(), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'visualizations/batch_epoch_{epoch}_batch_{batch_idx}.png')
    plt.close()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_classes = 20
batch_size = 64
num_epochs = 400
learning_rate = 0.0001 

train_dir = 'training_data'
full_dataset = ImagesDataset(train_dir, is_train=True)
classnames_to_ids = full_dataset.classnames_to_ids
ids_to_classnames = {v: k for k, v in classnames_to_ids.items()}

train_size = int(0.70 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

class SubsetDataset(ImagesDataset):
    def __init__(self, subset, is_train):
        self.subset = subset
        self.is_train = is_train

    def __getitem__(self, idx):
        return self.subset[idx]

    def __len__(self):
        return len(self.subset)

train_dataset = SubsetDataset(train_subset, is_train=True)
val_dataset = SubsetDataset(val_subset, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

model = MyCNN().to(device)
logger_info = f"Simple_net_test_loader_aug_shuffle_64_0.0001_70_split_no_cutmix"
logger = Logger(logger_info, ids_to_classnames)
from collections import Counter


# Calculate class weights
full_dataset.return_image = False
labels = [data for data in full_dataset]  
full_dataset.return_image = True  

class_counts = Counter(labels)
total_samples = sum(class_counts.values())
class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}

print(class_counts.items())
class_weights = torch.tensor([class_weights[cls] for cls in range(num_classes)], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

train_losses = []
val_losses = []
best_val_acc = 0
patience = 50
no_improvement = 0

class_correct = [0.] * num_classes
class_total = [0.] * num_classes
all_preds = []
all_labels = []

scaler = GradScaler()

for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    val_loss = 0.0
    val_acc = 0.0

    # Training
    model.train()
    for batch_idx, data in enumerate(tqdm(train_loader)):
        inputs, labels = data[0].to(device), data[1].to(device)

        # Apply CutMix with 50% probability
        #if np.random.rand() < 0.6:
        #    inputs, labels_a, labels_b, lam = cutmix((inputs, labels))
        #    outputs = model(inputs)
        #    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        #else:

        if batch_idx % 200 == 0:
            visualize_batch(inputs, labels, epoch, batch_idx)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        
        if not isinstance(labels, tuple):
            train_acc += torch.sum(preds == labels.data)

    # Validation
    model.eval()
    with torch.no_grad():
        for data in tqdm(val_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            val_acc += torch.sum(preds == labels.data)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct = preds.eq(labels.data.view_as(preds))
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += correct[i].item()
                class_total[label] += 1


    print('\nValidation Accuracy for each class:')

    conf_matrix = confusion_matrix(all_labels, all_preds)

    per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    print("\nPer-class accuracy:")
    for i, acc in enumerate(per_class_accuracy):
        class_name = ids_to_classnames[i]
        print(f"{class_name} (Class {i}): {acc:.2f}")

    print("Current learning rate: ", optimizer.param_groups[0]['lr'])
            
    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    val_acc /= len(val_loader.dataset)

    scheduler.step()

    logger.log_epoch(epoch+1, train_loss, train_acc, val_loss, val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improvement = 0
        torch.save(model.state_dict(), f'model_{logger_info}.pth')
        logger.log_best_model(epoch+1, val_acc)
    
    if no_improvement >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break
