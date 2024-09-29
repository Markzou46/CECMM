# -*- coding: utf-8 -*-
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

from data_generator import Train_Data_Generator, Test_Data_Generator
from augment import Augment
from net import Net

EXP = 'CEMm_EXP'


# Set random seed for reproducibility
def set_seed(seed=48):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed()

# Parameters
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
num_epochs = 100
batch_size = 16
val_interval = 1

# Dataset paths
df_path = './data/train+valid.csv'
df_root = './data/dataset'
df = pd.read_csv(df_path, encoding='utf-8')
df_train, df_valid = train_test_split(df, test_size=0.3, random_state=seed)

# Data augmentation and preprocessing
Aug = Augment()
train_transform = transforms.Compose([
    transforms.Lambda(lambda x: Aug.flip(x, lr=True, ud=True)),
    transforms.Lambda(lambda x: Aug.z_score_normalization(x)),
])
valid_transform = transforms.Compose([
    transforms.Lambda(lambda x: Aug.z_score_normalization(x)),
])

# Data loading
kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
train_generator64 = Train_Data_Generator(df_train, df_root, data_transform=train_transform)
train_loader64 = DataLoader(train_generator64, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
test_generator64 = Test_Data_Generator(df_valid, df_root, data_transform=valid_transform)
test_loader64 = DataLoader(test_generator64, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)

# Model setup
model = Net(frame_num=160).to(device)
LR = 0.0001

# Loss function and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Training and validation
best_train_acc = 0
best_metrics = {'acc': 0, 'auc': 0, 'f1': 0, 'recall': 0, 'epoch': 0}
best_auc_metrics = {'auc': 0, 'acc': 0, 'f1': 0, 'recall': 0, 'epoch': 0, 'fpr': [], 'tpr': [], 'thresholds': []}
metrics_history = {'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': [], 'auc': [], 'f1': [],
                   'recall': []}

for epoch in range(num_epochs):
    model.train()
    loss_epoch, correct, total = 0, 0, 0
    y_true, y_pred, outputs_all, patient_ids = [], [], [], []

    for data, target, patient_id in train_loader64:
        data = [d.to(device) for d in data]
        target = target.to(device).view(-1)

        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += target.size(0)
        _, predicted = torch.max(output.data, 1)
        correct += predicted.eq(target).sum().item()
        loss_epoch += loss.item() * target.size(0)

        y_true.extend(target.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        outputs_all.extend(output.detach().cpu().numpy())
        patient_ids.extend(patient_id)

    train_loss = loss_epoch / len(train_loader64.dataset)
    train_accuracy = correct / total
    metrics_history['train_loss'].append(train_loss)
    metrics_history['train_acc'].append(train_accuracy)

    # Calculate AUC
    outputs_all_np = np.vstack(outputs_all)
    y_scores_np = torch.softmax(torch.tensor(outputs_all_np), dim=1).numpy()[:, 1]
    auc = metrics.roc_auc_score(np.array(y_true), y_scores_np)

    # Print training stats
    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {train_loss:.4f} Acc: {train_accuracy:.2%} AUC: {auc:.2%}")

    # Update scheduler
    scheduler.step()

    # Validation at specified intervals
    if (epoch + 1) % val_interval == 0:
        model.eval()
        valid_loss_epoch, correct_val, total_val = 0, 0, 0
        y_true_val, y_pred_val, outputs_all_val, patient_ids_val = [], [], [], []

        with torch.no_grad():
            for data, target, patient_id in test_loader64:
                data = [d.to(device) for d in data]
                target = target.to(device).view(-1)

                output = model(data)
                loss = criterion(output, target)

                total_val += target.size(0)
                _, predicted = torch.max(output.data, 1)
                correct_val += predicted.eq(target).sum().item()
                valid_loss_epoch += loss.item() * target.size(0)

                y_true_val.extend(target.cpu().numpy())
                y_pred_val.extend(predicted.cpu().numpy())
                outputs_all_val.extend(output.detach().cpu().numpy())
                patient_ids_val.extend(patient_id)

        valid_loss = valid_loss_epoch / len(test_loader64.dataset)
        valid_accuracy = correct_val / total_val
        metrics_history['valid_loss'].append(valid_loss)
        metrics_history['valid_acc'].append(valid_accuracy)

        # Calculate AUC for validation
        outputs_all_val_np = np.vstack(outputs_all_val)
        y_scores_val_np = torch.softmax(torch.tensor(outputs_all_val_np), dim=1).numpy()[:, 1]
        auc_val = metrics.roc_auc_score(np.array(y_true_val), y_scores_val_np)

        print(
            f"Validation: Epoch [{epoch + 1}/{num_epochs}] Loss: {valid_loss:.4f} Acc: {valid_accuracy:.2%} AUC: {auc_val:.2%}")

        # Save best models based on validation accuracy
        if valid_accuracy >= best_metrics['acc']:
            best_metrics.update({'acc': valid_accuracy, 'auc': auc_val, 'epoch': epoch})
            torch.save(model.state_dict(), f"{EXP}_best_model.pth")

# Plotting results
train_loss = metrics_history['train_loss']
valid_loss = metrics_history['valid_loss']

sns.set(style="whitegrid")
plt.figure(figsize=(10, 8))
plt.plot(train_loss, 'b-o', label='Train Loss')
plt.plot(valid_loss, 'r-s', label='Valid Loss')
plt.title('Training and Validation Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f'{EXP}_loss.png')
plt.close()