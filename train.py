import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler
import numpy as np

class EarlyStopping:
    def __init__(self, patience=35, min_delta=0.001, verbose=True):
        """
        Args:
            patience (int): How many epochs to wait before stopping when no improvement
            min_delta (float): Minimum change to qualify as an improvement
            verbose (bool): If True, prints a message for each validation improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model, epoch, checkpoint_path):
        """
        Args:
            val_loss (float): Validation loss from current epoch
            model (nn.Module): Model to save if improvement is observed
            epoch (int): Current epoch number
            checkpoint_path (str): Path to save the checkpoint
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, checkpoint_path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, checkpoint_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, checkpoint_path):
        """Save model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss
        }, checkpoint_path)
        self.val_loss_min = val_loss

def train_epoch(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc="Training")
    for joints, bones, velocities, labels in progress_bar:
        # Di chuyển dữ liệu đến thiết bị
        joints = joints.to(device)
        bones = bones.to(device)
        velocities = velocities.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Huấn luyện với độ chính xác hỗn hợp
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(joints, bones, velocities)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Theo dõi các metrics
        total_loss += loss.item() * joints.size(0)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # Cập nhật thanh tiến trình
        progress_bar.set_postfix(
            loss=f"{total_loss/total_samples:.4f}",
            accuracy=f"{100*total_correct/total_samples:.2f}%"
        )

    return total_loss / total_samples, total_correct / total_samples

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(test_loader, desc="Evaluating")
    with torch.no_grad():
        for joints, bones, velocities, labels in progress_bar:
            # Di chuyển dữ liệu đến thiết bị
            joints = joints.to(device)
            bones = bones.to(device)
            velocities = velocities.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(joints, bones, velocities)
            loss = criterion(outputs, labels)

            # Theo dõi các metrics
            total_loss += loss.item() * joints.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Thu thập dự đoán
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Cập nhật thanh tiến trình
            progress_bar.set_postfix(
                loss=f"{total_loss/total_samples:.4f}",
                accuracy=f"{100*total_correct/total_samples:.2f}%"
            )

    return total_loss / total_samples, total_correct / total_samples, all_preds, all_labels

def save_checkpoint(model, optimizer, epoch, training_history, best_accuracy, best_f1, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_history': training_history,
        'best_accuracy': best_accuracy,
        'best_f1': best_f1
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return (
        checkpoint['epoch'],
        checkpoint['training_history'],
        checkpoint['best_accuracy'],
        checkpoint['best_f1']
    )