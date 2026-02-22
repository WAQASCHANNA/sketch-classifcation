import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
import argparse

from dataset import SketchDataset, get_transforms, download_and_extract_data
from model import SketchCNN

def train_model(data_dir, num_epochs=10, batch_size=64, learning_rate=1e-3, device="cpu"):
    # Ensure data is downloaded
    download_and_extract_data(data_dir)
    
    train_transform, val_test_transform = get_transforms()
    
    train_dataset = SketchDataset(data_dir, split="train", transform=train_transform)
    valid_dataset = SketchDataset(data_dir, split="valid", transform=val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    model = SketchCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    best_val_f1 = 0.0
    
    print(f"Training on {device}...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in train_loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_loop.set_postfix(loss=loss.item())
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss /= len(valid_dataset)
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Macro-F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_model.pth")
            print(f"--> Saved new best model with Val Macro-F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(data_dir="sketch_clf", num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, device=device)
