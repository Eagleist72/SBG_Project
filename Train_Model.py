"""
Neural Network Trainer (PyTorch Edition)
Description: Trains a Multi-Layer Perceptron (Deep Learning) model using the collected game data.
Now supports CUDA, Checkpointing, and Progress Bars.
Configurable via YAML.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm
from NeuralNet import BoardEvaluator
from ConfigLoader import load_config

# Load Configuration
config = load_config()

# Hyperparameters from Config
BATCH_SIZE = config['training']['batch_size']
LEARNING_RATE = config['training']['learning_rate']
EPOCHS = config['training']['epochs']
DEVICE_CONFIG = config['training']['device']
FILE_NAME = config['data']['file_name']
CHECKPOINT_PATH = config['paths']['training_checkpoint']
MODEL_SAVE_PATH = config['paths']['model_checkpoint']

# Check for CUDA
if DEVICE_CONFIG == 'auto':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device(DEVICE_CONFIG)
print(f"Using device: {device}")

class GameDataset(Dataset):
    def __init__(self, csv_file):
        print("Loading dataset...")
        try:
            self.df = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"Error: '{csv_file}' not found. Run collect_data.py first!")
            exit()
        
        # Features (X) and Labels (y)
        self.X = self.df.iloc[:, :-1].values.astype(np.float32)
        self.y = self.df.iloc[:, -1].values.astype(np.float32).reshape(-1, 1)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_brain(resume=False, checkpoint_path=CHECKPOINT_PATH):
    # Load Data
    full_dataset = GameDataset(FILE_NAME)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    # Initialize Model
    # Note: BoardEvaluator now inherently reads input/hidden sizes from config (via NeuralNet.py)
    model = BoardEvaluator().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    start_epoch = 0
    
    # Resume from checkpoint
    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Checkpoint logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save best model for game
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> New best model saved ({MODEL_SAVE_PATH})!")
        
        # Save regular checkpoint
        if os.path.exists(checkpoint_path):
            import shutil
            backup_path = checkpoint_path + ".bak"
            shutil.copy(checkpoint_path, backup_path)
            print(f"  -> Backup saved ({backup_path})")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss
        }, checkpoint_path)

    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    args = parser.parse_args()
    
    train_brain(resume=args.resume)