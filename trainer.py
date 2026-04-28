# STEP 1: Import & prepare the environment
'''
- Import required libraries
- Use get_args() to load hyperparameters (epochs, learning rate, weight decay, etc.)
'''
from args import get_args
import os
import torch
import torch.optim as optim
from utils import show_batch, plot_learning_curve      # NEW
import pandas as pd                                    # NEW


# STEP 2: Initialize Training function
'''
- Inputs: model, train_loader, val_loader, device (CPU/GPU)
- Move model to the selected device
'''
def train_model(model, train_loader, val_loader, device):
    args = get_args()
    model = model.to(device)
    
    # Initialize optimizer (Adam) and best validation loss tracker
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val_loss = float('inf')
    
    # Initialize epoch log                                                      # NEW
    history = {"epoch": [], "train_loss": [], "val_loss": []}
    
    # Loop over epochs (full passes over the dataset)
        # Iterate over batches from train_loader
        # Each batch contains:
        # - images: list of tensors
        # - targets: list of dictionaries (boxes, labels)
    for epoch in range(args.epochs):
        
        # Set model as training mode: use running_loss to sum loss up
        model.train()
        running_loss = 0.0
        
        for images, targets in train_loader:
            
            # Move images to device and convert to float32
            images = [image.to(device=device, dtype=torch.float32) for image in images]
            # Move annotations (bounding boxes and labels) to device
            targets = [
                {
                    'boxes': target['boxes'].to(device=device, dtype=torch.float32),
                    'labels': target['labels'].to(device=device, dtype=torch.int64)
                }
                for target in targets
            ]
            
            show_batch(images, targets, batch_idx=0)
            # Forward & Backward pass: 
            optimizer.zero_grad()
            
            # Model returns a dictionary of losses (classification, bbox, etc.)
            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())
            
            # Backpropagation and parameter update
            loss.backward()
            optimizer.step()
            
            # Accumulate training loss (scaled by batch size)
            running_loss += loss.item() * len(images)
            
        # Compute average training loss for the epoch & Run validation after each epoch
        train_epoch_loss = running_loss / len(train_loader.dataset)
        val_loss = validate_model(model, val_loader, device)
        
        # Save history                                                         # NEW
        history["epoch"].append(epoch + 1)                                     # NEW
        history["train_loss"].append(train_epoch_loss)                         # NEW
        history["val_loss"].append(val_loss)
            
        # Print training progress
        print(f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train loss: {train_epoch_loss:.4f} | "
            f"Val loss: {val_loss:.4f}")
        
        # Save CSV after every epoch                                           # NEW
        os.makedirs(args.out_dir, exist_ok=True)                               # NEW
        pd.DataFrame(history).to_csv(                                          # NEW
            os.path.join(args.out_dir, "training_log.csv"), index=False)
        
        # Save the best model (based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model.pth'))
            
    # Draw learning curve                                        # NEW
    plot_learning_curve(history, args.out_dir)                                        # NEW
    print(f"Log saved: {args.out_dir}/training_log.csv")                              # NEW
    print(f"Learning curve saved: {args.out_dir}/learning_curve.png")
    
    
# STEP 3: Initialize Validation function
'''
- Evaluate model performance on validation dataset
- No gradient computation (faster and saves memory)
'''
def validate_model(model, val_loader, device):
    # Set model to evaluation mode (disables dropout, batchnorm updates)
    model.train()
    val_loss_sum = 0.0
    val_count = 0

    # Disable gradient computation to save memory and let the model run faster
    with torch.no_grad():
        # Iterate over validation batches
        for images, targets in val_loader:
            # Move images to device
            images = [image.to(device=device, dtype=torch.float32) for image in images]
            
            # Move annotations to device
            targets = [
                {
                    'boxes': target['boxes'].to(device=device, dtype=torch.float32),
                    'labels': target['labels'].to(device=device, dtype=torch.int64)
                }
                for target in targets
            ]
            
            # Forward pass only (NO Backward)
            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())
            
            # Accumulate validation loss
            val_loss_sum += loss.item() * len(images)
            val_count += len(images)
        
        # Compute average validation loss
        val_epoch_loss = val_loss_sum / val_count

        return val_epoch_loss